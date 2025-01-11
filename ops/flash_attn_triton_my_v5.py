# 考虑对角线

import math

import torch
import triton
import triton.language as tl

total_sparsity = 0
total_num = 0
last_q = 64
arange = torch.arange(last_q, device="cuda")
LAST_Q_MASK = arange[None, None, :, None] >= arange[None, None, None, :]

@triton.jit
def _fwd_kernel(
    Q,
    K,
    softmax_scale,

    pIndices, #
    pscale, psum, #

    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,

    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,

    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,

    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,

    range_h: tl.constexpr, #
    row_blks: tl.constexpr, #
):

    pid = tl.program_id(0)
    stride_m = range_h // BLOCK_M
    stride_n = range_h // BLOCK_N # 每个小块包含多少最小块
    num_m = tl.floor((tl.sqrt(8.0 * pid + 1) - 1) / 2).to(tl.int32) # 当前线程块属于采样的第几个行块
    num_n = pid - ( num_m * num_m + num_m) // 2 # 当前线程块KV开始的位置
    start_m = num_m * stride_m + stride_m - 1 # 因为是采样range的末尾，所以 + stride_m - 1
    start_n = num_n * stride_n

    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )


    # 相关地址 ==============================
    idx_ptrs = pIndices + num_m * row_blks * BLOCK_N
    pscale_ptrs = pscale + num_m * row_blks
    psum_ptrs = psum + num_m * row_blks
    # ==============================
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") 
    l_i_new = tl.zeros([BLOCK_M], dtype=tl.float32) # 记录和                   
    q = tl.load(q_ptrs)

    # loop over k, v and update accumulator
    # end_n = tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    # for begin_n in range(0, end_n, BLOCK_N):

    end_n = tl.minimum((start_n + stride_n) * BLOCK_N, seqlen_k)
    for begin_n in range(start_n * BLOCK_N, end_n, BLOCK_N):
        # begin_n = tl.multiple_of(begin_n, BLOCK_N) # 将 begin_n 设为最接近且大于等于 begin_n 的 BLOCK_N 的整数倍
        # -- compute qk ----
        k = tl.load(k_ptrs + begin_n * stride_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong

        qk += tl.where(offs_m[:, None] >= (begin_n + offs_n)[None, :], 0, float("-inf")) # 因果注意力

        m_ij = tl.maximum(tl.max(qk) * softmax_scale, m_i) # 整块统一 # 改
        # m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, m_i) # 每行 # 改

        p = tl.exp(qk * softmax_scale - m_ij[:, None])

        # 存储列索引，部分和和缩放值 ==============================
        block_index = begin_n // BLOCK_N
        max_indices = tl.histogram(tl.argmax(qk, 1), BLOCK_N) # 直方图统计
        # tl.store(idx_ptrs + begin_n + offs_n, max_indices > -1)    # 密集测试============================================================== 
        tl.store(idx_ptrs + begin_n + offs_n, max_indices > 0)
        tl.store(pscale_ptrs + block_index, tl.sum(m_ij) / BLOCK_M) # 后续测试整块统一和每行的速度和内存占用
        tl.store(psum_ptrs + block_index, tl.sum(p))
        # ==============================

        m_i = m_ij # 改



@triton.jit # 稀疏计算
def _triton_sparse_attn_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    colIndices, column_count, # colIndices [num_sample, row_blks * BLOCK_N]; column_count [num_sample]
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    range_h: tl.constexpr, row_blks: tl.constexpr, b_per_sample: tl.constexpr, num_sample: tl.constexpr, dig: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    seqlen = tl.load(seqlens + off_hz // H)
    skip_m = start_m // b_per_sample
    if start_m * BLOCK_M >= seqlen:
        # print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    # initialize offsets
    # offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok


    cols_ptr = colIndices + skip_m * row_blks * BLOCK_N
    num_cols = tl.load(column_count + skip_m)


    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen
    max_cols = (start_m + 1) * BLOCK_M - BLOCK_M * dig

    for start_n in range(0, num_cols, BLOCK_N):
        n_mask = start_n + offs_n < num_cols # 将会是一个向量，超出范围的索引会被赋值为 False
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0) # mask=n_mask 指示是否加载该数据项，不加载数据则该位置会被赋值为 0（由 other=0 指定）
        max_mask = (cols < max_cols) & n_mask # 表明本轮计算的块号超出了本线程块需要执行的最大块号
        # -- load k, v -- 
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=max_mask[None, :], other=0.0) # 只加载col_index对应的列
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=max_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & max_mask, qk, float("-10000"))  # 不用-inf 是因为后续 exp(-inf - -inf) = NaN
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new) # 统一两个块之间的缩放
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug 创建一个与 l_i 形状相同的新变量
        acc *= acc_scale[:, None] # 对前一块的结果进行相同的缩放
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i -- l_i为softmax的分母
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new


    # # 计算对角元素，阶梯状 BLOCK_N必须等于BLOCK_M
    block_num = dig
    start_blk = max_cols
    if(start_m < dig): # 分支对时间影响不大
        block_num = start_m + 1
        start_blk = 0
    offs_n2 = tl.arange(0, BLOCK_M)
    for block_index in range(block_num):
        start_n2 = start_blk + block_index * BLOCK_M
        cols = start_n2 + offs_n2
        n_mask = cols < seqlen
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    # acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def _flash_attn_triton_decoding(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape # [[1, 10011, 1, 128]]
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    seqlen_q_rounded = math.ceil(seqlen_q / 64) * 64
    # seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    # lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32) # 原
    # tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32) # 原
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 64
    # BLOCK = 128
    # block_size_N = 64
    block_size_N = 128 # BLOCK = 64, block_size_N = 128 最优
    num_warps = 4 if d <= 64 else 8

    # range_h = 1024 # 一块采样所代表的高度
    range_h = 2 ** (math.floor(math.log(seqlen_q / 8192, 2))) * 256
    dig = 16 # local窗口 dig * block_size_N

    # threshold = -1 # 密集测试============================================================== 
    threshold = 0.1 # 阈值
    num_sample = seqlen_q // range_h # 需要采样的段数（分为的段数）
    num_sample_idx = triton.cdiv(seqlen_q, range_h) # 索引的块数,主要是最后一部分行块的索引需要特殊处理
    blocks_num = (num_sample**2 + num_sample) // 2
    grid = lambda META: (blocks_num, batch * nheads)  # 均衡采样 program_id n所属行块r=(8*n+1)**0.5//2; KV起始k=n-(r**2+r)//2 

    row_blks = math.ceil(seqlen_q / block_size_N) # 一行最多分为的块数
    pIndices = torch.zeros((num_sample_idx, row_blks * block_size_N), device=q.device, dtype = torch.bool)
    pscale = torch.zeros((num_sample, row_blks), device=q.device, dtype = torch.float32)
    psum = torch.zeros((num_sample, row_blks), device=q.device, dtype = torch.float32)

    _fwd_kernel[grid](
        q,
        k,
        softmax_scale,

        pIndices, #
        pscale, psum, #

        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),

        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,

        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,

        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=block_size_N, # 128更快

        range_h=range_h, # 
        row_blks=row_blks, #

        num_warps=num_warps,
        num_stages=1,
    )

    

    # 计算块级索引
    arange_temp = torch.arange(num_sample, device=q.device)
    posi = range_h // block_size_N * (arange_temp + 1) # len: 4096; rang_h: 1024  [ 8, 16, 24, 32]

    row_max, _ = torch.max(pscale, dim=1, keepdim=True)
    pscale -= row_max # 减去最大值

    psum.mul_(torch.exp(pscale))  # 原地乘法
    psum_temp = psum / torch.sum(psum, -1).unsqueeze(-1) * posi.unsqueeze(-1)

    # print(psum_temp)
    # exit()

    bIndices = psum_temp > threshold
    if(num_sample < num_sample_idx): # 处理最后一部分的索引
        bIndices = torch.vstack([bIndices, bIndices[-1:, :]])

    # pIndices和bIndices生成最终索引，并记录每个小行块（64）的列数
    col_mask = pIndices & torch.repeat_interleave(bIndices, repeats=block_size_N, dim=1)
    colNum = col_mask.sum(dim=-1) # [num_sample_idx] 统计每个行块的列数量
    
    # temp = torch.arange(row_blks * block_size_N).unsqueeze(0).repeat(num_sample_idx, 1).to(torch.int32).cuda() # 创建0-len的索引矩阵，极其费时！！！！！！！！！！！！！！
    temp = torch.arange(row_blks * block_size_N, dtype=torch.int32, device='cuda').unsqueeze(0).repeat(num_sample_idx, 1) 
    colIndices = torch.where(col_mask, temp, 2**30) # [num_sample_idx, row_blks * block_size_N] 元素为索引值

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    o = o.transpose(1, 2)

    batch_size, num_heads, context_size, head_dim = q.shape
    # pad = BLOCK - (context_size & (BLOCK - 1))
    pad = BLOCK - context_size & BLOCK 
    q = torch.nn.functional.pad(q, [0, 0, 0, pad, 0, 0, 0, 0])
    k = torch.nn.functional.pad(k, [0, 0, 0, pad, 0, 0, 0, 0])
    v = torch.nn.functional.pad(v, [0, 0, 0, pad, 0, 0, 0, 0])
    o = torch.nn.functional.pad(o, [0, pad, 0, pad, 0, 0, 0, 0])
    seqlens = torch.tensor([context_size], dtype=torch.int32, device=q.device)

    if(num_sample < num_sample_idx): # 处理最后一部分的索引
        colIndices[-1, :] = colIndices[-2, :] # 重用上一部分索引
        # topk预测尾部稀疏索引 预测最后未覆盖的三角形区域
        last_q = BLOCK
        last_k = seqlen_q - num_sample * range_h
        # top_k = math.ceil(last_k * 1) # 密集测试============================================================== 
        top_k = math.ceil(last_k * 0.2) # 80%稀疏度
        colNum[-1] += top_k + colNum[-2]

        qk = torch.einsum(f'bhmk, bhnk -> bhmn', q[:, :, -last_q:, :], k[:, :, -last_k:, :] * softmax_scale)
        qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK[..., -last_q:, -last_q:].to(q.device), qk[:, :, :, -last_q], -torch.inf)
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        vertical_sum = qk.sum(dim=-2)
        vertical_topk = torch.topk(vertical_sum, top_k, -1).indices
        # print(vertical_topk.shape)
        # print(vertical_topk)
        # exit()
        colIndices[-1, -top_k:] = vertical_topk[0][0] + num_sample * range_h

    colIndices, _ = torch.sort(colIndices, dim=-1) # [num_sample_idx, row_blks * block_size_N] 元素为索引值
    colIndices = torch.where(colIndices < 2**30, colIndices, 0)

    # print(colIndices[-1][-256:])
    # print(colNum)
    # exit()
    # global total_sparsity
    # global total_num
    # total_num += 1
    # total_sparsity += colNum.sum(dim=0) / num_sample_idx / seqlen_q * 2
    # if(total_num % 100 == 0):
    #     print(colNum)
    #     print(f"num: {total_num}; spa: {total_sparsity / total_num}")
    # exit()


    # 稀疏计算
    grid2 = (triton.cdiv(q.shape[-2], BLOCK), q.shape[0] * q.shape[1], 1) 
    _triton_sparse_attn_fwd_kernel[grid2](
        q, k, v, seqlens, softmax_scale,
        colIndices, colNum,
        o, 
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        range_h=range_h, row_blks=row_blks, b_per_sample = range_h // BLOCK, num_sample=num_sample, dig=dig, # b_per_sample 每个范围内有多少块（纵向）
        BLOCK_M=BLOCK, BLOCK_N=block_size_N,
        BLOCK_DMODEL=d,
        dtype=tl.bfloat16,
        num_warps=4, num_stages=2,
    )

    # o = o.transpose(1, 2)
    # return o #, pIndices, bIndices, pscale, psum #, lse, softmax_scale
    
    print(o[..., :context_size, :head_dim])
    exit()

    return o[..., :context_size, :head_dim]