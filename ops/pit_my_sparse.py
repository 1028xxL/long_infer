# myTest

import math
import time

import torch
import triton
import triton.language as tl

# @triton.autotune(
#    configs=[
#        triton.Config({}, num_stages=1, num_warps=4),
#        triton.Config({}, num_stages=1, num_warps=8),
#        triton.Config({}, num_stages=2, num_warps=4),
#        triton.Config({}, num_stages=2, num_warps=8),
#        triton.Config({}, num_stages=3, num_warps=4),
#        triton.Config({}, num_stages=3, num_warps=8),
#        triton.Config({}, num_stages=4, num_warps=4),
#        triton.Config({}, num_stages=4, num_warps=8),
#        triton.Config({}, num_stages=5, num_warps=4),
#        triton.Config({}, num_stages=5, num_warps=8),
#    ],
#    key=['N_CTX'],
# )
@triton.jit # 采样索引计算
def _triton_index_fwd_kernwl(
    Q, K, V, seqlen, sm_scale,
    pIndices, bIndices, pscale, psum, Out, # pIndices 记录x:m索引，输入时的维度为[num_sample, row_blks * block_size_N] 全为false; bIndices记录块索引，输入时的维度为[num_sample, row_blks] 全为false
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX, # batch_size, num_heads, seqlens
    range_h: tl.constexpr, sample_h: tl.constexpr, row_blks: tl.constexpr, # range_h表示一段采样所代表的范围1024-8192, sample_h表示一段采样的高度64, row_blks = ceil(seq_lens / BLOCK_N)
    th: tl.constexpr,
    BLOCK_M: tl.constexpr, # 块大小，这里是64
    BLOCK_N: tl.constexpr, # 块大小，这里是64
    BLOCK_DMODEL: tl.constexpr, # 块大小，这里是128
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0) # 确定当前线程所属的 Block（块）在 Grid（网格）中的位置，在0-156之间选确定一个线程块（对于注意力矩阵本质上纵向并行）
    off_hz = tl.program_id(1) # 在维度1，即head层面的线程块偏移，此处为1，可忽略
    # seqlen = tl.load(seqlens + off_hz // H) # seqlens本质上是一个数组，torch.Size([1]), tensor([10011], device='cuda:0', dtype=torch.int32); 加载指定位置的 seqlen
    if start_m * BLOCK_M >= seqlen: # 表示当前线程块的计算已经超出计算范围
        return

    # initialize offsets
    # offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # 定位到具体的行范围 连续取Q
    offs_m = (start_m + 1) * range_h - sample_h + tl.arange(0, BLOCK_M) # 采样取Q 非降速原因
    offs_n = tl.arange(0, BLOCK_N) # 表示一个范围: 0-BLOCK_N-1, BLOCK_N = 64
    offs_d = tl.arange(0, BLOCK_DMODEL) # BLOCK_DMODEL = 128
    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk # 本质是一个范围 [:, None]表示在最后一维增加维度[n]变[n, 1], [None, :]变[[n]]
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    idx_ptrs = pIndices + start_m * row_blks * BLOCK_N
    bidx_ptrs = bIndices + start_m * row_blks
    pscale_ptrs = pscale + start_m * row_blks
    psum_ptrs = psum + start_m * row_blks

    # initialize pointer to m and l
    # m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") # 指数运算需要减去的值
    m_i = float("-inf") # 指数运算需要减去的值 非降速原因
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) # softmax 分母
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # scale sm_scale by log_2(e) and use, e^x = 2^[log_2(e) * x]
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen # 将超出长度的部分置为0
    num_blks = tl.ceil(((start_m + 1) * range_h) / BLOCK_N).to(tl.int32) # 计算非降速原因
    # num_blks = 100 # test 数量非降速原因

    for block_index in range(num_blks):
        start_n = block_index * BLOCK_N
        cols = start_n + offs_n
        n_mask = cols < seqlen
        # -- load k, v -- 一定范围内的k,v
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None] # 上三角置为0
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)

        # 存储列索引 ==============================
        max_indices = tl.histogram(tl.argmax(qk, 1), BLOCK_N) # 直方图统计
        tl.store(idx_ptrs + cols, max_indices > 0) 
        # ==============================

        # -- compute scaling constant --
        # m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        m_i_new = tl.maximum(m_i, tl.max(qk)) # 整块统一
        alpha = tl.math.exp2(m_i - m_i_new)
        # p = tl.math.exp2(qk - m_i_new[:, None]) 
        p = tl.math.exp2(qk - m_i_new) # 非降速原因

        # 存储部分和和缩放值 ==============================
        tl.store(pscale_ptrs + block_index, m_i_new)
        tl.store(psum_ptrs + block_index, tl.sum(p))
        # ==============================

        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
    # write back O
    acc /= l_i[:, None]

    # 存储块索引 ==============================
    temp = tl.math.exp2(tl.load(pscale_ptrs + tl.arange(0, row_blks)) - m_i) # 归一化
    psum_temp = temp * tl.load(psum_ptrs + tl.arange(0, row_blks)) / tl.sum(l_i) * num_blks  # 统一缩放
    tl.store(bidx_ptrs + tl.arange(0, row_blks), psum_temp > th)  # th表示阈值
    # ==============================

    # acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)



@triton.jit # 密集, 模拟FA2
def _triton_index_fwd_kernwl2(
    Q, K, V, sm_scale,
    # L,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr, 
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.bfloat16)
    # loop over k, v and update accumulator
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.bfloat16), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / l_i[:, None]
    # l_ptrs = L + off_hz * N_CTX + offs_m
    # tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(tl.bfloat16))



@triton.jit # 稀疏计算
def _triton_sparse_attn_fwd_kernel(
    Q, K, V, seqlen, sm_scale,
    colIndices, column_count, # colIndices [num_sample, row_blks * BLOCK_N]; column_count [num_sample]
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    range_h: tl.constexpr, sample_h: tl.constexpr, row_blks: tl.constexpr, b_per_sample: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    skip_m = start_m // (b_per_sample - 1)
    if start_m + skip_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    # offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = (start_m + skip_m) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    cols_ptr = colIndices + skip_m * row_blks * BLOCK_N
    # num_cols = tl.load(column_count + start_m + start_m / (num_sample - 1))
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
    max_cols = (start_m + skip_m) * BLOCK_M
    # max_cols = (start_m + skip_m) * BLOCK_M - BLOCK_N # 加上对角的一个小方块

    for start_n in range(0, num_cols, BLOCK_N):
        n_mask = start_n + offs_n < num_cols # 将会是一个向量，超出范围的索引会被赋值为 False
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0) # mask=n_mask 指示是否加载该数据项，不加载数据则该位置会被赋值为 0（由 other=0 指定）
        max_mask = (cols < max_cols) & n_mask
        # -- load k, v -- 
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=max_mask[None, :], other=0.0) # 只加载col_index对应的列
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=max_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float("-inf"))
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

    #计算对角元素，阶梯状，都是下三角
    cols = (start_m + skip_m) * BLOCK_N + offs_n
    n_mask = cols < seqlen
    k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
    # -- compute qk --
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    causal_mask = cols[None, :] <= offs_m[:, None]
    qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
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

    # write back O
    acc /= l_i[:, None]
    # acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)



# 中间函数，调用triton内核函数
def _triton_mixed_sparse_attention(
    q: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqlens: torch.Tensor,    # [BATCH, ]
    sm_scale: float,
    block_size_M: int = 64,
    block_size_N: int = 64,
    range_h: int = 1024,
    # range_h: int = 1024 * 8,
    sample_h: int = 64,
    th: float = 0.1 # 阈值
) -> torch.Tensor:

    # shape constraints
    sample_h = block_size_M
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q) # 提前创建，分配内存

    # num_sample = math.ceil(seqlens / range_h) # 需要采样的段数（分为的段数）
    num_sample = seqlens // range_h # 需要采样的段数（分为的段数）
    row_blks = math.ceil(seqlens / block_size_N)
    # pIndices = torch.arange(row_blks * block_size_N).unsqueeze(0).repeat(num_sample, 1).to(torch.int32).cuda()
    pIndices = torch.zeros(num_sample, row_blks * block_size_N).to(torch.bool).cuda()
    bIndices = torch.zeros((num_sample, math.ceil(seqlens / block_size_N))).to(torch.bool).cuda()
    pscale = torch.zeros((num_sample, block_size_M), dtype = torch.float32).cuda()
    psum = torch.zeros((num_sample, row_blks), dtype = torch.float32).cuda()

    # triton.cdiv(q.shape[2], block_size_M) 计算在 M 维度上划分块的数量，本质上是一个线程块将处理一行块的数据
    # q.shape[0] * q.shape[1] 是批量大小和头数的乘积（似乎都是1），表示每个块上有多少个任务
    # grid1 = (108*8, q.shape[0] * q.shape[1], 1) #  总块数的线程块
    grid_dense = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1) #  总块数的线程块
    grid1 = (num_sample, q.shape[0] * q.shape[1], 1) # 采样块数的线程块 线程块数与利用率高度相关
    grid2 = (triton.cdiv(q.shape[2], block_size_M) - num_sample, q.shape[0] * q.shape[1], 1) 
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

    # q(k v o).strde: <built-in method stride of Tensor object at 0x7f690c7edc10>
    # q.stride(0): 1286144; q.stride(1): 1286144; q.stride(2): 128; q.stride(3): 1;  表示沿着每个维度访问下一个元素的步长 
    # 1286144 = 128 * 10048 * 1 * 1; 128表示qkv维度，10048表示输入长度（被64padding之后）


    # # 采样计算
    # start = time.time()
    _triton_index_fwd_kernwl[grid1](
        q, k, v, seqlens, sm_scale,
        pIndices, bIndices, pscale, psum, o, 
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        range_h=range_h, sample_h=sample_h, row_blks=row_blks, 
        th=th,
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=2,
        # num_warps=4, num_stages=4,
        # num_warps=2, num_stages=4,
        # num_warps=8, num_stages=4,
    )
    # print(f"seqlens：{seqlens}；range_h：{range_h}；采样计算时间：{time.time() - start}")
    # exit()

    
    # # # test1 密集的GPU利用率 似乎也很低
    # # start = time.time()
    # _triton_index_fwd_kernwl2[grid_dense](
    #     q, k, v, sm_scale,
    #     o, 
    #     q.stride(0), q.stride(1), q.stride(2), q.stride(3),
    #     k.stride(0), k.stride(1), k.stride(2), k.stride(3),
    #     v.stride(0), v.stride(1), v.stride(2), v.stride(3),
    #     o.stride(0), o.stride(1), o.stride(2), o.stride(3),
    #     q.shape[0], q.shape[1], q.shape[2],
    #     BLOCK_M=block_size_M, BLOCK_N=block_size_N, BLOCK_DMODEL=Lk,
    #     IS_CAUSAL=True,
    #     # num_warps=4, num_stages=2,
    #     num_warps=8, num_stages=4,
    # )
    # # print(f"seqlens：{seqlens}；密集计算时间：{time.time() - start}")
    # # exit()

    
    # pIndices和bIndices生成最终索引，并记录每个小行块（64）的列数
    col_mask = pIndices & torch.repeat_interleave(bIndices, repeats=block_size_N, dim=1)
    colNum =  col_mask.sum(dim=-1) # [num_sample] 统计每个行块的列数量

    
    # # 稀疏度大致计算
    # print(colNum.sum(dim=0) / num_sample / seqlens * 2)
    # exit()

    
    # temp = torch.arange(row_blks * block_size_N).unsqueeze(0).repeat(num_sample, 1).to(torch.int32).cuda() # 创建0-len的索引矩阵，及其费时！！！！！！！！！！！！！！
    temp = torch.arange(row_blks * block_size_N, dtype=torch.int32, device='cuda').unsqueeze(0).repeat(num_sample, 1)
    colIndices =torch.where(col_mask, temp, 2**30) # [num_sample, row_blks * block_size_N] 元素为索引值
    colIndices, _ = torch.sort(colIndices, dim=-1) # [num_sample, row_blks * block_size_N] 元素为索引值
    colIndices = torch.where(colIndices < 2**30, colIndices, 0)


    # # x稀疏计算
    # start = time.time()
    _triton_sparse_attn_fwd_kernel[grid2](
        q, k, v, seqlens, sm_scale,
        colIndices, colNum, 
        o, 
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        range_h=range_h, sample_h=sample_h, row_blks=row_blks, b_per_sample = range_h // sample_h,
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=2,
    )

    
    return o
