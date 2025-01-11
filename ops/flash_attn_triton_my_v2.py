import math

import torch
import triton
import triton.language as tl


# @triton.heuristics(
#     {
#         "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
#         "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
#         "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
#     }
# )

@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    # Lse, # 原
    # TMP, # 原
    softmax_scale,

    temp_acc, temp_scale, temp_sum,

    pIndices, pscale, psum, #

    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,

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
    threshold: tl.constexpr, #
):

    # start_m = tl.program_id(0) # 密集计算

    # start_m_old = tl.program_id(0)
    # stride = range_h // BLOCK_M
    # start_m = start_m_old * stride + stride - 1 # 采样计算

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
    # offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )

    # 相关地址 ==============================
    idx_ptrs = pIndices + num_m * row_blks * BLOCK_N
    pscale_ptrs = pscale + num_m * row_blks
    psum_ptrs = psum + num_m * row_blks
    # ==============================
    
    # t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m # 原
    # lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") # 原
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") 
    
    # l_i_new = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") # 记录和 # 原
    l_i_new = tl.zeros([BLOCK_M], dtype=tl.float32) # 记录和 # 改

    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
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

        # 整块统一 ============================== 
        m_ij = tl.maximum(tl.max(qk) * softmax_scale, m_i) # 整块统一 # 改
        # m_ij = tl.maximum(tl.max(qk) * softmax_scale, lse_i) # 整块统一 # 原
        # m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i) # 每行
        # ==============================

        p = tl.exp(qk * softmax_scale - m_ij[:, None])

        # 存储列索引，部分和和缩放值 ==============================
        block_index = begin_n // BLOCK_N
        max_indices = tl.histogram(tl.argmax(qk, 1), BLOCK_N) # 直方图统计
        tl.store(idx_ptrs + begin_n + offs_n, max_indices > 0)     
        tl.store(pscale_ptrs + block_index, tl.sum(m_ij) / BLOCK_M) # 测试整块统一和每行的速度和内存占用
        tl.store(psum_ptrs + block_index, tl.sum(p))
        # ==============================

        l_ij = tl.sum(p, 1) # 每行的部分和

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij) # 对应MInfer 中的alpha

        # # -- update output accumulator --
        # tl.store(t_ptrs, acc_o_scale) # 原
        # acc_o_scale = tl.load(t_ptrs) # 原
        acc_o = acc_o * acc_o_scale[:, None]
        # update acc_o
        v = tl.load(v_ptrs + begin_n * stride_vn)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        # m_i = m_ij # 当前块的指数缩放 # 原
        # l_i_new = tl.exp(lse_i - m_ij) + l_ij # 累计分母 对应l_i # 原
        # lse_i = m_ij + tl.log(l_i_new) # 意味着缩放值永远会增大  ??? # 原

        l_i_new = l_i_new * tl.exp(m_i - m_ij) + l_ij # 累计分母 对应l_i # 改
        m_i = m_ij # 改

    # 存储每个block的缩放和分母部分和
    tl.store(temp_scale + pid * BLOCK_M + tl.arange(0, BLOCK_M), m_i)
    tl.store(temp_sum + pid * BLOCK_M + tl.arange(0, BLOCK_M), l_i_new)

    # o_scale = tl.exp(m_i - lse_i) # 因为前面的l_i_new减去了m_ij  有啥目的 反复存取？ # 原
    # tl.store(t_ptrs, o_scale) # 原
    # o_scale = tl.load(t_ptrs) # 原
    # acc_o = acc_o * o_scale[:, None] # 原

    # acc_o /=  l_i_new[:, None] # 改

    # start_m = tl.program_id(0) # 密集
    # start_m = tl.program_id(0) * stride + stride - 1 # 采样

    # offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m # 原
    # tl.store(lse_ptrs, lse_i) # 原

    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # out_ptrs = (
    #     Out
    #     + off_b * stride_ob
    #     + off_h * stride_oh
    #     + (offs_m[:, None] * stride_om + offs_d[None, :]) # stride_om array([128], dtype=int32)
    # )

    off_temp = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    out_ptrs = temp_acc + off_temp[:, None] * stride_qm + offs_d[None, :]

    tl.store(out_ptrs, acc_o)


def _flash_attn_triton_decoding(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False):
    # shape constraints
    # print(q.shape) # [1, seqlen, 1, 128]
    # exit()
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
    block_size_N = 128
    # BLOCK = 128
    num_warps = 4 if d <= 64 else 8

    # grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads) # 密集  BLOCK_M 似乎是128
    # # 采样
    range_h = 1024 # 一块采样所代表的高度
    threshold = 0.5 # 阈值
    num_sample = seqlen_q // range_h # 需要采样的段数（分为的段数）
    stirde = lambda META: triton.cdiv(seqlen_q, META["BLOCK_M"])
    # grid = lambda META: (num_sample, batch * nheads)  # 标准采样
    blocks_num = (num_sample**2 + num_sample) // 2
    grid = lambda META: (blocks_num, batch * nheads)  # 均衡采样 program_id n所属行块r=(8*n+1)**0.5//2; KV起始k=n-(r**2+r)//2 

    row_blks = math.ceil(seqlen_q / block_size_N) # 一行最多分为的块数
    pIndices = torch.zeros((num_sample, row_blks * block_size_N), device=q.device, dtype = torch.bool)
    pscale = torch.empty((num_sample, row_blks), device=q.device, dtype = torch.float32)
    psum = torch.empty((num_sample, row_blks), device=q.device, dtype = torch.float32)

    temp_acc = torch.empty((blocks_num * BLOCK, d), device=q.device, dtype = torch.float32) # 存储每个线程块的分子部分和
    # temp_scale = torch.empty((blocks_num), device=q.device, dtype = torch.float32) # 存储每个线程块最终的缩放 整块统一
    temp_scale = torch.empty((blocks_num, BLOCK), device=q.device, dtype = torch.float32) # 存储每个线程块最终的缩放 每行单独
    temp_sum = torch.empty((blocks_num, BLOCK), device=q.device, dtype = torch.float32) # 存储每个线程块的分母部分和

    _fwd_kernel[grid](
        q,
        k,
        v,
        o,
        # lse, # 原
        # tmp, # 原
        softmax_scale,

        temp_acc, temp_scale, temp_sum,

        pIndices, pscale, psum, #

        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        o.stride(0),
        o.stride(2),
        o.stride(1),

        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        # causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        # BLOCK_N=BLOCK,
        BLOCK_N=block_size_N, # 128更快

        range_h=range_h, # 
        row_blks=row_blks, #
        threshold=threshold, #

        num_warps=num_warps,
        num_stages=1,
    )

    # 将采样部分和拼接到输出中
    temp_acc = temp_acc.view(-1, BLOCK, d) # ([640, 128]) -> ([10, 64, 128])
    temp_scale = temp_scale.view(-1, BLOCK, 1) # ([10, 64])-> ([10, 64, 1])
    temp_sum = temp_sum.view(-1, BLOCK, 1) # ([10, 64])-> ([10, 64, 1])

    group_sizes = torch.arange(1, num_sample + 1, device=q.device)  # [1, 2, ..., num_sample] 计算每个组的大小
    end_indices = group_sizes.cumsum(dim=0) - 1  # [0, 2, 5, 9, ...] 计算每个组的结束索引
    start_indices = torch.cat([torch.tensor([0], device=q.device), end_indices[:-1] + 1])  # [0, 1, 3, 6, ...] # 计算每个组的起始索引

    cumsum_scale = temp_scale.cumsum(dim=0) # 计算缩放的前缀累计和
    cumsum_padded_scale = torch.cat([torch.zeros(1, temp_scale.size(1), temp_scale.size(2), device=q.device), cumsum_scale], dim=0) # 在累积和的开头添加一个全零的切片，便于计算
    mean_scale = (cumsum_padded_scale[end_indices + 1] - cumsum_padded_scale[start_indices]) / group_sizes.unsqueeze(-1).unsqueeze(-1) # [4, 64, 1] 计算缩放的均值，最大值即不好求也容易溢出
    mean_scale_expand = mean_scale.repeat_interleave(group_sizes, dim=0) # [10, 64, 1] 扩展回去

    temp_acc *= torch.exp(temp_scale - mean_scale_expand)
    temp_sum *= torch.exp(temp_scale - mean_scale_expand)

    cumsum_temp_acc = temp_acc.cumsum(dim=0)
    cumsum_padded_acc = torch.cat([torch.zeros(1, temp_acc.size(1), temp_acc.size(2), device=q.device), cumsum_temp_acc], dim=0)  # [1 + 1 + 2 + ... +num_sample, 64, 128]
    sample_acc = cumsum_padded_acc[end_indices + 1] - cumsum_padded_acc[start_indices]

    cumsum_temp_sum = temp_sum.cumsum(dim=0)
    cumsum_padded_sum = torch.cat([torch.zeros(1, temp_sum.size(1), temp_sum.size(2), device=q.device), cumsum_temp_sum], dim=0)  # [1 + 1 + 2 + ... +num_sample, 64, 128]
    sample_sum = cumsum_padded_sum[end_indices + 1] - cumsum_padded_sum[start_indices]

    sample_o = sample_acc / sample_sum

    # print(sample_o.shape) # ([4, 64, 128])
    # print(sample_o[1][-1][0:16]) # 对应 o.transpose(1, 2)[0][2047][0][0:16]

    begin_indices = (torch.arange(1, num_sample + 1, device=q.device, dtype=torch.int32) * range_h - BLOCK).view(-1, 1)
    mid_indices = torch.arange(BLOCK, device=q.device, dtype=torch.int32).view(1, -1)
    o_indices = (begin_indices + mid_indices).view(-1)
    o[:,o_indices, :, :] = sample_o.view(batch, -1, nheads, d).to(torch.bfloat16)



    # 计算块级索引
    arange_temp = torch.arange(num_sample, device=q.device)
    posi = range_h // block_size_N * (arange_temp + 1) # len: 4096; rang_h: 1024  [ 8, 16, 24, 32]
    # pscale -= pscale[arange_temp, posi - 1].unsqueeze(-1) # 减去最大值，即最后一个 pscale: [num_sample, row_blks] [4, 4096 / 128]
    pscale -= mean_scale[arange_temp, 0, 0].unsqueeze(-1) # 减去均值
    psum.mul_(torch.exp(pscale))  # 原地乘法
    psum_temp = psum / torch.sum(psum, -1).unsqueeze(-1) * posi.unsqueeze(-1)
    bIndices = psum_temp > threshold
    # print(psum_temp)
    # print(bIndices)
    # exit()



    return o #, pIndices, bIndices, pscale, psum #, lse, softmax_scale


# 标准注意力
def scaled_dot_product_attention(
    query, key, value):
    """
    计算缩放点积注意力，并应用因果掩码。
    """
    d_k = query.size(-1)
    
    # 计算 QK^T
    # [batch_size, num_heads, seq_len, embed_dim] @ [batch_size, num_heads, embed_dim, seq_len] -> [batch_size, num_heads, seq_len, seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    
    # 创建因果掩码
    # mask shape: [1, 1, seq_len, seq_len]
    seq_len = query.size(-2)
    mask = torch.tril(torch.ones((seq_len, seq_len), device=query.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    
    # 应用掩码，将未来的时间步设置为 -inf
    scores = scores.masked_fill(~mask, float('-inf'))
    
    # 计算注意力权重
    attn = torch.nn.functional.softmax(scores, dim=-1)
    
    # 计算最终输出
    output = torch.matmul(attn, value)
    
    return output



# # block_size_M: int = 64
# # block_size_N: int = 64
# len = 1024 * 4 # 测试时使用64的倍数，后续调整
# query = torch.randn(1, 1, len, 128).to(torch.bfloat16).cuda()
# key = torch.randn(1, 1, len, 128).to(torch.bfloat16).cuda()
# value = torch.randn(1, 1, len, 128).to(torch.bfloat16).cuda()
# batch_size, num_heads, context_size, head_dim = query.shape
# # pad = block_size_M - (context_size & (block_size_M - 1)) # &取余
# # query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0]) # pad 的参数格式是 [left, right, top, bottom, front, back]
# # key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
# # value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])
# # seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
# # sm_scale = head_dim ** -0.5

# out, pIndices, bIndices, pscale, psum = _flash_attn_triton_decoding(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1,2), causal=True)

# # print(f"out.shape:      {out.shape}") # [1, 1, seqlen, 128]
# # print(f"pIndices.shape: {pIndices.shape}") # [num_sample, row_blks * block_size_N]
# # print(f"bIndices.shape: {bIndices.shape}") # [num_sample, row_blks]
# # print(f"pscale.shape:   {pscale.shape}") # [num_sample, row_blks]
# # print(f"psum.shape:     {psum.shape}") # [num_sample, row_blks]
# # # print(pIndices[-1, 0:256])
# # # print(bIndices[-1,:])
# # print(pscale[-1,:])
# # print(psum[-1,:])

# print(out[0][2047][0][0:16])
# # print(out[0][959][0][0:16])
# # print(out[0][1023][0][0:16])

# # # 常规验证
# o = scaled_dot_product_attention(query, key, value)
# print(o.transpose(1, 2)[0][2047][0][0:16])
# # print(o.transpose(1, 2)[0][960][0][0:16])
# # print(o.transpose(1, 2)[0][1023][0][0:16])