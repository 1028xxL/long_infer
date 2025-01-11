import math

import torch
import triton
import triton.language as tl


# @triton.heuristics(
#     {
#         "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0, # 默认全True
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
    Lse,
    TMP,
    softmax_scale,

    pIndices, bIndices, pscale, psum, #

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

    start_m_old = tl.program_id(0) #
    stride = range_h // BLOCK_M #
    start_m = start_m_old * stride + stride - 1 # 采样计算

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
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )

    # 相关地址 ==============================
    idx_ptrs = pIndices + start_m_old * row_blks * BLOCK_N
    bidx_ptrs = bIndices + start_m_old * row_blks
    pscale_ptrs = pscale + start_m_old * row_blks
    psum_ptrs = psum + start_m_old * row_blks
    # ==============================
   
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") 
    
    l_i_new = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") # 记录和

    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    q = tl.load(q_ptrs)

    # loop over k, v and update accumulator
    end_n = tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong

        # # 存储列索引 ==============================
        # max_indices = tl.histogram(tl.argmax(qk, 1), BLOCK_N) # 直方图统计
        # tl.store(idx_ptrs + start_n + offs_n, max_indices > 0)
        # # ==============================

        qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        
        # m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i) # 每行单独计算

        # 整块统一 ==============================
        m_ij = tl.maximum(tl.max(qk) * softmax_scale, lse_i) # 整块统一
        # m_ij = tl.maximum(tl.max(qk) * softmax_scale, lse_i) # 每行
        # ==============================

        p = tl.exp(qk * softmax_scale - m_ij[:, None])

        # # 存储部分和和缩放值 ==============================
        # block_index = start_n // BLOCK_N
        # tl.store(pscale_ptrs + block_index, tl.sum(m_ij) / BLOCK_M) # 测试整块统一和每行的速度和内存占用
        # tl.store(psum_ptrs + block_index, tl.sum(p))
        # # ==============================

        l_ij = tl.sum(p, 1) # 每行的部分和

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij) # 对应MInfer 中的alpha

        # # -- update output accumulator --
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        # update acc_o
        v = tl.load(v_ptrs + start_n * stride_vn)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij # 当前块的指数缩放
        l_i_new = tl.exp(lse_i - m_ij) + l_ij # 累计分母 对应l_i
        lse_i = m_ij + tl.log(l_i_new) # 意味着缩放值永远会增大  ???

    # # 存储块索引 ============================== 似乎可以不在triton内部实现
    # temp = tl.exp(tl.load(pscale_ptrs + tl.arange(0, row_blks)) - (tl.sum(m_i) / BLOCK_M)) # 将每块的部分和归一化   tl.sum(m_i) / BLOCK_M 确认无误，在随机数测试中大概是8~9
    # psum_temp = temp * tl.load(psum_ptrs + tl.arange(0, row_blks)) / tl.sum(l_i_new) * (start_m + 1) * BLOCK_M / BLOCK_N # 统一缩放  lse_i在随机数测试中大概是8~9，大于m_i
    # tl.store(bidx_ptrs + tl.arange(0, row_blks), psum_temp > threshold)  # threshold表示阈值
    # # ==============================

    o_scale = tl.exp(m_i - lse_i) # 因为前面的l_i_new减去了m_ij  有啥目的 反复存取？
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]

    start_m = tl.program_id(0) # 密集

    # start_m = tl.program_id(0) * stride + stride - 1 # 采样

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
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
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 64
    # BLOCK = 128

    block_size_N = 128
    num_warps = 4 if d <= 64 else 8

    
    # # 采样
    range_h = 1024 # 一块采样所代表的高度
    threshold = 1 # 阈值
    num_sample = seqlen_q // range_h # 需要采样的段数（分为的段数）
    row_blks = math.ceil(seqlen_q / block_size_N) # 一行最多分为的块数
    pIndices = torch.zeros((num_sample, row_blks * block_size_N), dtype = torch.bool).cuda() # 创建这几个中间会张量耗费一定的时间
    bIndices = torch.zeros((num_sample, row_blks),dtype = torch.bool).cuda()
    pscale = torch.zeros((num_sample, row_blks), dtype = torch.float32).cuda()
    psum = torch.zeros((num_sample, row_blks), dtype = torch.float32).cuda()
    grid = lambda META: (seqlen_q // range_h, batch * nheads) 

    # 密集
    # grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)  BLOCK_M 似乎是128

    _fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        tmp,
        softmax_scale,

        pIndices, bIndices, pscale, psum, #

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

    # # 计算块级索引
    # arange_temp = torch.arange(num_sample, device=q.device)
    # posi = range_h // block_size_N * (arange_temp + 1) # len: 4096; rang_h: 1024  [ 8, 16, 24, 32]
    # pscale -= pscale[arange_temp, posi - 1].unsqueeze(-1)
    # psum.mul_(torch.exp(pscale))  # 原地乘法
    # psum_temp = psum / torch.sum(psum, -1).unsqueeze(-1) * posi.unsqueeze(-1)
    # bIndices = psum_temp > threshold

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

# print(f"out.shape:      {out.shape}") # [1, 1, seqlen, 128]
# print(f"pIndices.shape: {pIndices.shape}") # [num_sample, row_blks * block_size_N]
# print(f"bIndices.shape: {bIndices.shape}") # [num_sample, row_blks]
# print(f"pscale.shape:   {pscale.shape}") # [num_sample, row_blks]
# print(f"psum.shape:     {psum.shape}") # [num_sample, row_blks]
# # print(pIndices[-1, 0:256])
# # print(bIndices[-1,:])
# print(pscale[-2,:])
# print(psum[-2,:])

# # # print(out[0][959][0][0:16])
# # print(out[0][1023][0][0:16])

# # # 常规验证
# # o = scaled_dot_product_attention(query, key, value)
# # # print(o.transpose(1, 2)[0][959][0][0:16])
# # print(o.transpose(1, 2)[0][1023][0][0:16])