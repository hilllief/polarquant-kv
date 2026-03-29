"""完整的融合 CUDA Kernel（CuPy RawKernel）。

三个核心 kernel：
1. fused_compress: 旋转 + 范数 + 归一化 + 量化 + bit pack（一个 kernel）
2. fused_score: 从 4-bit packed 直接算注意力分数（一个 kernel）
3. fused_attention: score + online softmax + V 解压 + 加权求和（一个 kernel）
"""

import math
import cupy as cp
import torch
import numpy as np

# ============================================================
# Kernel 1: 融合压缩
# 每个 thread block 处理一个向量（一个 head 的一个 token）
# ============================================================

_FUSED_COMPRESS_4BIT = cp.RawKernel(r'''
extern "C" __global__
void fused_compress_4bit(
    const float* __restrict__ input,      // [N, D] FP32 (已从 FP16 转换)
    const float* __restrict__ R,          // [D, D] 旋转矩阵 (行主序)
    float* __restrict__ radius_out,       // [N]
    unsigned char* __restrict__ packed_out,// [N, D/2]
    float* __restrict__ gmins_out,        // [N, G]
    float* __restrict__ gscales_out,      // [N, G]
    int N, int D, int group_size
) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= N) return;
    int tid = threadIdx.x;
    int num_groups = (D + group_size - 1) / group_size;

    extern __shared__ float smem[];
    float* rotated = smem;        // [D]
    float* direction = smem + D;  // [D]

    // Step 1: 旋转 v_rot[k] = sum_j input[j] * R[j*D + k]
    if (tid < D) {
        float sum = 0.0f;
        const float* v = input + vec_idx * D;
        for (int j = 0; j < D; j++) {
            sum += v[j] * R[j * D + tid];
        }
        rotated[tid] = sum;
    }
    __syncthreads();

    // Step 2: 范数 (parallel reduction)
    __shared__ float reduce_buf[256];
    reduce_buf[tid] = (tid < D) ? rotated[tid] * rotated[tid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float radius = sqrtf(reduce_buf[0]);

    if (tid == 0) radius_out[vec_idx] = radius;

    // Step 3: 归一化
    if (tid < D) {
        direction[tid] = (radius > 1e-30f) ? (rotated[tid] / radius) : 0.0f;
    }
    __syncthreads();

    // Step 4: 分组量化 + 4-bit pack
    if (tid < num_groups) {
        int g_start = tid * group_size;
        int g_end = min(g_start + group_size, D);

        float gmin = direction[g_start];
        float gmax = gmin;
        for (int i = g_start + 1; i < g_end; i++) {
            float v = direction[i];
            gmin = fminf(gmin, v);
            gmax = fmaxf(gmax, v);
        }
        float range = gmax - gmin;
        float scale = (range > 0.0f) ? (range / 15.0f) : 1.0f;

        gmins_out[vec_idx * num_groups + tid] = gmin;
        gscales_out[vec_idx * num_groups + tid] = scale;

        for (int i = g_start; i < g_end; i += 2) {
            unsigned char q0 = (unsigned char)fminf(15.0f,
                fmaxf(0.0f, roundf((direction[i] - gmin) / scale)));
            unsigned char q1 = 0;
            if (i + 1 < g_end) {
                q1 = (unsigned char)fminf(15.0f,
                    fmaxf(0.0f, roundf((direction[i+1] - gmin) / scale)));
            }
            packed_out[vec_idx * (D/2) + i/2] = q0 | (q1 << 4);
        }
    }
}
''', 'fused_compress_4bit')


# ============================================================
# Kernel 2: 融合注意力分数（从 4-bit packed 直接算 score）
# 每个 thread 处理一个 key token
# ============================================================

_FUSED_SCORE_4BIT = cp.RawKernel(r'''
extern "C" __global__
void fused_score_4bit(
    const float* __restrict__ q_rotated,
    const unsigned char* __restrict__ packed_keys,
    const float* __restrict__ gmins,
    const float* __restrict__ gscales,
    const float* __restrict__ radius,
    float* __restrict__ scores,
    int S, int D, int group_size, int num_groups, float scale
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;

    float dot = 0.0f;
    int packed_dim = D / 2;

    for (int g = 0; g < num_groups; g++) {
        float gmin = gmins[s * num_groups + g];
        float gsc = gscales[s * num_groups + g];
        int g_start = g * group_size;

        for (int i = 0; i < group_size; i += 2) {
            int idx = g_start + i;
            int byte_idx = idx / 2;
            unsigned char packed = packed_keys[s * packed_dim + byte_idx];

            float v0 = (float)(packed & 0x0F) * gsc + gmin;
            float v1 = (float)((packed >> 4) & 0x0F) * gsc + gmin;

            dot += v0 * q_rotated[idx];
            if (idx + 1 < D) dot += v1 * q_rotated[idx + 1];
        }
    }

    scores[s] = dot * radius[s] * scale;
}
''', 'fused_score_4bit')


# ============================================================
# Kernel 3: 完整融合注意力（score + online softmax + V解压 + 加权求和）
# 每个 thread block 处理一个 query token 对所有 key tokens
# Flash Attention 风格的 online softmax
# ============================================================

_FUSED_ATTENTION_4BIT = cp.RawKernel(r'''
extern "C" __global__
void fused_attention_4bit(
    const float* __restrict__ q_rotated,     // [Hq, D]
    const unsigned char* __restrict__ pk,    // [Hkv, S, D/2]
    const float* __restrict__ k_gmins,       // [Hkv, S, G]
    const float* __restrict__ k_gscales,     // [Hkv, S, G]
    const float* __restrict__ k_radius,      // [Hkv, S]
    const unsigned char* __restrict__ pv,    // [Hkv, S, D/2]
    const float* __restrict__ v_gmins,       // [Hkv, S, G]
    const float* __restrict__ v_gscales,     // [Hkv, S, G]
    const float* __restrict__ v_radius,      // [Hkv, S]
    float* __restrict__ output,              // [Hq, D]
    int Hq, int Hkv, int S, int D,
    int group_size, int num_groups, float scale
) {
    int hq = blockIdx.x;
    if (hq >= Hq) return;
    int tid = threadIdx.x;
    int hkv = hq * Hkv / Hq;
    int packed_dim = D / 2;

    extern __shared__ float smem[];
    float* acc = smem;                    // [D]
    float* reduce_buf = smem + D;         // [blockDim.x]
    // shared scalars at fixed offsets after reduce_buf
    float* shared_max = smem + D + blockDim.x;
    float* shared_sum = smem + D + blockDim.x + 1;
    float* shared_score = smem + D + blockDim.x + 2;
    float* shared_correction = smem + D + blockDim.x + 3;

    if (tid < D) acc[tid] = 0.0f;
    if (tid == 0) { *shared_max = -1e30f; *shared_sum = 0.0f; }
    __syncthreads();

    for (int s = 0; s < S; s++) {
        // --- Score: parallel dot product + reduction ---
        float my_dot = 0.0f;
        for (int i = tid; i < D; i += blockDim.x) {
            int g = i / group_size;
            float gmin = k_gmins[(hkv * S + s) * num_groups + g];
            float gsc = k_gscales[(hkv * S + s) * num_groups + g];
            int byte_idx = i / 2;
            unsigned char p = pk[(hkv * S + s) * packed_dim + byte_idx];
            float val = (i % 2 == 0) ?
                (float)(p & 0x0F) * gsc + gmin :
                (float)((p >> 4) & 0x0F) * gsc + gmin;
            my_dot += val * q_rotated[hq * D + i];
        }
        reduce_buf[tid] = my_dot;
        __syncthreads();
        for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
            if (tid < stride) reduce_buf[tid] += reduce_buf[tid + stride];
            __syncthreads();
        }

        // --- Online softmax (thread 0 computes, broadcasts) ---
        if (tid == 0) {
            float score = reduce_buf[0] * k_radius[hkv * S + s] * scale;
            float old_max = *shared_max;
            float new_max = fmaxf(old_max, score);
            float corr = expf(old_max - new_max);
            *shared_sum = (*shared_sum) * corr + expf(score - new_max);
            *shared_max = new_max;
            *shared_score = score;
            *shared_correction = corr;
        }
        __syncthreads();

        float corr = *shared_correction;
        float weight = expf(*shared_score - *shared_max);

        // --- V 解压 + 加权累加 ---
        if (tid < D) {
            acc[tid] *= corr;
            int g = tid / group_size;
            float vgmin = v_gmins[(hkv * S + s) * num_groups + g];
            float vgsc = v_gscales[(hkv * S + s) * num_groups + g];
            int byte_idx = tid / 2;
            unsigned char vp = pv[(hkv * S + s) * packed_dim + byte_idx];
            float vval = (tid % 2 == 0) ?
                (float)(vp & 0x0F) * vgsc + vgmin :
                (float)((vp >> 4) & 0x0F) * vgsc + vgmin;
            acc[tid] += weight * vval * v_radius[hkv * S + s];
        }
        __syncthreads();
    }

    if (tid < D) {
        output[hq * D + tid] = acc[tid] / (*shared_sum);
    }
}
''', 'fused_attention_4bit')


# ============================================================
# Python 接口
# ============================================================

def _to_cp(t: torch.Tensor) -> cp.ndarray:
    """torch → cupy zero-copy。"""
    return cp.from_dlpack(t.contiguous())


def _to_torch(a: cp.ndarray) -> torch.Tensor:
    """cupy → torch zero-copy。"""
    return torch.from_dlpack(a)


def fused_compress_4bit_cuda(
    kv: torch.Tensor,
    rotation_matrix: torch.Tensor,
    group_size: int = 32,
) -> dict:
    """融合 4-bit 压缩（单 kernel）。

    Args:
        kv: [B, H, S, D] or [N, D], FP16/FP32, CUDA
        rotation_matrix: [D, D], FP32, CUDA

    Returns:
        dict: radius, packed, gmins, gscales (all CUDA tensors)
    """
    original_shape = kv.shape
    D = kv.shape[-1]
    kv_flat = kv.reshape(-1, D).float().contiguous()
    N = kv_flat.shape[0]
    num_groups = (D + group_size - 1) // group_size

    radius_cp = cp.zeros(N, dtype=cp.float32)
    packed_cp = cp.zeros((N, D // 2), dtype=cp.uint8)
    gmins_cp = cp.zeros((N, num_groups), dtype=cp.float32)
    gscales_cp = cp.zeros((N, num_groups), dtype=cp.float32)

    block_size = 128  # >= D and >= num_groups, power of 2
    smem = 2 * D * 4  # 2 * D * sizeof(float)

    _FUSED_COMPRESS_4BIT(
        (N,), (block_size,),
        (_to_cp(kv_flat), _to_cp(rotation_matrix),
         radius_cp, packed_cp, gmins_cp, gscales_cp,
         N, D, group_size),
        shared_mem=smem,
    )

    return {
        "radius": _to_torch(radius_cp),
        "packed": _to_torch(packed_cp),
        "gmins": _to_torch(gmins_cp),
        "gscales": _to_torch(gscales_cp),
        "shape": original_shape,
    }


def fused_attention_scores_4bit(
    q_rotated: torch.Tensor,
    packed_keys: torch.Tensor,
    gmins: torch.Tensor,
    gscales: torch.Tensor,
    radius: torch.Tensor,
    head_dim: int,
    group_size: int = 32,
) -> torch.Tensor:
    """融合 4-bit 注意力分数（单 kernel）。"""
    S = packed_keys.shape[0]
    num_groups = (head_dim + group_size - 1) // group_size
    scale = 1.0 / math.sqrt(head_dim)

    scores_cp = cp.zeros(S, dtype=cp.float32)
    threads = 256
    blocks = (S + threads - 1) // threads

    _FUSED_SCORE_4BIT(
        (blocks,), (threads,),
        (_to_cp(q_rotated), _to_cp(packed_keys),
         _to_cp(gmins), _to_cp(gscales), _to_cp(radius),
         scores_cp, S, head_dim, group_size, num_groups, np.float32(scale)),
    )
    return _to_torch(scores_cp)


def fused_attention_4bit_cuda(
    query: torch.Tensor,
    compressed_keys: "CompressedKVCacheGPU",
    compressed_values: "CompressedKVCacheGPU",
    rotation_matrix: torch.Tensor,
) -> torch.Tensor:
    """完整融合注意力（无 Python 循环）。

    Args:
        query: [B, Hq, 1, D], FP16, CUDA
        compressed_keys/values: CompressedKVCacheGPU
        rotation_matrix: [D, D], FP32

    Returns:
        [B, Hq, 1, D], FP16
    """
    B, Hq, Sq, D = query.shape
    assert Sq == 1, "融合 kernel 当前只支持 decode (seq_q=1)"

    S = compressed_keys.seq_len
    Hkv = compressed_keys.radius.shape[1]
    group_size = compressed_keys.group_size
    num_groups = (D + group_size - 1) // group_size
    scale = 1.0 / math.sqrt(D)

    if S == 0:
        return torch.zeros_like(query)

    # 预计算旋转空间的 query: [B, Hq, D]
    q_rot = (query.squeeze(2).float() @ rotation_matrix.T).contiguous()

    # Flatten batch 维度: [B*Hq, D] 和 [B*Hkv, S, ...]
    q_flat = q_rot.reshape(B * Hq, D).contiguous()
    pk_flat = compressed_keys.quantized_direction.reshape(B * Hkv, S, -1).contiguous()
    km_flat = compressed_keys.group_mins.float().reshape(B * Hkv, S, num_groups).contiguous()
    ks_flat = compressed_keys.group_scales.float().reshape(B * Hkv, S, num_groups).contiguous()
    kr_flat = compressed_keys.radius.float().reshape(B * Hkv, S).contiguous()
    pv_flat = compressed_values.quantized_direction.reshape(B * Hkv, S, -1).contiguous()
    vm_flat = compressed_values.group_mins.float().reshape(B * Hkv, S, num_groups).contiguous()
    vs_flat = compressed_values.group_scales.float().reshape(B * Hkv, S, num_groups).contiguous()
    vr_flat = compressed_values.radius.float().reshape(B * Hkv, S).contiguous()

    out_flat = cp.zeros((B * Hq, D), dtype=cp.float32)

    block_size = 128
    smem = (D + block_size + 4) * 4

    # 单次 kernel launch，grid = (B * Hq,)
    total_heads = B * Hq
    total_kv_heads = B * Hkv

    _FUSED_ATTENTION_4BIT(
        (total_heads,), (block_size,),
        (_to_cp(q_flat), _to_cp(pk_flat), _to_cp(km_flat), _to_cp(ks_flat), _to_cp(kr_flat),
         _to_cp(pv_flat), _to_cp(vm_flat), _to_cp(vs_flat), _to_cp(vr_flat), out_flat,
         total_heads, total_kv_heads, S, D, group_size, num_groups, np.float32(scale)),
        shared_mem=smem,
    )

    outputs = _to_torch(out_flat).reshape(B, Hq, D)

    # 逆旋转
    outputs = outputs @ rotation_matrix

    return outputs.unsqueeze(2).half()


# ============================================================
# Kernel: 批量解压 4-bit direction（不乘 radius，不逆旋转）
# 输出 FP16 direction tensor，用于后续 PyTorch matmul
# ============================================================

_BATCH_DEQUANT_4BIT = cp.RawKernel(r'''
extern "C" __global__
void batch_dequant_4bit(
    const unsigned char* __restrict__ packed,  // [N, D/2]
    const float* __restrict__ gmins,           // [N, G]
    const float* __restrict__ gscales,         // [N, G]
    float* __restrict__ direction_out,         // [N, D]
    int N, int D, int group_size, int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * D;
    if (idx >= total) return;

    int n = idx / D;
    int d = idx % D;
    int g = d / group_size;
    int byte_idx = d / 2;

    float gmin = gmins[n * num_groups + g];
    float gsc = gscales[n * num_groups + g];

    unsigned char p = packed[n * (D/2) + byte_idx];
    float val;
    if (d % 2 == 0)
        val = (float)(p & 0x0F) * gsc + gmin;
    else
        val = (float)((p >> 4) & 0x0F) * gsc + gmin;

    direction_out[idx] = val;
}
''', 'batch_dequant_4bit')


def fast_dequant_direction_4bit(
    packed: torch.Tensor,
    gmins: torch.Tensor,
    gscales: torch.Tensor,
    head_dim: int,
    group_size: int = 32,
) -> torch.Tensor:
    """高效批量解压 4-bit direction 到 FP16。

    Args:
        packed: [..., D/2], uint8
        gmins: [..., G], FP16/FP32
        gscales: [..., G], FP16/FP32

    Returns:
        [..., D], FP16
    """
    original_shape = packed.shape[:-1]
    D = head_dim
    num_groups = (D + group_size - 1) // group_size
    N = packed.reshape(-1, packed.shape[-1]).shape[0]

    packed_flat = packed.reshape(N, -1).contiguous()
    gm_flat = gmins.float().reshape(N, num_groups).contiguous()
    gs_flat = gscales.float().reshape(N, num_groups).contiguous()

    out = torch.empty(N, D, dtype=torch.float32, device=packed.device)
    out_cp = cp.from_dlpack(out)

    total = N * D
    threads = 256
    blocks = (total + threads - 1) // threads

    _BATCH_DEQUANT_4BIT(
        (blocks,), (threads,),
        (cp.from_dlpack(packed_flat), cp.from_dlpack(gm_flat),
         cp.from_dlpack(gs_flat), out_cp,
         N, D, group_size, num_groups),
    )

    return out.reshape(*original_shape, D)
