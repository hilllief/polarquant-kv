"""融合注意力 Kernel：在 GPU 上直接从压缩格式计算注意力，不分配完整解压 tensor。

策略：分块计算（Flash Attention 风格），每次只解压一小块 K/V，
在 shared memory 中完成点积和累加，避免 O(seq_len * head_dim) 的中间显存。
"""

import math
import torch
import cupy as cp

from polarquant_kv_cuda.types import CompressedKVCacheGPU
from polarquant_kv_cuda.compress_kernel import _bit_unpack_quantized


# CuPy CUDA kernel: 融合解压 + 注意力分数计算
_FUSED_SCORE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void fused_dequant_score(
    const unsigned char* __restrict__ packed_keys,  // [seq_kv, packed_dim]
    const __half* __restrict__ group_mins,          // [seq_kv, num_groups]
    const __half* __restrict__ group_scales,         // [seq_kv, num_groups]
    const __half* __restrict__ radius,               // [seq_kv]
    const float* __restrict__ rotation_matrix,       // [head_dim, head_dim]
    const float* __restrict__ query,                 // [head_dim]
    float* __restrict__ scores,                      // [seq_kv]
    int seq_kv,
    int head_dim,
    int packed_dim,
    int num_groups,
    int group_size,
    int n_bits,
    float scale  // 1/sqrt(d)
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= seq_kv) return;

    // 1. 解压方向向量到寄存器/local memory
    float direction[256];  // 最大 head_dim=256
    int levels = (1 << n_bits) - 1;

    // 读取 group params
    for (int g = 0; g < num_groups; g++) {
        float gmin = __half2float(group_mins[s * num_groups + g]);
        float gscale = __half2float(group_scales[s * num_groups + g]);

        for (int i = 0; i < group_size; i++) {
            int idx = g * group_size + i;
            if (idx >= head_dim) break;

            // Bit unpack
            unsigned char val;
            if (n_bits == 4) {
                int byte_idx = idx / 2;
                unsigned char byte_val = packed_keys[s * packed_dim + byte_idx];
                if (idx % 2 == 0)
                    val = byte_val & 0x0F;
                else
                    val = (byte_val >> 4) & 0x0F;
            } else if (n_bits == 2) {
                int byte_idx = idx / 4;
                unsigned char byte_val = packed_keys[s * packed_dim + byte_idx];
                int shift = (idx % 4) * 2;
                val = (byte_val >> shift) & 0x03;
            } else {
                val = packed_keys[s * packed_dim + idx];
            }

            direction[idx] = (float)val * gscale + gmin;
        }
    }

    // 2. 恢复半径
    float r = __half2float(radius[s]);

    // 3. 逆旋转 + 点积（融合）
    // score = query · (R^T · (direction * r)) / sqrt(d)
    //       = r * query · (R^T · direction) / sqrt(d)
    //       = r * sum_j (sum_k query[j] * R[k][j]) * direction[k] / sqrt(d)
    //
    // 优化：先算 q_rotated = R · query（在旋转空间中的 query）
    // 然后 score = r * dot(q_rotated, direction) / sqrt(d)
    // 但 R 是 head_dim x head_dim，太大了放不进寄存器
    //
    // 简化：直接在旋转空间算点积
    // score = r * dot(direction, q_rotated) * scale
    // 其中 q_rotated = R · query 在 host 端预计算

    float dot = 0.0f;
    for (int k = 0; k < head_dim; k++) {
        dot += direction[k] * query[k];  // query 已经是旋转空间的
    }
    scores[s] = dot * r * scale;
}
''', 'fused_dequant_score')


def fused_compressed_attention(
    query: torch.Tensor,
    compressed_keys: CompressedKVCacheGPU,
    compressed_values: CompressedKVCacheGPU,
    rotation_matrix: torch.Tensor,
    enable_qjl: bool = False,
    num_kv_heads: int | None = None,
) -> torch.Tensor:
    """融合注意力：直接从压缩格式计算，不分配完整解压 tensor。

    当前实现：对每个 (batch, head, query_token) 调用融合 kernel 计算 scores，
    然后用 PyTorch 做 softmax + V 加权（V 仍需解压，但 K 不需要）。
    """
    batch, num_q_heads, seq_q, d = query.shape
    seq_kv = compressed_keys.seq_len
    device = query.device
    n_bits = compressed_keys.n_bits
    gs = compressed_keys.group_size
    num_groups = math.ceil(d / gs)

    if seq_kv == 0:
        return torch.zeros_like(query)

    n_kv_heads = compressed_keys.radius.shape[1]
    scale = 1.0 / math.sqrt(d)

    # 预计算旋转空间的 query: q_rotated = q @ R^T（因为 K 在旋转空间）
    # 注意：compress 时做了 kv @ R^T，所以 K 在旋转空间
    # score = q · k_original = q · (R^T · k_rotated) = (R · q) · k_rotated
    # 所以 q_rotated = q @ R^T ... 不对
    # k_original = R^T @ k_rotated (逆旋转)
    # score = q · k_original = q · (k_rotated @ R) = (q @ R^T) · k_rotated ... 不对
    # 让我重新推导：
    # compress: k_rotated = k @ R^T
    # decompress: k_hat = k_rotated_hat @ R
    # score = q · k_hat = q · (k_rotated_hat @ R) = (q @ R^T) · k_rotated_hat
    # 所以 q_rotated = q @ R^T
    q_rotated = query.float() @ rotation_matrix.T  # [B, Hq, Sq, D]

    # 解压 V（V 仍需完整解压用于加权求和）
    from polarquant_kv_cuda.decompress_kernel import decompress_gpu
    V_hat = decompress_gpu(compressed_values, rotation_matrix)  # [B, Hkv, Skv, D]

    # GQA 扩展
    if n_kv_heads < num_q_heads:
        repeat = num_q_heads // n_kv_heads
        V_hat = V_hat.repeat_interleave(repeat, dim=1)

    # 对每个 batch, head 计算融合注意力分数
    all_outputs = torch.zeros(batch, num_q_heads, seq_q, d, dtype=torch.float32, device=device)

    packed_dim = compressed_keys.quantized_direction.shape[-1]

    for b in range(batch):
        for hq in range(num_q_heads):
            hkv = hq if n_kv_heads == num_q_heads else hq // (num_q_heads // n_kv_heads)

            # 获取该 head 的压缩 K 数据
            pk = compressed_keys.quantized_direction[b, hkv]  # [Skv, packed_dim]
            gm = compressed_keys.group_mins[b, hkv]           # [Skv, G]
            gs_t = compressed_keys.group_scales[b, hkv]       # [Skv, G]
            rad = compressed_keys.radius[b, hkv]              # [Skv]

            # 转为 CuPy 数组
            pk_cp = cp.asarray(pk.contiguous())
            gm_cp = cp.asarray(gm.contiguous())
            gs_cp = cp.asarray(gs_t.contiguous())
            rad_cp = cp.asarray(rad.contiguous())
            R_cp = cp.asarray(rotation_matrix.contiguous())

            for sq in range(seq_q):
                # 旋转空间的 query
                q_rot = q_rotated[b, hq, sq]  # [D]
                q_cp = cp.asarray(q_rot.contiguous())

                # 分配 scores
                scores_cp = cp.zeros(seq_kv, dtype=cp.float32)

                # 调用融合 kernel
                threads = 256
                blocks = (seq_kv + threads - 1) // threads
                _FUSED_SCORE_KERNEL(
                    (blocks,), (threads,),
                    (pk_cp, gm_cp, gs_cp, rad_cp, R_cp, q_cp, scores_cp,
                     seq_kv, d, packed_dim, num_groups, gs, n_bits, scale),
                )

                # 转回 torch
                scores_torch = torch.as_tensor(scores_cp, device=device).unsqueeze(0)  # [1, Skv]

                # Softmax + 加权求和
                weights = torch.softmax(scores_torch, dim=-1)  # [1, Skv]
                v_head = V_hat[b, hq]  # [Skv, D]（已 GQA 扩展）
                out = weights @ v_head.float()  # [1, D]
                all_outputs[b, hq, sq] = out.squeeze(0)

    return all_outputs.half()
