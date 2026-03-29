"""融合注意力 V2：batch 化 CuPy kernel，避免 Python 循环。

核心优化：K 不分配完整解压 tensor，直接从压缩格式计算 scores。
V 仍需解压（因为加权求和需要完整 V 向量）。
"""

import math
import torch
import cupy as cp

from polarquant_kv_cuda.types import CompressedKVCacheGPU
from polarquant_kv_cuda.decompress_kernel import decompress_gpu


def fused_compressed_attention_v2(
    query: torch.Tensor,
    compressed_keys: CompressedKVCacheGPU,
    compressed_values: CompressedKVCacheGPU,
    rotation_matrix: torch.Tensor,
    enable_qjl: bool = False,
    num_kv_heads: int | None = None,
) -> torch.Tensor:
    """融合注意力 V2：K 不解压，直接在旋转空间算点积。

    优化思路：
    score = q · k_hat = q · (direction * radius @ R)
          = radius * (q @ R^T) · direction
          = radius * q_rotated · direction

    q_rotated 只需算一次，direction 从压缩格式在线解压。
    这样 K 的内存读取量从 head_dim * 2 bytes (FP16) 降到
    packed_dim * 1 byte + group_params，约 1/3 ~ 1/4。
    """
    batch, num_q_heads, seq_q, d = query.shape
    seq_kv = compressed_keys.seq_len
    device = query.device
    n_bits = compressed_keys.n_bits
    gs = compressed_keys.group_size
    num_groups = math.ceil(d / gs)
    d_padded = num_groups * gs
    scale = 1.0 / math.sqrt(d)

    if seq_kv == 0:
        return torch.zeros_like(query)

    n_kv_heads = compressed_keys.radius.shape[1]

    # 预计算旋转空间的 query
    q_rotated = query.float() @ rotation_matrix.T  # [B, Hq, Sq, D]

    # 在线计算 scores：使用融合 CUDA kernel（不解压 K）
    from polarquant_kv_cuda.fused_cuda_kernels import fused_attention_scores_4bit

    n_bits = compressed_keys.n_bits
    packed_keys = compressed_keys.quantized_direction  # [B, Hkv, Skv, packed_dim]
    gm = compressed_keys.group_mins.float()
    gsc = compressed_keys.group_scales.float()
    rad = compressed_keys.radius.float()

    # GQA 扩展
    if n_kv_heads < num_q_heads:
        repeat = num_q_heads // n_kv_heads
        packed_keys = packed_keys.repeat_interleave(repeat, dim=1)
        gm = gm.repeat_interleave(repeat, dim=1)
        gsc = gsc.repeat_interleave(repeat, dim=1)
        rad = rad.repeat_interleave(repeat, dim=1)

    # 对每个 (batch, head, query_token) 调用融合 kernel
    scores = torch.zeros(batch, num_q_heads, seq_q, seq_kv, dtype=torch.float32, device=device)

    if n_bits == 4:
        for b in range(batch):
            for h in range(num_q_heads):
                for sq in range(seq_q):
                    scores[b, h, sq] = fused_attention_scores_4bit(
                        q_rotated[b, h, sq],
                        packed_keys[b, h],
                        gm[b, h],
                        gsc[b, h],
                        rad[b, h],
                        d, gs,
                    )
    else:
        # 非 4-bit fallback: 用 PyTorch 向量化
        from polarquant_kv_cuda.compress_kernel import _bit_unpack_quantized
        quantized = _bit_unpack_quantized(packed_keys, n_bits, d_padded).float()
        quantized_grouped = quantized.reshape(batch, num_q_heads, seq_kv, num_groups, gs)
        direction = (quantized_grouped * gsc.unsqueeze(-1) + gm.unsqueeze(-1))
        direction = direction.reshape(batch, num_q_heads, seq_kv, d_padded)[..., :d]
        scores = torch.matmul(q_rotated, direction.transpose(-2, -1))
        scores = scores * rad.unsqueeze(2) * scale

    # Softmax
    weights = torch.softmax(scores, dim=-1)

    # V 仍需解压
    V_hat = decompress_gpu(compressed_values, rotation_matrix)
    if n_kv_heads < num_q_heads:
        V_hat = V_hat.repeat_interleave(num_q_heads // n_kv_heads, dim=1)

    output = torch.matmul(weights, V_hat.float())
    return output.half()
