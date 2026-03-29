"""融合注意力 V3：纯 PyTorch 向量化，在旋转空间算 score。

不用 CuPy kernel，不用 Python 循环。
核心优化：K 在旋转空间解压（不做逆旋转），score 在旋转空间算。
省掉 K 的逆旋转 matmul（最大的开销之一）。
"""

import math
import torch

from polarquant_kv_cuda.types import CompressedKVCacheGPU
from polarquant_kv_cuda.compress_kernel import _bit_unpack_quantized
from polarquant_kv_cuda.decompress_kernel import decompress_gpu


def fused_attention_v3(
    query: torch.Tensor,
    compressed_keys: CompressedKVCacheGPU,
    compressed_values: CompressedKVCacheGPU,
    rotation_matrix: torch.Tensor,
    enable_qjl: bool = False,
    num_kv_heads: int | None = None,
) -> torch.Tensor:
    """V3: 在旋转空间算 score，K 不做逆旋转。

    score = q · k_hat
          = (q @ R^T) · (direction * radius)   [旋转空间]
          = radius * q_rotated · direction

    K 只需解压到 direction（不乘 radius，不逆旋转），
    然后 score = q_rotated @ direction^T * radius * scale
    全部用 PyTorch 向量化 matmul，一步到位。
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

    # Step 1: q_rotated = q @ R^T  [B, Hq, Sq, D]
    q_rot = query.float() @ rotation_matrix.T

    # Step 2: 解压 K 的 direction（不乘 radius，不逆旋转）
    quantized = _bit_unpack_quantized(
        compressed_keys.quantized_direction, n_bits, d_padded
    ).float()  # [B, Hkv, Skv, d_padded]

    quantized_grouped = quantized.reshape(
        batch, n_kv_heads, seq_kv, num_groups, gs
    )
    gm = compressed_keys.group_mins.float()
    gsc = compressed_keys.group_scales.float()
    direction = (quantized_grouped * gsc.unsqueeze(-1) + gm.unsqueeze(-1))
    direction = direction.reshape(batch, n_kv_heads, seq_kv, d_padded)[..., :d]

    radius = compressed_keys.radius.float()  # [B, Hkv, Skv]

    # GQA
    if n_kv_heads < num_q_heads:
        repeat = num_q_heads // n_kv_heads
        direction = direction.repeat_interleave(repeat, dim=1)
        radius = radius.repeat_interleave(repeat, dim=1)

    # Step 3: scores = q_rot @ direction^T * radius * scale
    # [B, Hq, Sq, D] @ [B, Hq, D, Skv] = [B, Hq, Sq, Skv]
    scores = torch.matmul(q_rot, direction.transpose(-2, -1))
    scores = scores * radius.unsqueeze(2) * scale

    # Step 4: softmax
    weights = torch.softmax(scores, dim=-1)

    # Step 5: V 解压（需要完整解压，因为加权求和在原始空间）
    V_hat = decompress_gpu(compressed_values, rotation_matrix)
    if n_kv_heads < num_q_heads:
        V_hat = V_hat.repeat_interleave(num_q_heads // n_kv_heads, dim=1)

    # Step 6: output = weights @ V_hat
    output = torch.matmul(weights, V_hat.float())
    return output.half()
