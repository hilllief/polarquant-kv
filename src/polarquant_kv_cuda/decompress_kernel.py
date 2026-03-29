"""PolarQuant 解压 Kernel（含 bit unpacking）。"""

import math
import torch

from polarquant_kv_cuda.types import CompressedKVCacheGPU
from polarquant_kv_cuda.compress_kernel import _bit_unpack_quantized


def decompress_gpu(
    cache: CompressedKVCacheGPU,
    rotation_matrix: torch.Tensor,
) -> torch.Tensor:
    """GPU 上的 PolarQuant 解压。"""
    if cache.seq_len == 0:
        shape = cache.radius.shape
        return torch.empty(*shape, cache.original_dim,
                           dtype=torch.float16, device=cache.radius.device)

    d = cache.original_dim
    gs = cache.group_size
    n_bits = cache.n_bits
    num_groups = math.ceil(d / gs)
    d_padded = num_groups * gs

    batch, num_heads, seq_len = cache.radius.shape

    # Bit unpack
    quantized = _bit_unpack_quantized(cache.quantized_direction, n_bits, d_padded)
    quantized = quantized.float()

    # 反量化
    quantized_grouped = quantized.reshape(batch, num_heads, seq_len, num_groups, gs)
    group_mins = cache.group_mins.float()
    group_scales = cache.group_scales.float()

    dequant = quantized_grouped * group_scales.unsqueeze(-1) + group_mins.unsqueeze(-1)
    direction = dequant.reshape(batch, num_heads, seq_len, d_padded)[..., :d]

    # 恢复半径
    radius = cache.radius.float()
    kv_rotated = direction * radius.unsqueeze(-1)

    # 逆旋转
    kv_recovered = kv_rotated @ rotation_matrix

    # Clamp 到 FP16 安全范围
    fp16_max = torch.finfo(torch.float16).max
    kv_recovered = kv_recovered.clamp(-fp16_max, fp16_max)

    return kv_recovered.half()
