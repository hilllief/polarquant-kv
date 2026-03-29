"""高层 API。"""

import torch

from polarquant_kv_cuda.types import CompressedKVCacheGPU
from polarquant_kv_cuda.compress_kernel import compress_gpu
from polarquant_kv_cuda.decompress_kernel import decompress_gpu
from polarquant_kv_cuda.attention_kernel import compressed_attention_gpu


def compress(kv, rotation_matrix, n_bits=4, group_size=32, jl_matrix=None):
    return compress_gpu(kv, rotation_matrix, n_bits, group_size, jl_matrix)


def decompress(cache, rotation_matrix):
    return decompress_gpu(cache, rotation_matrix)


def compressed_attention(query, compressed_keys, compressed_values,
                         rotation_matrix, jl_matrix=None, enable_qjl=True,
                         num_kv_heads=None):
    return compressed_attention_gpu(
        query, compressed_keys, compressed_values,
        rotation_matrix, jl_matrix, enable_qjl, num_kv_heads,
    )


def get_memory_bytes(cache: CompressedKVCacheGPU) -> int:
    """返回压缩 KV Cache 的精确显存占用字节数。"""
    total = 0
    for field in [cache.radius, cache.quantized_direction,
                  cache.group_mins, cache.group_scales]:
        total += field.nelement() * field.element_size()
    if cache.qjl_signs is not None:
        total += cache.qjl_signs.nelement() * cache.qjl_signs.element_size()
    if cache.residual_norms is not None:
        total += cache.residual_norms.nelement() * cache.residual_norms.element_size()
    return total
