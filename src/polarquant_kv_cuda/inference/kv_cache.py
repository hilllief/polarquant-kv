"""KV Cache Manager — 压缩 KV Cache 的管理器。"""

import math
import torch

from polarquant_kv_cuda.types import CompressedKVCacheGPU
from polarquant_kv_cuda.compress_kernel import compress_gpu
from polarquant_kv_cuda.decompress_kernel import decompress_gpu
from polarquant_kv_cuda.compressor import get_memory_bytes
from polarquant_kv_cuda.rotation import generate_rotation_matrix
from polarquant_kv_cuda.inference.config import PolarQuantConfig


class KVCacheManager:
    """管理压缩的 KV Cache。支持 prefill（批量写入）和 decode（逐 token 追加）。"""

    def __init__(
        self,
        config: PolarQuantConfig,
        max_seq_len: int,
        batch: int,
        num_heads: int,
        head_dim: int,
        device: str = "cuda",
        rotation_seed: int = 42,
    ):
        self.config = config
        self.max_seq_len = max_seq_len
        self.batch = batch
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        self.R = generate_rotation_matrix(head_dim, seed=rotation_seed, device=device)

        # 标准模式（不压缩）的缓存
        self._key_cache: torch.Tensor | None = None
        self._value_cache: torch.Tensor | None = None

        # 压缩模式的缓存
        self._compressed_keys: CompressedKVCacheGPU | None = None
        self._compressed_values: CompressedKVCacheGPU | None = None

        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def memory_bytes(self) -> int:
        if not self.config.enable_compression:
            if self._key_cache is None:
                return 0
            return (self._key_cache.nelement() + self._value_cache.nelement()) * 2  # FP16
        if self._compressed_keys is None:
            return 0
        return get_memory_bytes(self._compressed_keys) + get_memory_bytes(self._compressed_values)

    @property
    def compression_ratio(self) -> float:
        if self._seq_len == 0:
            return 0.0
        original = self.batch * self.num_heads * self._seq_len * self.head_dim * 2 * 2  # K+V, FP16
        compressed = self.memory_bytes
        return original / compressed if compressed > 0 else 0.0

    def append(self, key: torch.Tensor, value: torch.Tensor):
        """追加 KV 向量到缓存。

        Args:
            key: [batch, num_heads, new_seq, head_dim], FP16
            value: [batch, num_heads, new_seq, head_dim], FP16
        """
        new_seq = key.shape[2]
        if self._seq_len + new_seq > self.max_seq_len:
            raise RuntimeError(
                f"KV Cache 已满: 当前 {self._seq_len} + 新增 {new_seq} > 最大 {self.max_seq_len}"
            )

        if not self.config.enable_compression:
            self._append_standard(key, value)
        else:
            self._append_compressed(key, value)

        self._seq_len += new_seq

    def _append_standard(self, key, value):
        if self._key_cache is None:
            self._key_cache = key
            self._value_cache = value
        else:
            self._key_cache = torch.cat([self._key_cache, key], dim=2)
            self._value_cache = torch.cat([self._value_cache, value], dim=2)

    def _append_compressed(self, key, value):
        ck = compress_gpu(key, self.R, self.config.n_bits, self.config.group_size)
        cv = compress_gpu(value, self.R, self.config.n_bits, self.config.group_size)

        if self._compressed_keys is None:
            self._compressed_keys = ck
            self._compressed_values = cv
        else:
            self._compressed_keys = self._concat_compressed(self._compressed_keys, ck)
            self._compressed_values = self._concat_compressed(self._compressed_values, cv)

    def _concat_compressed(self, a: CompressedKVCacheGPU, b: CompressedKVCacheGPU) -> CompressedKVCacheGPU:
        return CompressedKVCacheGPU(
            radius=torch.cat([a.radius, b.radius], dim=2),
            quantized_direction=torch.cat([a.quantized_direction, b.quantized_direction], dim=2),
            group_mins=torch.cat([a.group_mins, b.group_mins], dim=2),
            group_scales=torch.cat([a.group_scales, b.group_scales], dim=2),
            qjl_signs=None, residual_norms=None,
            n_bits=a.n_bits, group_size=a.group_size, original_dim=a.original_dim,
            seq_len=a.seq_len + b.seq_len, max_seq_len=a.max_seq_len + b.max_seq_len,
        )

    def get_kv(self):
        """返回 KV Cache 用于注意力计算。

        Returns:
            压缩模式: (CompressedKVCacheGPU, CompressedKVCacheGPU)
            标准模式: (key_tensor, value_tensor)
        """
        if not self.config.enable_compression:
            return self._key_cache, self._value_cache
        return self._compressed_keys, self._compressed_values

    def reset(self):
        self._key_cache = None
        self._value_cache = None
        self._compressed_keys = None
        self._compressed_values = None
        self._seq_len = 0
