"""压缩多头注意力层。"""

import math
import torch
import torch.nn as nn

from polarquant_kv_cuda.types import CompressedKVCacheGPU
from polarquant_kv_cuda.attention_kernel import compressed_attention_gpu
from polarquant_kv_cuda.inference.config import PolarQuantConfig
from polarquant_kv_cuda.inference.kv_cache import KVCacheManager


class CompressedMultiHeadAttention(nn.Module):
    """可替换标准 MHA 的压缩注意力层。"""

    def __init__(
        self,
        config: PolarQuantConfig,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cache_manager: KVCacheManager | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            cache_manager: KV Cache 管理器（可选）

        Returns:
            [batch, seq_len, embed_dim]
        """
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)

        # [B, S, H, D] -> [B, H, S, D]
        q = q.transpose(1, 2).half()
        k = k.transpose(1, 2).half()
        v = v.transpose(1, 2).half()

        if cache_manager is not None:
            cache_manager.append(k, v)
            output = self._attend_with_cache(q, cache_manager)
        else:
            output = self._attend_standard(q, k, v)

        # [B, H, S, D] -> [B, S, H, D] -> [B, S, E]
        output = output.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(output.float())

    def _attend_standard(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float()
        ).half()

    def _attend_with_cache(self, q, cache_manager: KVCacheManager):
        cached_k, cached_v = cache_manager.get_kv()

        if not cache_manager.config.enable_compression:
            # GQA: 扩展 KV heads 到 Q heads 数量
            k, v = cached_k, cached_v
            if k.shape[1] < q.shape[1]:
                repeat = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            return torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float()
            ).half()

        return compressed_attention_gpu(
            q, cached_k, cached_v,
            cache_manager.R,
            num_kv_heads=self.num_kv_heads if self.num_kv_heads < self.num_heads else None,
        )
