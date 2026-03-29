"""简单 Transformer 层（测试用）。"""

import torch
import torch.nn as nn

from polarquant_kv_cuda.inference.config import PolarQuantConfig
from polarquant_kv_cuda.inference.attention import CompressedMultiHeadAttention
from polarquant_kv_cuda.inference.kv_cache import KVCacheManager


class SimpleTransformerLayer(nn.Module):
    """单层 Transformer: Attention + FFN。"""

    def __init__(self, config: PolarQuantConfig, embed_dim: int, num_heads: int,
                 num_kv_heads: int | None = None, ffn_dim: int | None = None):
        super().__init__()
        self.attn = CompressedMultiHeadAttention(config, embed_dim, num_heads, num_kv_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        ffn_dim = ffn_dim or embed_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim, bias=False),
        )

    def forward(self, x: torch.Tensor, cache_manager: KVCacheManager | None = None):
        h = self.norm1(x.float())
        x = x + self.attn(h, cache_manager)
        x = x + self.ffn(self.norm2(x.float()))
        return x


class SimpleTransformer(nn.Module):
    """多层 Transformer。"""

    def __init__(self, config: PolarQuantConfig, num_layers: int, embed_dim: int,
                 num_heads: int, num_kv_heads: int | None = None, vocab_size: int = 1000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(config, embed_dim, num_heads, num_kv_heads)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.config = config
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self, input_ids: torch.Tensor, cache_managers: list[KVCacheManager] | None = None):
        x = self.embed(input_ids).float()
        for i, layer in enumerate(self.layers):
            cm = cache_managers[i] if cache_managers else None
            x = layer(x, cm)
        return self.head(x.float())

    def create_cache_managers(self, max_seq_len: int, batch: int, device: str = "cuda"):
        return [
            KVCacheManager(
                self.config, max_seq_len, batch,
                self.num_kv_heads, self.head_dim, device,
                rotation_seed=42 + i,
            )
            for i in range(len(self.layers))
        ]
