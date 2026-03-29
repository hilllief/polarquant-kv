"""GPU 数据结构。"""

from dataclasses import dataclass
import torch


@dataclass
class CompressedKVCacheGPU:
    """GPU 上的压缩 KV Cache。"""

    radius: torch.Tensor            # [batch, num_heads, seq_len], FP16
    quantized_direction: torch.Tensor  # [batch, num_heads, seq_len, d_padded], uint8
    group_mins: torch.Tensor        # [batch, num_heads, seq_len, num_groups], FP16
    group_scales: torch.Tensor      # [batch, num_heads, seq_len, num_groups], FP16
    qjl_signs: torch.Tensor | None  # [batch, num_heads, seq_len, jl_dim_packed], uint8
    residual_norms: torch.Tensor | None  # [batch, num_heads, seq_len], FP16
    n_bits: int
    group_size: int
    original_dim: int
    seq_len: int                    # 当前有效序列长度
    max_seq_len: int                # 预分配最大序列长度
