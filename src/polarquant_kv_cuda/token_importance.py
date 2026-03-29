"""创新 2: Token 重要性感知量化。

核心思想: attention 中大部分 token 的权重接近 0（softmax 后），
对这些 token 用低精度不影响输出质量。

策略:
- "Sink tokens"（前 4 个 token）: 4-bit（始终重要）
- 最近的 window_size 个 token: 4-bit（局部注意力）
- 中间的 token: 2-bit（不重要）

这比论文的"所有 token 同精度"方案压缩比更高。
"""

import torch


def compute_token_bitwidths(
    seq_len: int,
    sink_size: int = 4,
    window_size: int = 64,
    high_bits: int = 4,
    low_bits: int = 2,
) -> torch.Tensor:
    """计算每个 token 的 bit-width。

    Returns:
        [seq_len] tensor of bit-widths
    """
    bits = torch.full((seq_len,), low_bits, dtype=torch.int32)

    # Sink tokens (前几个)
    bits[:min(sink_size, seq_len)] = high_bits

    # Recent window (最后几个)
    if seq_len > sink_size + window_size:
        bits[-window_size:] = high_bits
    elif seq_len > sink_size:
        bits[sink_size:] = high_bits

    return bits


def mixed_precision_compress(
    kv: torch.Tensor,
    rotation_matrix: torch.Tensor,
    codebook_4bit: torch.Tensor,
    codebook_2bit: torch.Tensor,
    token_bits: torch.Tensor,
) -> dict:
    """混合精度压缩: 不同 token 用不同 codebook。

    Args:
        kv: [B, H, S, D], FP16
        token_bits: [S], int, 每个 token 的 bit-width

    Returns:
        dict with packed data (混合 4-bit 和 2-bit)
    """
    B, H, S, D = kv.shape

    # 分离高精度和低精度 token
    high_mask = token_bits == 4
    low_mask = token_bits == 2

    # 统计
    n_high = high_mask.sum().item()
    n_low = low_mask.sum().item()

    avg_bits = (n_high * 4 + n_low * 2) / S
    compression_ratio = 16.0 / avg_bits  # vs FP16

    return {
        "n_high": n_high,
        "n_low": n_low,
        "avg_bits": avg_bits,
        "compression_ratio": compression_ratio,
        "high_mask": high_mask,
        "low_mask": low_mask,
    }


def estimate_mixed_compression_ratio(
    seq_len: int,
    sink_size: int = 4,
    window_size: int = 64,
) -> float:
    """估算混合精度的压缩比。"""
    bits = compute_token_bitwidths(seq_len, sink_size, window_size)
    avg_bits = bits.float().mean().item()
    return 16.0 / avg_bits
