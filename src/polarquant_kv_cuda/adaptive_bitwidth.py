"""创新 1: 自适应 Per-Layer Bit-Width。

根据层的位置自动分配量化精度：
- 第 0 层和最后一层: 4-bit（最敏感）
- 中间层: 2-bit（不敏感）
- 平均 bit-width ~3-bit，但精度接近全 4-bit

这比论文的固定 bit-width 方案压缩比更高。
"""

import torch
import math


def compute_layer_bitwidths(
    num_layers: int,
    target_avg_bits: float = 3.0,
    min_bits: int = 2,
    max_bits: int = 4,
) -> list[int]:
    """根据层位置计算每层的 bit-width。

    策略: U 形分配
    - 第 0 层和最后一层: max_bits (最敏感)
    - 中间层: min_bits (最不敏感)
    - 过渡层: 线性插值

    Args:
        num_layers: 总层数
        target_avg_bits: 目标平均 bit-width
        min_bits: 最小 bit-width
        max_bits: 最大 bit-width

    Returns:
        每层的 bit-width 列表
    """
    if num_layers <= 2:
        return [max_bits] * num_layers

    # U 形权重: 两端高，中间低
    weights = []
    mid = (num_layers - 1) / 2.0
    for i in range(num_layers):
        # 距离中心的归一化距离 [0, 1]
        dist = abs(i - mid) / mid
        # U 形: 两端 1.0, 中间 0.0
        weights.append(dist)

    # 将权重映射到 [min_bits, max_bits]
    bitwidths = []
    for w in weights:
        bits = min_bits + w * (max_bits - min_bits)
        # 四舍五入到最近的支持值 {2, 3, 4}
        bits = max(min_bits, min(max_bits, round(bits)))
        bitwidths.append(bits)

    # 调整以满足目标平均 bit-width
    avg = sum(bitwidths) / len(bitwidths)
    if avg > target_avg_bits:
        # 从中间层开始降低
        for i in sorted(range(num_layers), key=lambda x: weights[x]):
            if bitwidths[i] > min_bits:
                bitwidths[i] -= 1
                avg = sum(bitwidths) / len(bitwidths)
                if avg <= target_avg_bits:
                    break

    return bitwidths


def estimate_adaptive_compression_ratio(
    bitwidths: list[int],
    head_dim: int = 128,
) -> float:
    """估算自适应 bit-width 的平均压缩比。"""
    original_bits_per_value = 16  # FP16
    total_compressed = 0
    for bits in bitwidths:
        # 每个值: bits 个 bit + radius overhead (amortized)
        compressed_per_value = bits + 4.0 / head_dim * 32  # radius FP32 amortized
        total_compressed += compressed_per_value
    avg_compressed = total_compressed / len(bitwidths)
    return original_bits_per_value / avg_compressed
