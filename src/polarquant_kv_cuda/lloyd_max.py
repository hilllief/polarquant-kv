"""Lloyd-Max 最优标量量化器（基于 Beta 分布）。

TurboQuant 的核心创新：随机旋转后每个坐标服从 Beta((d-1)/2, (d-1)/2)，
可以预计算最优 codebook，不需要 per-group min/max。
"""

import numpy as np
from scipy.stats import beta as beta_dist
import torch


def build_lloyd_max_codebook(
    dim: int,
    n_bits: int,
    n_iterations: int = 300,
    grid_size: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """构建 Lloyd-Max 最优 codebook。

    Args:
        dim: 向量维度（决定 Beta 分布参数）
        n_bits: 量化位数
        n_iterations: Lloyd-Max 迭代次数
        grid_size: 数值积分网格大小

    Returns:
        (centroids, boundaries)
        centroids: [2^n_bits] 个质心值
        boundaries: [2^n_bits + 1] 个分界点
    """
    n_levels = 1 << n_bits
    alpha = (dim - 1) / 2.0

    # Beta((d-1)/2, (d-1)/2) 在 [-1, 1] 上
    # scipy 的 beta 分布在 [0, 1]，需要变换
    sigma = 1.0 / np.sqrt(dim)
    x_range = 6 * sigma  # ±6σ 覆盖 99.99% 的概率质量

    # 在 [-x_range, x_range] 上建立密集网格
    grid = np.linspace(-x_range, x_range, grid_size)

    # Beta 分布的 PDF（变换到 [-1, 1]）
    # x ∈ [-1, 1] → t = (x+1)/2 ∈ [0, 1]
    t_grid = (grid + 1) / 2
    t_grid = np.clip(t_grid, 1e-10, 1 - 1e-10)
    pdf = beta_dist.pdf(t_grid, alpha, alpha) / 2  # 除以 2 是 Jacobian

    # 归一化
    pdf = pdf / (pdf.sum() * (grid[1] - grid[0]))

    # 初始化质心：均匀分布
    centroids = np.linspace(-x_range, x_range, n_levels)

    # Lloyd-Max 迭代
    for _ in range(n_iterations):
        # 计算分界点（相邻质心的中点）
        boundaries = np.zeros(n_levels + 1)
        boundaries[0] = -x_range * 2
        boundaries[-1] = x_range * 2
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

        # 更新质心（每个区间的加权均值）
        new_centroids = np.zeros(n_levels)
        for i in range(n_levels):
            mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
            if mask.sum() > 0:
                weights = pdf[mask]
                w_sum = weights.sum()
                if w_sum > 0:
                    new_centroids[i] = (grid[mask] * weights).sum() / w_sum
                else:
                    new_centroids[i] = centroids[i]
            else:
                new_centroids[i] = centroids[i]

        centroids = new_centroids

    return centroids.astype(np.float32), boundaries.astype(np.float32)


def quantize_with_codebook(
    values: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """用 codebook 量化：找最近的 centroid。"""
    # values: [...], centroids: [n_levels]
    # 返回 indices: [...], uint8
    diffs = np.abs(values[..., np.newaxis] - centroids)
    return diffs.argmin(axis=-1).astype(np.uint8)


def dequantize_with_codebook(
    indices: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """用 codebook 反量化：查表。"""
    return centroids[indices]


def get_codebook_torch(dim: int, n_bits: int, device="cuda") -> torch.Tensor:
    """获取 codebook 的 PyTorch tensor（缓存）。"""
    centroids, _ = build_lloyd_max_codebook(dim, n_bits)
    return torch.from_numpy(centroids).float().to(device)
