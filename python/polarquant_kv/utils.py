"""工具函数。"""

import math

import numpy as np
import psutil


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度。"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-30 or norm_b < 1e-30:
        return 0.0
    return float(np.dot(a.flatten(), b.flatten()) / (norm_a * norm_b))


def compute_compression_ratio(
    d: int,
    n_bits: int,
    group_size: int,
    jl_dim: int,
) -> float:
    """根据 AC-5.5 公式计算理论压缩比。

    compression_ratio = (d * 16) / (16 + n_bits * d + num_groups * 32 + jl_dim)
    """
    num_groups = math.ceil(d / group_size)
    original_bits = d * 16  # FP16
    compressed_bits = 16 + n_bits * d + num_groups * 32 + jl_dim
    return original_bits / compressed_bits


def attention_score_mse(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """计算注意力分数的均方误差。"""
    return float(np.mean((scores_a - scores_b) ** 2))


def estimate_memory_bytes(
    batch: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    n_bits: int,
    group_size: int,
    jl_dim: int,
) -> int:
    """预估压缩后的内存占用（字节）。"""
    num_groups = math.ceil(head_dim / group_size)
    # 每个向量的压缩存储 (bits)
    per_vector_bits = 16 + n_bits * head_dim + num_groups * 32 + jl_dim
    total_vectors = batch * num_heads * seq_len
    return math.ceil(total_vectors * per_vector_bits / 8)


def check_memory_warning(estimated_bytes: int) -> bool:
    """检查预估内存是否超过可用内存的 80%，返回是否需要警告。"""
    available = psutil.virtual_memory().available
    return estimated_bytes > available * 0.8
