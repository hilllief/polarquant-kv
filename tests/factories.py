"""测试数据工厂函数。"""

import numpy as np


def make_random_vector(d: int, rng: np.random.Generator, dtype=np.float32) -> np.ndarray:
    """生成随机向量。"""
    return rng.standard_normal(d).astype(dtype)


def make_zero_vector(d: int, dtype=np.float32) -> np.ndarray:
    """生成零向量。"""
    return np.zeros(d, dtype=dtype)


def make_extreme_vector(d: int) -> np.ndarray:
    """生成包含极端值的向量（FP16 范围）。"""
    v = np.zeros(d, dtype=np.float32)
    v[0] = np.finfo(np.float16).max  # 65504.0
    v[1] = np.finfo(np.float16).tiny  # 最小正 subnormal
    v[2] = -np.finfo(np.float16).max
    v[3] = np.finfo(np.float16).smallest_subnormal
    if d > 4:
        rng = np.random.Generator(np.random.PCG64(99))
        v[4:] = rng.standard_normal(d - 4).astype(np.float32)
    return v


def make_batch_kv(
    batch: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    rng: np.random.Generator,
    dtype=np.float32,
) -> np.ndarray:
    """生成 batch KV 向量 [batch, num_heads, seq_len, head_dim]。"""
    return rng.standard_normal((batch, num_heads, seq_len, head_dim)).astype(dtype)
