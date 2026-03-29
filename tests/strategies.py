"""Hypothesis 自定义策略。"""

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


# --- 维度策略 ---
small_dims = st.integers(min_value=2, max_value=64)
medium_dims = st.integers(min_value=32, max_value=256)
typical_head_dim = st.sampled_from([64, 128, 256])

# --- 量化参数策略 ---
n_bits_strategy = st.sampled_from([2, 3, 4, 6, 8])
group_size_strategy = st.sampled_from([16, 32, 64, 128])
jl_dim_strategy = st.sampled_from([32, 64, 128])

# --- 向量策略 ---
safe_floats = st.floats(
    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
)

extreme_floats = st.floats(
    allow_nan=False, allow_infinity=False, allow_subnormal=True
)


def vector_strategy(d: int, elements=safe_floats):
    """生成 d 维 float32 向量。"""
    return arrays(dtype=np.float32, shape=(d,), elements=elements)


def nonzero_vector_strategy(d: int, elements=safe_floats):
    """生成非零 d 维 float32 向量。"""
    return vector_strategy(d, elements).filter(lambda v: np.linalg.norm(v) > 1e-10)


# --- Batch 参数策略 ---
small_batch = st.integers(min_value=1, max_value=4)
small_num_heads = st.integers(min_value=1, max_value=8)
small_seq_len = st.integers(min_value=1, max_value=32)

# --- 种子策略 ---
seed_strategy = st.integers(min_value=0, max_value=2**31 - 1)
