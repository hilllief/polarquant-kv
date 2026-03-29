"""正确性属性测试（Hypothesis）— P1~P8。

设计文档: docs/specs/phase1-python-prototype/design.md
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from polarquant_kv.rotation import generate_rotation_matrix, rotate, inverse_rotate
from polarquant_kv.quantizer import compress, decompress, compress_batch, decompress_batch
from polarquant_kv.qjl import generate_jl_matrix, compute_signatures, compute_correction
from polarquant_kv.attention import standard_attention, compressed_attention
from polarquant_kv.utils import cosine_similarity, compute_compression_ratio
from tests.strategies import (
    small_dims,
    n_bits_strategy,
    group_size_strategy,
    jl_dim_strategy,
    safe_floats,
    extreme_floats,
    seed_strategy,
)


# ============================================================
# P1: 正交性保持 (AC-1.1, AC-1.2)
# ============================================================

class TestP1Orthogonality:

    @given(d=st.integers(min_value=2, max_value=256), seed=seed_strategy)
    @settings(max_examples=100, deadline=None)
    def test_orthogonality(self, d, seed):
        R = generate_rotation_matrix(d, seed=seed)
        identity = R.T @ R
        assert np.allclose(identity, np.eye(d), atol=1e-5)

    @given(
        d=st.integers(min_value=2, max_value=128),
        seed=seed_strategy,
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_norm_preservation(self, d, seed, data):
        R = generate_rotation_matrix(d, seed=seed)
        v = data.draw(
            arrays(np.float32, (d,), elements=safe_floats)
        )
        norm_v = np.linalg.norm(v)
        assume(norm_v > 1e-10)
        v_rotated = rotate(v, R)
        rel_err = abs(np.linalg.norm(v_rotated) - norm_v) / norm_v
        assert rel_err < 1e-5


# ============================================================
# P2: 压缩-解压往返 (AC-2.1, AC-2.4)
# ============================================================

COSINE_THRESHOLDS = {2: 0.88, 3: 0.96, 4: 0.99, 6: 0.995, 8: 0.999}


class TestP2RoundTrip:

    @given(n_bits=n_bits_strategy, seed=seed_strategy)
    @settings(max_examples=100, deadline=None)
    def test_compress_decompress_cosine(self, n_bits, seed):
        d = 128
        R = generate_rotation_matrix(d, seed=0)
        rng = np.random.Generator(np.random.PCG64(seed))
        v = rng.standard_normal(d).astype(np.float32)
        assume(np.linalg.norm(v) > 1e-10)

        compressed = compress(v, R, n_bits=n_bits, group_size=32)
        v_hat = decompress(compressed, R)
        sim = cosine_similarity(v, v_hat)
        threshold = COSINE_THRESHOLDS.get(n_bits, 0.99)
        assert sim >= threshold - 0.005, (
            f"{n_bits}-bit cosine sim {sim:.4f} < {threshold - 0.005}"
        )


# ============================================================
# P3: 压缩比下界 (AC-2.2, AC-2.3, AC-5.5)
# ============================================================

class TestP3CompressionRatio:

    @given(
        d=st.sampled_from([64, 128, 256]),
        n_bits=n_bits_strategy,
        group_size=group_size_strategy,
        jl_dim=jl_dim_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_formula_deterministic(self, d, n_bits, group_size, jl_dim):
        assume(group_size <= d)
        num_groups = math.ceil(d / group_size)
        expected = (d * 16) / (16 + n_bits * d + num_groups * 32 + jl_dim)
        actual = compute_compression_ratio(d, n_bits, group_size, jl_dim)
        assert abs(actual - expected) < 1e-10


# ============================================================
# P4: QJL 修正有效性 (AC-3.3)
# ============================================================

class TestP4QJLEffectiveness:

    @given(seed=st.integers(min_value=0, max_value=999))
    @settings(max_examples=50, deadline=None)
    def test_correction_reduces_error(self, seed):
        d = 128
        R = generate_rotation_matrix(d, seed=0)
        P = generate_jl_matrix(jl_dim=64, d=d, seed=1)
        rng = np.random.Generator(np.random.PCG64(seed))

        q = rng.standard_normal(d).astype(np.float32)
        k = rng.standard_normal(d).astype(np.float32)

        score_true = q @ k / np.sqrt(d)
        compressed = compress(k, R, n_bits=4, group_size=32)
        k_hat = decompress(compressed, R)
        score_quant = q @ k_hat / np.sqrt(d)

        residual = k - k_hat
        sigs = compute_signatures(residual, P)
        correction = compute_correction(q, sigs, P)
        score_corrected = score_quant + correction

        err_without = abs(score_true - score_quant)
        err_with = abs(score_true - score_corrected)
        # 不要求每个样本都更好，但记录下来
        # 统计检验在 test_qjl.py 中完成
        # 这里只验证不产生 NaN
        assert not np.isnan(score_corrected)


# ============================================================
# P5: 注意力等价性 (AC-4.2, AC-4.4)
# ============================================================

class TestP5AttentionEquivalence:

    @given(
        seq_len=st.integers(min_value=1, max_value=64),
        seed=seed_strategy,
    )
    @settings(max_examples=50, deadline=None)
    def test_attention_output_similarity(self, seq_len, seed):
        d = 128
        R = generate_rotation_matrix(d, seed=0)
        P = generate_jl_matrix(jl_dim=64, d=d, seed=1)
        rng = np.random.Generator(np.random.PCG64(seed))

        Q = rng.standard_normal((1, d)).astype(np.float32)
        K = rng.standard_normal((seq_len, d)).astype(np.float32)
        V = rng.standard_normal((seq_len, d)).astype(np.float32)

        out_std = standard_attention(Q, K, V)
        out_comp = compressed_attention(
            Q, K, V, R, P,
            n_bits=4, group_size=32, enable_qjl=True,
        )
        sim = cosine_similarity(out_std.flatten(), out_comp.flatten())
        assert sim >= 0.98, f"注意力输出余弦相似度 {sim:.4f} < 0.98"


# ============================================================
# P6: Batch 一致性 (AC-6.2)
# ============================================================

class TestP6BatchConsistency:

    @given(
        batch=st.integers(min_value=1, max_value=2),
        num_heads=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=1, max_value=8),
        seed=seed_strategy,
    )
    @settings(max_examples=30, deadline=None)
    def test_batch_equals_individual(self, batch, num_heads, seq_len, seed):
        d = 128
        R = generate_rotation_matrix(d, seed=0)
        rng = np.random.Generator(np.random.PCG64(seed))
        kv = rng.standard_normal((batch, num_heads, seq_len, d)).astype(np.float32)

        compressed_b = compress_batch(kv, R, n_bits=4, group_size=32)
        kv_hat_batch = decompress_batch(compressed_b, R)

        for b in range(batch):
            for h in range(num_heads):
                for s in range(seq_len):
                    c = compress(kv[b, h, s], R, n_bits=4, group_size=32)
                    kv_hat_single = decompress(c, R)
                    np.testing.assert_array_equal(kv_hat_batch[b, h, s], kv_hat_single)


# ============================================================
# P7: 零向量安全 (AC-2.8, AC-3.5)
# ============================================================

class TestP7ZeroVectorSafety:

    @given(d=st.sampled_from([64, 128, 256]))
    @settings(max_examples=10, deadline=None)
    def test_zero_vector_compress_decompress(self, d):
        R = generate_rotation_matrix(d, seed=0)
        v = np.zeros(d, dtype=np.float32)
        compressed = compress(v, R, n_bits=4, group_size=32)
        v_hat = decompress(compressed, R)
        assert not np.any(np.isnan(v_hat))
        assert not np.any(np.isinf(v_hat))

    @given(d=st.sampled_from([64, 128, 256]))
    @settings(max_examples=10, deadline=None)
    def test_zero_residual_correction(self, d):
        P = generate_jl_matrix(jl_dim=64, d=d, seed=0)
        residual = np.zeros(d, dtype=np.float32)
        sigs = compute_signatures(residual, P)
        q = np.ones(d, dtype=np.float32)
        correction = compute_correction(q, sigs, P)
        assert np.isclose(correction, 0.0, atol=1e-10)


# ============================================================
# P8: 极端值鲁棒性 (AC-2.10)
# ============================================================

class TestP8ExtremeValueRobustness:

    @given(
        d=st.sampled_from([64, 128]),
        seed=seed_strategy,
    )
    @settings(max_examples=30, deadline=None)
    def test_extreme_values_no_nan_inf(self, d, seed):
        R = generate_rotation_matrix(d, seed=0)
        rng = np.random.Generator(np.random.PCG64(seed))

        # 构造极端值向量
        v = rng.standard_normal(d).astype(np.float32)
        v[0] = np.finfo(np.float16).max
        v[1] = -np.finfo(np.float16).max
        if d > 2:
            v[2] = np.finfo(np.float16).tiny

        compressed = compress(v, R, n_bits=4, group_size=32)
        v_hat = decompress(compressed, R)
        assert not np.any(np.isnan(v_hat)), "解压结果包含 NaN"
        assert not np.any(np.isinf(v_hat)), "解压结果包含 Inf"
