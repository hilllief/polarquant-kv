"""需求 4: 压缩注意力计算 — 单元测试。

覆盖 AC-4.1 ~ AC-4.7。
"""

import numpy as np
import pytest

from polarquant_kv.attention import standard_attention, compressed_attention
from polarquant_kv.quantizer import compress, decompress
from polarquant_kv.rotation import generate_rotation_matrix
from polarquant_kv.qjl import generate_jl_matrix, compute_signatures
from polarquant_kv.utils import cosine_similarity

SEED = 42
D = 128


@pytest.fixture
def R():
    return generate_rotation_matrix(D, seed=SEED)


@pytest.fixture
def P():
    return generate_jl_matrix(jl_dim=64, d=D, seed=SEED)


@pytest.fixture
def rng():
    return np.random.Generator(np.random.PCG64(SEED))


def _compress_keys_with_sigs(keys, R, P, n_bits=4, group_size=32):
    """辅助：压缩 keys 并计算 QJL 签名。"""
    seq_len = keys.shape[0]
    compressed_list = []
    sigs_list = []
    for i in range(seq_len):
        c = compress(keys[i], R, n_bits=n_bits, group_size=group_size)
        k_hat = decompress(c, R)
        residual = keys[i] - k_hat
        sig = compute_signatures(residual, P)
        compressed_list.append(c)
        sigs_list.append(sig)
    return compressed_list, sigs_list


class TestCompressedAttentionScore:
    """AC-4.1: 压缩注意力分数计算"""

    def test_score_shape(self, R, P, rng):
        seq_len = 16
        Q = rng.standard_normal((1, D)).astype(np.float32)
        K = rng.standard_normal((seq_len, D)).astype(np.float32)
        V = rng.standard_normal((seq_len, D)).astype(np.float32)

        output = compressed_attention(
            Q, K, V, R, P,
            n_bits=4, group_size=32, enable_qjl=True,
        )
        assert output.shape == (1, D)

    def test_score_close_to_standard(self, R, P, rng):
        seq_len = 16
        Q = rng.standard_normal((1, D)).astype(np.float32)
        K = rng.standard_normal((seq_len, D)).astype(np.float32)
        V = rng.standard_normal((seq_len, D)).astype(np.float32)

        out_std = standard_attention(Q, K, V)
        out_comp = compressed_attention(
            Q, K, V, R, P,
            n_bits=4, group_size=32, enable_qjl=True,
        )
        sim = cosine_similarity(out_std.flatten(), out_comp.flatten())
        assert sim >= 0.98, f"注意力输出余弦相似度 {sim:.4f} < 0.98"


class TestQJLCorrectionAccuracy:
    """AC-4.2: QJL 修正后注意力分数误差"""

    def test_mean_abs_error_reasonable(self, R, P):
        """注意力分数的平均绝对误差在合理范围内。"""
        mean_errors = []
        for i in range(50):
            rng = np.random.Generator(np.random.PCG64(i))
            seq_len = min(64, rng.integers(1, 128))
            Q = rng.standard_normal((1, D)).astype(np.float32)
            K = rng.standard_normal((seq_len, D)).astype(np.float32)
            V = rng.standard_normal((seq_len, D)).astype(np.float32)

            scores_true = Q @ K.T / np.sqrt(D)
            out_comp = compressed_attention(
                Q, K, V, R, P,
                n_bits=4, group_size=32, enable_qjl=True,
                return_scores=True,
            )
            scores_comp = out_comp["scores"]
            mean_errors.append(np.mean(np.abs(scores_true - scores_comp)))

        avg_mean_error = np.mean(mean_errors)
        # 4-bit 量化的平均 score 误差应 < 0.1
        assert avg_mean_error < 0.1, f"平均绝对误差 {avg_mean_error:.4f} >= 0.1"


class TestFullAttentionOutput:
    """AC-4.3, AC-4.4: 完整注意力输出 + 余弦相似度"""

    def test_full_output_cosine_similarity(self, R, P, rng):
        seq_len = 32
        Q = rng.standard_normal((1, D)).astype(np.float32)
        K = rng.standard_normal((seq_len, D)).astype(np.float32)
        V = rng.standard_normal((seq_len, D)).astype(np.float32)

        out_std = standard_attention(Q, K, V)
        out_comp = compressed_attention(
            Q, K, V, R, P,
            n_bits=4, group_size=32, enable_qjl=True,
        )
        sim = cosine_similarity(out_std.flatten(), out_comp.flatten())
        assert sim >= 0.985, f"注意力输出余弦相似度 {sim:.4f} < 0.985"

    def test_multi_query(self, R, P, rng):
        """多个 query 的注意力输出。"""
        n_queries = 4
        seq_len = 16
        Q = rng.standard_normal((n_queries, D)).astype(np.float32)
        K = rng.standard_normal((seq_len, D)).astype(np.float32)
        V = rng.standard_normal((seq_len, D)).astype(np.float32)

        out_std = standard_attention(Q, K, V)
        out_comp = compressed_attention(
            Q, K, V, R, P,
            n_bits=4, group_size=32, enable_qjl=True,
        )
        assert out_comp.shape == out_std.shape
        for i in range(n_queries):
            sim = cosine_similarity(out_std[i], out_comp[i])
            assert sim >= 0.98, f"Query {i} 余弦相似度 {sim:.4f} < 0.98"


class TestEmptySequence:
    """AC-4.5: 空序列"""

    def test_empty_seq_returns_empty(self, R, P, rng):
        Q = rng.standard_normal((1, D)).astype(np.float32)
        K = np.empty((0, D), dtype=np.float32)
        V = np.empty((0, D), dtype=np.float32)

        out = compressed_attention(
            Q, K, V, R, P,
            n_bits=4, group_size=32, enable_qjl=True,
        )
        assert out.shape[0] == 1
        # 空序列的注意力输出应为零向量
        assert np.allclose(out, 0.0)


class TestGQA:
    """AC-4.6: Grouped Query Attention"""

    def test_gqa_num_kv_heads_less_than_q_heads(self, R, P, rng):
        num_q_heads = 8
        num_kv_heads = 2
        seq_len = 16

        Q = rng.standard_normal((num_q_heads, seq_len, D)).astype(np.float32)
        K = rng.standard_normal((num_kv_heads, seq_len, D)).astype(np.float32)
        V = rng.standard_normal((num_kv_heads, seq_len, D)).astype(np.float32)

        out_std = standard_attention(Q, K, V, num_kv_heads=num_kv_heads)
        out_comp = compressed_attention(
            Q, K, V, R, P,
            n_bits=4, group_size=32, enable_qjl=True,
            num_kv_heads=num_kv_heads,
        )
        assert out_comp.shape == out_std.shape


class TestEnableQJLSwitch:
    """AC-4.7: enable_qjl 开关"""

    def test_disable_qjl(self, R, P, rng):
        seq_len = 16
        Q = rng.standard_normal((1, D)).astype(np.float32)
        K = rng.standard_normal((seq_len, D)).astype(np.float32)
        V = rng.standard_normal((seq_len, D)).astype(np.float32)

        out_with = compressed_attention(
            Q, K, V, R, P,
            n_bits=4, group_size=32, enable_qjl=True,
        )
        out_without = compressed_attention(
            Q, K, V, R, P,
            n_bits=4, group_size=32, enable_qjl=False,
        )
        # 两者应该不同（QJL 修正有效果）
        assert not np.allclose(out_with, out_without, atol=1e-8)

    def test_disable_qjl_still_works(self, R, P, rng):
        """禁用 QJL 后仍能正常计算。"""
        seq_len = 16
        Q = rng.standard_normal((1, D)).astype(np.float32)
        K = rng.standard_normal((seq_len, D)).astype(np.float32)
        V = rng.standard_normal((seq_len, D)).astype(np.float32)

        out = compressed_attention(
            Q, K, V, R, P,
            n_bits=4, group_size=32, enable_qjl=False,
        )
        assert out.shape == (1, D)
        assert not np.any(np.isnan(out))
