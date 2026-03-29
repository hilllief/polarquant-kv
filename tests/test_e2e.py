"""E2E 测试 — 端到端业务流程。

场景：生成 KV → 压缩 → 压缩注意力计算 → 精度验证。
"""

import numpy as np
import pytest

from polarquant_kv.rotation import generate_rotation_matrix
from polarquant_kv.quantizer import compress, decompress, compress_batch, decompress_batch
from polarquant_kv.qjl import generate_jl_matrix, compute_signatures, compute_correction
from polarquant_kv.attention import standard_attention, compressed_attention
from polarquant_kv.benchmark import evaluate_config, hyperparameter_search, generate_phase1_report
from polarquant_kv.utils import cosine_similarity

SEED = 42
D = 128


class TestSingleHeadSingleToken:
    """单 head 单 token 完整流程。"""

    def test_full_pipeline(self):
        R = generate_rotation_matrix(D, seed=SEED)
        P = generate_jl_matrix(jl_dim=64, d=D, seed=SEED)
        rng = np.random.Generator(np.random.PCG64(SEED))

        # 1. 生成 Q, K, V
        Q = rng.standard_normal((1, D)).astype(np.float32)
        K = rng.standard_normal((1, D)).astype(np.float32)
        V = rng.standard_normal((1, D)).astype(np.float32)

        # 2. 标准注意力
        out_std = standard_attention(Q, K, V)

        # 3. 压缩注意力
        out_comp = compressed_attention(
            Q, K, V, R, P,
            n_bits=4, group_size=32, enable_qjl=True,
        )

        # 4. 精度验证
        sim = cosine_similarity(out_std.flatten(), out_comp.flatten())
        assert sim >= 0.99, f"单 token 余弦相似度 {sim:.4f}"
        assert out_comp.shape == out_std.shape


class TestMultiHeadMultiToken:
    """多 head 多 token batch 流程。"""

    def test_batch_pipeline(self):
        R = generate_rotation_matrix(D, seed=SEED)
        P = generate_jl_matrix(jl_dim=64, d=D, seed=SEED)
        rng = np.random.Generator(np.random.PCG64(SEED))

        num_heads = 4
        seq_len = 32

        # 对每个 head 独立测试
        for h in range(num_heads):
            Q = rng.standard_normal((1, D)).astype(np.float32)
            K = rng.standard_normal((seq_len, D)).astype(np.float32)
            V = rng.standard_normal((seq_len, D)).astype(np.float32)

            out_std = standard_attention(Q, K, V)
            out_comp = compressed_attention(
                Q, K, V, R, P,
                n_bits=4, group_size=32, enable_qjl=True,
            )
            sim = cosine_similarity(out_std.flatten(), out_comp.flatten())
            assert sim >= 0.99, f"Head {h} 余弦相似度 {sim:.4f}"


class TestGQAPipeline:
    """GQA 场景完整流程。"""

    def test_gqa_e2e(self):
        R = generate_rotation_matrix(D, seed=SEED)
        P = generate_jl_matrix(jl_dim=64, d=D, seed=SEED)
        rng = np.random.Generator(np.random.PCG64(SEED))

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
        # 逐 head 检查
        for h in range(num_q_heads):
            sim = cosine_similarity(out_std[h].flatten(), out_comp[h].flatten())
            assert sim >= 0.98, f"GQA head {h} 余弦相似度 {sim:.4f}"


class TestHyperparamSearchPipeline:
    """超参数搜索 + 报告生成完整流程。"""

    def test_search_and_report(self):
        results = hyperparameter_search(
            n_bits_range=[4, 8],
            group_size_range=[32],
            jl_dim_range=[64],
            d=128,
            num_samples=10,
        )
        assert len(results["configs"]) == 2

        report = generate_phase1_report(results)
        assert isinstance(report, str)
        assert len(report) > 50  # 报告不应太短
        assert "压缩比" in report or "compression" in report.lower()


class TestBatchCompressDecompressE2E:
    """Batch 压缩解压端到端。"""

    def test_batch_roundtrip(self):
        R = generate_rotation_matrix(D, seed=SEED)
        rng = np.random.Generator(np.random.PCG64(SEED))

        batch, num_heads, seq_len = 2, 4, 16
        kv = rng.standard_normal((batch, num_heads, seq_len, D)).astype(np.float32)

        compressed = compress_batch(kv, R, n_bits=4, group_size=32)
        kv_hat = decompress_batch(compressed, R)

        assert kv_hat.shape == kv.shape
        # 每个向量的余弦相似度
        for b in range(batch):
            for h in range(num_heads):
                for s in range(seq_len):
                    sim = cosine_similarity(kv[b, h, s], kv_hat[b, h, s])
                    assert sim >= 0.99, (
                        f"Batch[{b}][{h}][{s}] 余弦相似度 {sim:.4f}"
                    )
