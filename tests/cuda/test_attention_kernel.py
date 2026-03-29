"""需求 4: 融合注意力 Kernel — 单元测试。

覆盖 AC-4.1 ~ AC-4.5, AC-4.7。
"""

import torch
import pytest

from polarquant_kv_cuda.compress_kernel import compress_gpu
from polarquant_kv_cuda.attention_kernel import compressed_attention_gpu
from polarquant_kv_cuda.rotation import generate_rotation_matrix

D = 128
SEED = 42


@pytest.fixture
def R():
    return generate_rotation_matrix(D, seed=SEED, device="cuda")


@pytest.fixture
def P():
    torch.manual_seed(SEED + 1)
    return torch.randn(64, D, dtype=torch.float32, device="cuda") / (64 ** 0.5)


class TestFusedAttention:
    """AC-4.1: 单函数调用完成注意力"""

    def test_output_shape(self, R, P, random_kv_gpu):
        Q = torch.randn(1, 1, 1, D, dtype=torch.float16, device="cuda")
        K = random_kv_gpu(batch=1, num_heads=1, seq_len=16)
        V = random_kv_gpu(batch=1, num_heads=1, seq_len=16)
        ck = compress_gpu(K, R, n_bits=4, group_size=32, jl_matrix=P)
        cv = compress_gpu(V, R, n_bits=4, group_size=32)
        out = compressed_attention_gpu(Q, ck, cv, R, jl_matrix=P)
        assert out.shape == Q.shape
        assert out.is_cuda

    def test_accuracy_vs_standard(self, R, P, random_kv_gpu):
        """AC-4.2: 余弦相似度 ≥ 0.985"""
        torch.manual_seed(SEED)
        Q = torch.randn(1, 1, 1, D, dtype=torch.float16, device="cuda")
        K = random_kv_gpu(batch=1, num_heads=1, seq_len=32)
        V = random_kv_gpu(batch=1, num_heads=1, seq_len=32)

        # 标准注意力
        out_std = torch.nn.functional.scaled_dot_product_attention(
            Q.float(), K.float(), V.float()
        ).half()

        # 压缩注意力
        ck = compress_gpu(K, R, n_bits=4, group_size=32, jl_matrix=P)
        cv = compress_gpu(V, R, n_bits=4, group_size=32)
        out_comp = compressed_attention_gpu(Q, ck, cv, R, jl_matrix=P)

        cos = torch.nn.functional.cosine_similarity(
            out_std.flatten().float(), out_comp.flatten().float(), dim=0
        ).item()
        assert cos >= 0.985, f"余弦相似度 {cos:.4f} < 0.985"


class TestGQA:
    """AC-4.3: GQA 支持"""

    def test_gqa_shape(self, R, P):
        num_q_heads, num_kv_heads, seq = 8, 2, 16
        Q = torch.randn(1, num_q_heads, 1, D, dtype=torch.float16, device="cuda")
        K = torch.randn(1, num_kv_heads, seq, D, dtype=torch.float16, device="cuda")
        V = torch.randn(1, num_kv_heads, seq, D, dtype=torch.float16, device="cuda")
        ck = compress_gpu(K, R, n_bits=4, group_size=32)
        cv = compress_gpu(V, R, n_bits=4, group_size=32)
        out = compressed_attention_gpu(Q, ck, cv, R, num_kv_heads=num_kv_heads)
        assert out.shape == Q.shape


class TestQJLSwitch:
    """AC-4.4: enable_qjl 开关"""

    def test_disable_qjl(self, R, P, random_kv_gpu):
        Q = torch.randn(1, 1, 1, D, dtype=torch.float16, device="cuda")
        K = random_kv_gpu(batch=1, num_heads=1, seq_len=16)
        V = random_kv_gpu(batch=1, num_heads=1, seq_len=16)
        ck = compress_gpu(K, R, n_bits=4, group_size=32, jl_matrix=P)
        cv = compress_gpu(V, R, n_bits=4, group_size=32)
        out = compressed_attention_gpu(Q, ck, cv, R, jl_matrix=P, enable_qjl=False)
        assert out.shape == Q.shape
        assert not torch.any(torch.isnan(out))


class TestEmptySequence:
    """AC-4.5: 空序列"""

    def test_empty_seq_returns_zero(self, R, P):
        Q = torch.randn(1, 1, 1, D, dtype=torch.float16, device="cuda")
        K = torch.empty(1, 1, 0, D, dtype=torch.float16, device="cuda")
        V = torch.empty(1, 1, 0, D, dtype=torch.float16, device="cuda")
        ck = compress_gpu(K, R, n_bits=4, group_size=32)
        cv = compress_gpu(V, R, n_bits=4, group_size=32)
        out = compressed_attention_gpu(Q, ck, cv, R)
        assert torch.allclose(out, torch.zeros_like(out))
