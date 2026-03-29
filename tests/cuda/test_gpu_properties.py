"""正确性属性测试（GPU 版）— P1~P6。"""

import numpy as np
import torch
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from polarquant_kv_cuda.compress_kernel import compress_gpu
from polarquant_kv_cuda.decompress_kernel import decompress_gpu
from polarquant_kv_cuda.rotation import generate_rotation_matrix
from polarquant_kv_cuda.qjl_kernel import bit_pack, bit_unpack

D = 128


class TestP1GPUCPUConsistency:
    """P1: GPU-CPU 数值一致性"""

    @given(seed=st.integers(min_value=0, max_value=999))
    @settings(max_examples=30, deadline=None)
    def test_compress_consistency(self, seed):
        from polarquant_kv.rotation import generate_rotation_matrix as gen_R_cpu
        from polarquant_kv.quantizer import compress as compress_cpu, decompress as decompress_cpu

        R_cpu = gen_R_cpu(D, seed=0)
        R_gpu = generate_rotation_matrix(D, seed=0, device="cuda")

        rng = np.random.Generator(np.random.PCG64(seed))
        v_np = rng.standard_normal(D).astype(np.float32)
        assume(np.linalg.norm(v_np) > 1e-6)

        c_cpu = compress_cpu(v_np, R_cpu, n_bits=4, group_size=32)
        v_hat_cpu = decompress_cpu(c_cpu, R_cpu)

        v_gpu = torch.from_numpy(v_np).half().cuda().reshape(1, 1, 1, D)
        c_gpu = compress_gpu(v_gpu, R_gpu, n_bits=4, group_size=32)
        v_hat_gpu = decompress_gpu(c_gpu, R_gpu).squeeze().cpu().float().numpy()

        cos = np.dot(v_hat_cpu, v_hat_gpu) / (
            np.linalg.norm(v_hat_cpu) * np.linalg.norm(v_hat_gpu) + 1e-30
        )
        assert cos >= 0.999, f"GPU-CPU cos {cos:.4f} < 0.999"


class TestP2RoundTripGPU:
    """P2: 压缩-解压往返 GPU"""

    @given(seed=st.integers(min_value=0, max_value=999))
    @settings(max_examples=50, deadline=None)
    def test_roundtrip_cosine(self, seed):
        R = generate_rotation_matrix(D, seed=0, device="cuda")
        torch.manual_seed(seed)
        v = torch.randn(1, 1, 1, D, dtype=torch.float16, device="cuda")
        norm = v.float().norm()
        assume(norm.item() > 1e-6)

        compressed = compress_gpu(v, R, n_bits=4, group_size=32)
        v_hat = decompress_gpu(compressed, R)
        cos = torch.nn.functional.cosine_similarity(
            v.flatten().float(), v_hat.flatten().float(), dim=0
        ).item()
        assert cos >= 0.99, f"Roundtrip cos {cos:.4f} < 0.99"


class TestP3FusedAttentionEquivalence:
    """P3: 融合注意力等价性"""

    @given(seq_len=st.integers(min_value=1, max_value=64), seed=st.integers(0, 999))
    @settings(max_examples=20, deadline=None)
    def test_attention_cosine(self, seq_len, seed):
        from polarquant_kv_cuda.attention_kernel import compressed_attention_gpu

        R = generate_rotation_matrix(D, seed=0, device="cuda")
        torch.manual_seed(seed)
        Q = torch.randn(1, 1, 1, D, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 1, seq_len, D, dtype=torch.float16, device="cuda")
        V = torch.randn(1, 1, seq_len, D, dtype=torch.float16, device="cuda")

        out_std = torch.nn.functional.scaled_dot_product_attention(
            Q.float(), K.float(), V.float()
        ).half()

        ck = compress_gpu(K, R, n_bits=4, group_size=32)
        cv = compress_gpu(V, R, n_bits=4, group_size=32)
        out_comp = compressed_attention_gpu(Q, ck, cv, R)

        cos = torch.nn.functional.cosine_similarity(
            out_std.flatten().float(), out_comp.flatten().float(), dim=0
        ).item()
        assert cos >= 0.98, f"Attention cos {cos:.4f} < 0.98"


class TestP4ZeroVectorGPU:
    """P4: 零向量安全 GPU"""

    @given(d=st.sampled_from([64, 128, 256]))
    @settings(max_examples=10, deadline=None)
    def test_zero_vector_safe(self, d):
        R = generate_rotation_matrix(d, seed=0, device="cuda")
        v = torch.zeros(1, 1, 1, d, dtype=torch.float16, device="cuda")
        compressed = compress_gpu(v, R, n_bits=4, group_size=32)
        v_hat = decompress_gpu(compressed, R)
        assert not torch.any(torch.isnan(v_hat))
        assert not torch.any(torch.isinf(v_hat))


class TestP5IncrementalConsistency:
    """P5: 增量追加一致性"""

    @given(seq_len=st.integers(min_value=1, max_value=16), seed=st.integers(0, 999))
    @settings(max_examples=20, deadline=None)
    def test_incremental_equals_full(self, seq_len, seed):
        R = generate_rotation_matrix(D, seed=0, device="cuda")
        torch.manual_seed(seed)
        kv = torch.randn(1, 1, seq_len, D, dtype=torch.float16, device="cuda")

        full_hat = decompress_gpu(compress_gpu(kv, R, n_bits=4, group_size=32), R)

        parts = []
        for t in range(seq_len):
            single = compress_gpu(kv[:, :, t:t+1, :], R, n_bits=4, group_size=32)
            parts.append(decompress_gpu(single, R))
        incr_hat = torch.cat(parts, dim=2)

        assert torch.allclose(full_hat, incr_hat, atol=0.01)


class TestP6BitPackingRoundTrip:
    """P6: Bit Packing 往返"""

    @given(jl_dim=st.sampled_from([32, 64, 128]), seed=st.integers(0, 999))
    @settings(max_examples=30, deadline=None)
    def test_pack_unpack(self, jl_dim, seed):
        torch.manual_seed(seed)
        signs = torch.randint(0, 2, (jl_dim,), dtype=torch.bool, device="cuda")
        packed = bit_pack(signs)
        unpacked = bit_unpack(packed, jl_dim)
        assert torch.equal(signs, unpacked)
