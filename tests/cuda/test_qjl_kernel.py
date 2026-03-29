"""需求 3: QJL 投影 Kernel — 单元测试。

覆盖 AC-3.1 ~ AC-3.3。
"""

import torch
import pytest

from polarquant_kv_cuda.qjl_kernel import (
    compute_signatures_gpu,
    bit_pack,
    bit_unpack,
)

D = 128
JL_DIM = 64


class TestSignatureComputation:
    """AC-3.1: JL 投影 + 符号位提取"""

    def test_output_shapes(self, jl_matrix_gpu):
        residual = torch.randn(1, 1, 4, D, dtype=torch.float32, device="cuda")
        packed_signs, norms = compute_signatures_gpu(residual, jl_matrix_gpu)
        assert packed_signs.is_cuda
        assert norms.is_cuda
        assert norms.shape == (1, 1, 4)

    def test_signs_are_packed(self, jl_matrix_gpu):
        residual = torch.randn(1, 1, 4, D, dtype=torch.float32, device="cuda")
        packed_signs, _ = compute_signatures_gpu(residual, jl_matrix_gpu)
        # 64 bits = 8 bytes = 8 uint8
        expected_packed_dim = (JL_DIM + 7) // 8
        assert packed_signs.shape[-1] == expected_packed_dim


class TestSignatureConsistency:
    """AC-3.2: 与 Phase 1 一致性"""

    def test_sign_agreement_rate(self, jl_matrix_gpu):
        """符号位一致率 ≥ 99%。"""
        import numpy as np
        from polarquant_kv.qjl import generate_jl_matrix, compute_signatures

        torch.manual_seed(42)
        residual_gpu = torch.randn(D, dtype=torch.float32, device="cuda")
        residual_np = residual_gpu.cpu().numpy()

        # GPU 版本
        packed, _ = compute_signatures_gpu(
            residual_gpu.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            jl_matrix_gpu,
        )
        signs_gpu = bit_unpack(packed, JL_DIM).squeeze().cpu().numpy().astype(bool)

        # CPU 版本（Phase 1）
        P_np = jl_matrix_gpu.cpu().numpy()
        sigs_cpu = compute_signatures(residual_np, P_np)
        signs_cpu = sigs_cpu.signs

        agreement = np.mean(signs_gpu == signs_cpu)
        assert agreement >= 0.99, f"符号位一致率 {agreement:.2%} < 99%"


class TestBitPacking:
    """AC-3.3: Bit packing 正确性"""

    def test_pack_unpack_roundtrip(self):
        signs = torch.tensor([True, False, True, True, False, False, True, False,
                              True, True, False, False, True, True, True, False],
                             dtype=torch.bool, device="cuda")
        packed = bit_pack(signs)
        unpacked = bit_unpack(packed, len(signs))
        assert torch.equal(signs, unpacked)

    def test_pack_unpack_64_bits(self):
        torch.manual_seed(42)
        signs = torch.randint(0, 2, (64,), dtype=torch.bool, device="cuda")
        packed = bit_pack(signs)
        assert packed.shape == (8,)  # 64 bits = 8 bytes
        unpacked = bit_unpack(packed, 64)
        assert torch.equal(signs, unpacked)

    def test_pack_unpack_random(self):
        for jl_dim in [32, 64, 128]:
            torch.manual_seed(jl_dim)
            signs = torch.randint(0, 2, (jl_dim,), dtype=torch.bool, device="cuda")
            packed = bit_pack(signs)
            unpacked = bit_unpack(packed, jl_dim)
            assert torch.equal(signs, unpacked), f"jl_dim={jl_dim} 失败"
