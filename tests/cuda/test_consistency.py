"""需求 6: 数值一致性验证 — GPU vs CPU。

覆盖 AC-6.1 ~ AC-6.3。
"""

import numpy as np
import torch
import pytest

from polarquant_kv_cuda.compress_kernel import compress_gpu
from polarquant_kv_cuda.decompress_kernel import decompress_gpu
from polarquant_kv_cuda.attention_kernel import compressed_attention_gpu
from polarquant_kv_cuda.rotation import generate_rotation_matrix

# Phase 1 参考实现
from polarquant_kv.rotation import generate_rotation_matrix as gen_R_cpu
from polarquant_kv.quantizer import compress as compress_cpu, decompress as decompress_cpu
from polarquant_kv.attention import standard_attention as std_attn_cpu

D = 128
SEED = 42


class TestCompressConsistency:
    """AC-6.1: 压缩一致性"""

    def test_gpu_cpu_compress_cosine(self):
        R_cpu = gen_R_cpu(D, seed=SEED)
        R_gpu = generate_rotation_matrix(D, seed=SEED, device="cuda")

        np.random.seed(SEED)
        v_np = np.random.randn(D).astype(np.float32)
        v_gpu = torch.from_numpy(v_np).half().cuda().reshape(1, 1, 1, D)

        # CPU 压缩 + 解压
        c_cpu = compress_cpu(v_np, R_cpu, n_bits=4, group_size=32)
        v_hat_cpu = decompress_cpu(c_cpu, R_cpu)

        # GPU 压缩 + 解压
        c_gpu = compress_gpu(v_gpu, R_gpu, n_bits=4, group_size=32)
        v_hat_gpu = decompress_gpu(c_gpu, R_gpu).squeeze().cpu().float().numpy()

        cos = np.dot(v_hat_cpu, v_hat_gpu) / (
            np.linalg.norm(v_hat_cpu) * np.linalg.norm(v_hat_gpu)
        )
        assert cos >= 0.999, f"GPU-CPU 压缩余弦相似度 {cos:.4f} < 0.999"

    def test_multiple_vectors(self):
        R_cpu = gen_R_cpu(D, seed=SEED)
        R_gpu = generate_rotation_matrix(D, seed=SEED, device="cuda")

        for i in range(20):
            np.random.seed(SEED + i)
            v_np = np.random.randn(D).astype(np.float32)
            v_gpu = torch.from_numpy(v_np).half().cuda().reshape(1, 1, 1, D)

            c_cpu = compress_cpu(v_np, R_cpu, n_bits=4, group_size=32)
            v_hat_cpu = decompress_cpu(c_cpu, R_cpu)

            c_gpu = compress_gpu(v_gpu, R_gpu, n_bits=4, group_size=32)
            v_hat_gpu = decompress_gpu(c_gpu, R_gpu).squeeze().cpu().float().numpy()

            cos = np.dot(v_hat_cpu, v_hat_gpu) / (
                np.linalg.norm(v_hat_cpu) * np.linalg.norm(v_hat_gpu) + 1e-30
            )
            assert cos >= 0.999, f"Vector {i}: GPU-CPU cos {cos:.4f} < 0.999"


class TestAttentionConsistency:
    """AC-6.2: 注意力一致性"""

    def test_gpu_cpu_attention_cosine(self):
        R_cpu = gen_R_cpu(D, seed=SEED)
        R_gpu = generate_rotation_matrix(D, seed=SEED, device="cuda")

        np.random.seed(SEED)
        Q_np = np.random.randn(1, D).astype(np.float32)
        K_np = np.random.randn(16, D).astype(np.float32)
        V_np = np.random.randn(16, D).astype(np.float32)

        # CPU 标准注意力
        out_cpu = std_attn_cpu(Q_np, K_np, V_np).flatten()

        # GPU 标准注意力
        Q_gpu = torch.from_numpy(Q_np).half().cuda().reshape(1, 1, 1, D)
        K_gpu = torch.from_numpy(K_np).half().cuda().reshape(1, 1, 16, D)
        V_gpu = torch.from_numpy(V_np).half().cuda().reshape(1, 1, 16, D)

        out_gpu_std = torch.nn.functional.scaled_dot_product_attention(
            Q_gpu.float(), K_gpu.float(), V_gpu.float()
        ).squeeze().cpu().float().numpy().flatten()

        cos = np.dot(out_cpu, out_gpu_std) / (
            np.linalg.norm(out_cpu) * np.linalg.norm(out_gpu_std)
        )
        assert cos >= 0.99, f"注意力 GPU-CPU cos {cos:.4f} < 0.99"
