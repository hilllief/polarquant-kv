"""需求 1 增量追加测试。

覆盖 AC-1.6。
"""

import torch
import pytest

from polarquant_kv_cuda.compress_kernel import compress_gpu
from polarquant_kv_cuda.decompress_kernel import decompress_gpu
from polarquant_kv_cuda.rotation import generate_rotation_matrix

D = 128
SEED = 42


class TestIncrementalAppend:
    """AC-1.6: 增量追加压缩"""

    def test_append_equals_full_compress(self):
        R = generate_rotation_matrix(D, seed=SEED, device="cuda")
        torch.manual_seed(SEED)
        seq_len = 8
        kv = torch.randn(1, 1, seq_len, D, dtype=torch.float16, device="cuda")

        # 全量压缩
        full = compress_gpu(kv, R, n_bits=4, group_size=32)
        full_hat = decompress_gpu(full, R)

        # 逐 token 压缩
        parts = []
        for t in range(seq_len):
            single = compress_gpu(kv[:, :, t:t+1, :], R, n_bits=4, group_size=32)
            parts.append(decompress_gpu(single, R))
        incr_hat = torch.cat(parts, dim=2)

        # 应该高度一致（允许 FP16 精度差异）
        assert torch.allclose(full_hat, incr_hat, atol=1e-3), "增量追加与全量压缩结果不一致"
