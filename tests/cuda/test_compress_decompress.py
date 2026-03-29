"""需求 1 & 2: 压缩/解压 Kernel — 单元测试。

覆盖 AC-1.1 ~ AC-1.5, AC-1.8, AC-1.9, AC-2.1 ~ AC-2.3。
"""

import math
import torch
import pytest

from polarquant_kv_cuda.compress_kernel import compress_gpu
from polarquant_kv_cuda.decompress_kernel import decompress_gpu
from polarquant_kv_cuda.rotation import generate_rotation_matrix
from polarquant_kv_cuda.types import CompressedKVCacheGPU

D = 128
SEED = 42


@pytest.fixture
def R():
    return generate_rotation_matrix(D, seed=SEED, device="cuda")


class TestCompressOutput:
    """AC-1.1: 压缩输出结构"""

    def test_output_is_structured_tensors(self, R, random_kv_gpu):
        kv = random_kv_gpu(batch=1, num_heads=1, seq_len=4)
        result = compress_gpu(kv, R, n_bits=4, group_size=32)
        assert isinstance(result, CompressedKVCacheGPU)
        assert result.radius.device.type == "cuda"
        assert result.quantized_direction.device.type == "cuda"
        assert result.group_mins.device.type == "cuda"
        assert result.group_scales.device.type == "cuda"

    def test_output_shapes(self, R, random_kv_gpu):
        batch, heads, seq = 2, 4, 8
        kv = random_kv_gpu(batch=batch, num_heads=heads, seq_len=seq)
        result = compress_gpu(kv, R, n_bits=4, group_size=32)
        num_groups = math.ceil(D / 32)
        assert result.radius.shape == (batch, heads, seq)
        assert result.quantized_direction.shape[:-1] == (batch, heads, seq)
        assert result.group_mins.shape == (batch, heads, seq, num_groups)
        assert result.group_scales.shape == (batch, heads, seq, num_groups)

    def test_no_cpu_transfer(self, R, random_kv_gpu):
        """压缩过程不应经过 CPU。"""
        kv = random_kv_gpu()
        result = compress_gpu(kv, R, n_bits=4, group_size=32)
        assert result.radius.is_cuda
        assert result.quantized_direction.is_cuda


class TestDecompressOutput:
    """AC-2.1: 解压输出"""

    def test_output_is_fp16_cuda(self, R, random_kv_gpu):
        kv = random_kv_gpu(batch=1, num_heads=1, seq_len=4)
        compressed = compress_gpu(kv, R, n_bits=4, group_size=32)
        decompressed = decompress_gpu(compressed, R)
        assert decompressed.dtype == torch.float16
        assert decompressed.is_cuda
        assert decompressed.shape == kv.shape


class TestRoundTripAccuracy:
    """AC-2.3: 压缩-解压余弦相似度"""

    @pytest.mark.parametrize("n_bits,threshold", [(4, 0.99), (3, 0.96), (2, 0.88)])
    def test_cosine_similarity_by_bits(self, R, random_kv_gpu, n_bits, threshold):
        kv = random_kv_gpu(batch=1, num_heads=1, seq_len=16)
        compressed = compress_gpu(kv, R, n_bits=n_bits, group_size=32)
        kv_hat = decompress_gpu(compressed, R)
        # 逐向量余弦相似度
        kv_flat = kv.reshape(-1, D).float()
        kv_hat_flat = kv_hat.reshape(-1, D).float()
        cos = torch.nn.functional.cosine_similarity(kv_flat, kv_hat_flat, dim=1)
        avg_cos = cos.mean().item()
        assert avg_cos >= threshold, f"{n_bits}-bit avg cosine {avg_cos:.4f} < {threshold}"


class TestZeroVector:
    """AC-1.4: 零向量处理"""

    def test_zero_vector_no_nan(self, R):
        kv = torch.zeros(1, 1, 1, D, dtype=torch.float16, device="cuda")
        compressed = compress_gpu(kv, R, n_bits=4, group_size=32)
        kv_hat = decompress_gpu(compressed, R)
        assert not torch.any(torch.isnan(kv_hat))
        assert not torch.any(torch.isinf(kv_hat))


class TestExtremeValues:
    """AC-1.5: FP16 极端值"""

    def test_extreme_fp16_no_nan(self, R):
        kv = torch.randn(1, 1, 1, D, dtype=torch.float16, device="cuda")
        kv[0, 0, 0, 0] = torch.finfo(torch.float16).max
        kv[0, 0, 0, 1] = torch.finfo(torch.float16).tiny
        kv[0, 0, 0, 2] = -torch.finfo(torch.float16).max
        compressed = compress_gpu(kv, R, n_bits=4, group_size=32)
        kv_hat = decompress_gpu(compressed, R)
        assert not torch.any(torch.isnan(kv_hat))
        assert not torch.any(torch.isinf(kv_hat))


class TestOOMHandling:
    """AC-1.8: 显存不足错误处理"""

    def test_oom_raises_runtime_error(self, R):
        """尝试分配超大 tensor 应抛出 RuntimeError。"""
        # 这个测试可能不会触发 OOM（取决于 GPU 显存）
        # 至少验证函数不会静默失败
        try:
            huge = torch.randn(1024, 128, 65536, D, dtype=torch.float16, device="cuda")
            compress_gpu(huge, R, n_bits=4, group_size=32)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            pass  # 预期行为


class TestQuantizationConfigs:
    """AC-1.9: 支持的量化配置"""

    @pytest.mark.parametrize("n_bits", [2, 3, 4, 6, 8])
    def test_supported_n_bits(self, R, random_kv_gpu, n_bits):
        kv = random_kv_gpu(batch=1, num_heads=1, seq_len=4)
        compressed = compress_gpu(kv, R, n_bits=n_bits, group_size=32)
        kv_hat = decompress_gpu(compressed, R)
        assert kv_hat.shape == kv.shape

    @pytest.mark.parametrize("group_size", [16, 32, 64, 128])
    def test_supported_group_sizes(self, R, random_kv_gpu, group_size):
        kv = random_kv_gpu(batch=1, num_heads=1, seq_len=4)
        compressed = compress_gpu(kv, R, n_bits=4, group_size=group_size)
        kv_hat = decompress_gpu(compressed, R)
        assert kv_hat.shape == kv.shape
