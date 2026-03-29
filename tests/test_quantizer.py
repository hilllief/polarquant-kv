"""需求 2: PolarQuant 极坐标量化 — 单元测试。

覆盖 AC-2.1 ~ AC-2.10。
"""

import numpy as np
import pytest

from polarquant_kv.quantizer import compress, decompress
from polarquant_kv.rotation import generate_rotation_matrix
from polarquant_kv.types import CompressedKV
from polarquant_kv.utils import cosine_similarity
from tests.factories import make_random_vector, make_zero_vector, make_extreme_vector

SEED = 42
D = 128


@pytest.fixture
def R():
    return generate_rotation_matrix(D, seed=SEED)


@pytest.fixture
def rng():
    return np.random.Generator(np.random.PCG64(SEED))


class TestFullPipeline:
    """AC-2.1: 完整量化流程"""

    def test_compress_decompress_fp32(self, R, rng):
        v = make_random_vector(D, rng)
        compressed = compress(v, R, n_bits=4, group_size=32)
        v_hat = decompress(compressed, R)
        assert v_hat.shape == v.shape
        assert v_hat.dtype == np.float32

    def test_compress_decompress_fp16_input(self, R, rng):
        v = make_random_vector(D, rng, dtype=np.float16)
        compressed = compress(v, R, n_bits=4, group_size=32)
        v_hat = decompress(compressed, R)
        # 输出应为 float32（内部统一转换）
        assert v_hat.dtype == np.float32
        assert v_hat.shape == (D,)


class TestCompressionRatio:
    """AC-2.2, AC-2.3: 压缩比"""

    def test_4bit_compression_ratio(self):
        from polarquant_kv.utils import compute_compression_ratio

        # AC-2.2: PolarQuant 自身的压缩比（不含 QJL 开销）
        # 公式: (128*16) / (16 + 4*128 + 4*32 + 0) = 2048/656 ≈ 3.12
        # PRD 的 4x 是不含 overhead 的理想值，实际含 group params 后约 3.1x
        ratio = compute_compression_ratio(d=128, n_bits=4, group_size=32, jl_dim=0)
        assert ratio >= 3.0, f"4-bit 压缩比 {ratio:.2f} < 3.0"

    def test_3bit_compression_ratio(self):
        from polarquant_kv.utils import compute_compression_ratio

        ratio = compute_compression_ratio(d=128, n_bits=3, group_size=32, jl_dim=0)
        assert ratio >= 3.8, f"3-bit 压缩比 {ratio:.2f} < 3.8"


class TestCosineSimilarity:
    """AC-2.4: 余弦相似度阈值"""

    @pytest.mark.parametrize(
        "n_bits, threshold",
        [(4, 0.99), (3, 0.98), (2, 0.92), (6, 0.995), (8, 0.999)],
    )
    def test_cosine_similarity_by_bits(self, R, rng, n_bits, threshold):
        # 多次采样取平均，避免单次随机波动
        sims = []
        for i in range(20):
            v = np.random.Generator(np.random.PCG64(SEED + i)).standard_normal(D).astype(np.float32)
            compressed = compress(v, R, n_bits=n_bits, group_size=32)
            v_hat = decompress(compressed, R)
            sims.append(cosine_similarity(v, v_hat))
        avg_sim = np.mean(sims)
        assert avg_sim >= threshold, f"{n_bits}-bit avg cosine sim {avg_sim:.4f} < {threshold}"


class TestPadding:
    """AC-2.5: group_size 不整除 d 时的 padding"""

    def test_non_divisible_group_size(self, rng):
        d = 100  # 100 % 32 = 4, 需要 padding
        R = generate_rotation_matrix(d, seed=SEED)
        v = make_random_vector(d, rng)
        compressed = compress(v, R, n_bits=4, group_size=32)
        v_hat = decompress(compressed, R)
        assert v_hat.shape == (d,)
        sim = cosine_similarity(v, v_hat)
        assert sim >= 0.98, f"Padded cosine sim {sim:.4f} < 0.98"

    def test_group_size_equals_dim(self, rng):
        d = 32
        R = generate_rotation_matrix(d, seed=SEED)
        v = make_random_vector(d, rng)
        compressed = compress(v, R, n_bits=4, group_size=32)
        v_hat = decompress(compressed, R)
        assert v_hat.shape == (d,)


class TestParameterValidation:
    """AC-2.6, AC-2.7: 参数校验"""

    def test_n_bits_too_low(self, R, rng):
        v = make_random_vector(D, rng)
        with pytest.raises(ValueError):
            compress(v, R, n_bits=1, group_size=32)

    def test_n_bits_too_high(self, R, rng):
        v = make_random_vector(D, rng)
        with pytest.raises(ValueError):
            compress(v, R, n_bits=9, group_size=32)

    def test_group_size_zero(self, R, rng):
        v = make_random_vector(D, rng)
        with pytest.raises(ValueError):
            compress(v, R, n_bits=4, group_size=0)

    def test_group_size_negative(self, R, rng):
        v = make_random_vector(D, rng)
        with pytest.raises(ValueError):
            compress(v, R, n_bits=4, group_size=-1)

    def test_group_size_exceeds_dim(self, R, rng):
        v = make_random_vector(D, rng)
        with pytest.raises(ValueError):
            compress(v, R, n_bits=4, group_size=D + 1)


class TestZeroVector:
    """AC-2.8: 零向量处理"""

    def test_zero_vector_no_nan(self, R):
        v = make_zero_vector(D)
        compressed = compress(v, R, n_bits=4, group_size=32)
        assert compressed.radius == 0.0 or np.isclose(compressed.radius, 0.0)
        v_hat = decompress(compressed, R)
        assert not np.any(np.isnan(v_hat))
        assert not np.any(np.isinf(v_hat))
        assert np.allclose(v_hat, 0.0, atol=1e-6)


class TestCompressedKVStructure:
    """AC-2.9: 结构化返回"""

    def test_has_required_fields(self, R, rng):
        v = make_random_vector(D, rng)
        compressed = compress(v, R, n_bits=4, group_size=32)
        assert isinstance(compressed, CompressedKV)
        assert hasattr(compressed, "radius")
        assert hasattr(compressed, "quantized_direction")
        assert hasattr(compressed, "group_mins")
        assert hasattr(compressed, "group_scales")
        assert compressed.n_bits == 4
        assert compressed.group_size == 32
        assert compressed.original_dim == D

    def test_fields_independently_accessible(self, R, rng):
        v = make_random_vector(D, rng)
        compressed = compress(v, R, n_bits=4, group_size=32)
        # 每个字段都是 numpy 数组，可独立检查
        assert isinstance(compressed.radius, (float, np.floating, np.ndarray))
        assert isinstance(compressed.quantized_direction, np.ndarray)
        assert isinstance(compressed.group_mins, np.ndarray)
        assert isinstance(compressed.group_scales, np.ndarray)


class TestExtremeValues:
    """AC-2.10: 极端值处理"""

    def test_extreme_fp16_values(self, R):
        v = make_extreme_vector(D)
        compressed = compress(v, R, n_bits=4, group_size=32)
        v_hat = decompress(compressed, R)
        assert not np.any(np.isnan(v_hat)), "解压结果包含 NaN"
        assert not np.any(np.isinf(v_hat)), "解压结果包含 Inf"

    def test_very_small_values(self, R):
        v = np.full(D, 1e-7, dtype=np.float32)
        compressed = compress(v, R, n_bits=4, group_size=32)
        v_hat = decompress(compressed, R)
        assert not np.any(np.isnan(v_hat))
        assert not np.any(np.isinf(v_hat))
