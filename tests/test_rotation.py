"""需求 1: 正交旋转矩阵生成 — 单元测试。

覆盖 AC-1.1 ~ AC-1.5。
"""

import numpy as np
import pytest

from polarquant_kv.rotation import generate_rotation_matrix, rotate, inverse_rotate


class TestOrthogonality:
    """AC-1.1: R^T · R = I"""

    def test_small_dim(self):
        R = generate_rotation_matrix(4, seed=42)
        identity = R.T @ R
        assert np.allclose(identity, np.eye(4), atol=1e-6)

    def test_typical_head_dim(self):
        R = generate_rotation_matrix(128, seed=42)
        identity = R.T @ R
        assert np.allclose(identity, np.eye(128), atol=1e-6)

    def test_shape(self):
        R = generate_rotation_matrix(64, seed=0)
        assert R.shape == (64, 64)

    def test_det_is_plus_or_minus_one(self):
        R = generate_rotation_matrix(32, seed=7)
        det = np.linalg.det(R)
        assert abs(abs(det) - 1.0) < 1e-5


class TestNormPreservation:
    """AC-1.2: ||R·v|| = ||v||"""

    def test_norm_preserved_random_vector(self):
        d = 128
        R = generate_rotation_matrix(d, seed=42)
        rng = np.random.Generator(np.random.PCG64(99))
        v = rng.standard_normal(d).astype(np.float32)
        v_rotated = rotate(v, R)
        rel_err = abs(np.linalg.norm(v_rotated) - np.linalg.norm(v)) / np.linalg.norm(v)
        assert rel_err < 1e-6

    def test_norm_preserved_unit_vector(self):
        d = 64
        R = generate_rotation_matrix(d, seed=10)
        v = np.zeros(d, dtype=np.float32)
        v[0] = 1.0
        v_rotated = rotate(v, R)
        assert abs(np.linalg.norm(v_rotated) - 1.0) < 1e-6

    def test_inverse_rotate_recovers_original(self):
        d = 128
        R = generate_rotation_matrix(d, seed=42)
        rng = np.random.Generator(np.random.PCG64(55))
        v = rng.standard_normal(d).astype(np.float32)
        v_recovered = inverse_rotate(rotate(v, R), R)
        assert np.allclose(v, v_recovered, atol=1e-5)


class TestReproducibility:
    """AC-1.3: 种子可复现性"""

    def test_same_seed_same_matrix(self):
        R1 = generate_rotation_matrix(128, seed=42)
        R2 = generate_rotation_matrix(128, seed=42)
        assert np.array_equal(R1, R2)

    def test_different_seed_different_matrix(self):
        R1 = generate_rotation_matrix(128, seed=42)
        R2 = generate_rotation_matrix(128, seed=43)
        assert not np.array_equal(R1, R2)

    def test_none_seed_is_random(self):
        R1 = generate_rotation_matrix(32, seed=None)
        R2 = generate_rotation_matrix(32, seed=None)
        # 极小概率相等，实际不会
        assert not np.array_equal(R1, R2)


class TestInvalidInput:
    """AC-1.4: 无效维度"""

    def test_zero_dim(self):
        with pytest.raises(ValueError):
            generate_rotation_matrix(0)

    def test_negative_dim(self):
        with pytest.raises(ValueError):
            generate_rotation_matrix(-5)

    def test_float_dim(self):
        with pytest.raises((ValueError, TypeError)):
            generate_rotation_matrix(3.5)  # type: ignore

    def test_dim_one(self):
        # d=1 是合法的边界情况，1x1 正交矩阵 = [1] 或 [-1]
        R = generate_rotation_matrix(1, seed=42)
        assert R.shape == (1, 1)
        assert abs(abs(R[0, 0]) - 1.0) < 1e-6


class TestHighDimension:
    """AC-1.5: 高维正交性"""

    def test_d_4096(self):
        R = generate_rotation_matrix(4096, seed=42)
        # 抽样检查：取前 10 行和前 10 列的子矩阵验证
        sub = R[:10, :]
        product = sub @ sub.T
        # 对角线应接近 1，非对角线接近 0
        assert np.allclose(np.diag(product), 1.0, atol=1e-5)
        off_diag = product - np.diag(np.diag(product))
        assert np.max(np.abs(off_diag)) < 1e-5

    def test_d_256(self):
        R = generate_rotation_matrix(256, seed=42)
        identity = R.T @ R
        assert np.allclose(identity, np.eye(256), atol=1e-5)
