"""共享 fixtures。"""

import numpy as np
import pytest

SEED = 42
HEAD_DIM = 128


@pytest.fixture
def rng():
    """可复现的随机数生成器。"""
    return np.random.Generator(np.random.PCG64(SEED))


@pytest.fixture
def head_dim():
    return HEAD_DIM


@pytest.fixture
def rotation_matrix_128():
    """128 维正交旋转矩阵（延迟导入，实现完成前测试会失败）。"""
    from polarquant_kv.rotation import generate_rotation_matrix

    return generate_rotation_matrix(HEAD_DIM, seed=SEED)


@pytest.fixture
def jl_matrix_64x128():
    """64×128 JL 投影矩阵。"""
    from polarquant_kv.qjl import generate_jl_matrix

    return generate_jl_matrix(jl_dim=64, d=HEAD_DIM, seed=SEED)


@pytest.fixture
def random_vectors(rng):
    """生成随机测试向量的工厂。"""

    def _make(shape, dtype=np.float32):
        return rng.standard_normal(shape).astype(dtype)

    return _make
