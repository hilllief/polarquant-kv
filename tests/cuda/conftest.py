"""GPU 测试共享 fixtures。"""

import pytest
import torch

SEED = 42
HEAD_DIM = 128
DEVICE = "cuda"


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture(autouse=True)
def require_cuda():
    _skip_if_no_cuda()


@pytest.fixture
def device():
    return DEVICE


@pytest.fixture
def head_dim():
    return HEAD_DIM


@pytest.fixture
def R_gpu():
    """128 维正交旋转矩阵 (GPU)。"""
    from polarquant_kv_cuda.rotation import generate_rotation_matrix
    return generate_rotation_matrix(HEAD_DIM, seed=SEED, device=DEVICE)


@pytest.fixture
def jl_matrix_gpu():
    """64×128 JL 投影矩阵 (GPU)。"""
    torch.manual_seed(SEED + 1)
    P = torch.randn(64, HEAD_DIM, dtype=torch.float32, device=DEVICE) / (64 ** 0.5)
    return P


@pytest.fixture
def random_kv_gpu():
    """生成随机 KV tensor 的工厂。"""
    def _make(batch=1, num_heads=1, seq_len=16, head_dim=HEAD_DIM):
        torch.manual_seed(SEED)
        return torch.randn(batch, num_heads, seq_len, head_dim,
                           dtype=torch.float16, device=DEVICE)
    return _make
