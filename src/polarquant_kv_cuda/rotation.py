"""旋转矩阵（PyTorch 实现）。"""

import torch
import numpy as np
from scipy.stats import ortho_group


def generate_rotation_matrix(d: int, seed: int | None = None, device="cuda") -> torch.Tensor:
    """生成 d×d 随机正交矩阵，放到指定设备上。"""
    if d <= 0:
        raise ValueError(f"维度 d 必须为正整数，收到 {d}")
    if d == 1:
        rng = np.random.Generator(np.random.PCG64(seed))
        sign = rng.choice([-1.0, 1.0])
        return torch.tensor([[sign]], dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed)
    R_np = ortho_group.rvs(d, random_state=rng).astype(np.float64)
    return torch.from_numpy(R_np).float().to(device)


def rotate(v: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """v @ R^T"""
    return (v.float() @ R.T).to(v.dtype)


def inverse_rotate(v: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """v @ R"""
    return (v.float() @ R).to(v.dtype)
