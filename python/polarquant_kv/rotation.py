"""需求 1: 正交旋转矩阵生成。"""

import numpy as np
from scipy.stats import ortho_group


def generate_rotation_matrix(d: int, seed: int | None = None) -> np.ndarray:
    """生成 d×d 随机正交矩阵。

    使用 scipy.stats.ortho_group 生成均匀分布的随机正交矩阵。

    Args:
        d: 矩阵维度，必须为正整数。
        seed: 随机种子，None 表示不固定。

    Returns:
        d×d 正交矩阵 (float64)。

    Raises:
        ValueError: d <= 0 或非整数。
    """
    if not isinstance(d, (int, np.integer)):
        raise ValueError(f"维度 d 必须为整数，收到 {type(d).__name__}")
    if d <= 0:
        raise ValueError(f"维度 d 必须为正整数，收到 {d}")

    if d == 1:
        # 1x1 正交矩阵只有 [1] 或 [-1]
        rng = np.random.Generator(np.random.PCG64(seed))
        sign = rng.choice([-1.0, 1.0])
        return np.array([[sign]], dtype=np.float64)

    rng = np.random.default_rng(seed)
    R = ortho_group.rvs(d, random_state=rng)
    return R.astype(np.float64)


def rotate(v: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """对向量（或 batch 向量）应用旋转。

    Args:
        v: 向量，shape (..., d)。
        rotation_matrix: d×d 正交矩阵。

    Returns:
        旋转后的向量，shape 与 v 相同。
    """
    return (v.astype(np.float64) @ rotation_matrix.T).astype(v.dtype)


def inverse_rotate(v: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """逆旋转（R^T · v）。

    Args:
        v: 旋转后的向量，shape (..., d)。
        rotation_matrix: d×d 正交矩阵。

    Returns:
        逆旋转后的向量，shape 与 v 相同。
    """
    return (v.astype(np.float64) @ rotation_matrix).astype(v.dtype)
