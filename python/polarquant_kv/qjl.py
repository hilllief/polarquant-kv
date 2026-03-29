"""需求 3: QJL 误差修正。"""

import numpy as np

from polarquant_kv.types import QJLSignatures


def generate_jl_matrix(
    jl_dim: int,
    d: int,
    seed: int | None = None,
) -> np.ndarray:
    """生成 JL 随机投影矩阵 P (jl_dim × d)。

    每个元素服从 N(0, 1/m)，其中 m = jl_dim。
    """
    if jl_dim <= 0:
        raise ValueError(f"jl_dim 必须为正整数，收到 {jl_dim}")

    rng = np.random.Generator(np.random.PCG64(seed))
    P = rng.standard_normal((jl_dim, d)).astype(np.float64) / np.sqrt(jl_dim)
    return P


def compute_signatures(
    residual: np.ndarray,
    jl_matrix: np.ndarray,
) -> QJLSignatures:
    """计算量化残差的 JL 投影符号位。"""
    jl_dim = jl_matrix.shape[0]
    residual_f64 = residual.astype(np.float64)
    projected = jl_matrix @ residual_f64
    signs = projected >= 0
    residual_norm = float(np.linalg.norm(residual_f64))
    return QJLSignatures(signs=signs, jl_dim=jl_dim, residual_norm=residual_norm)


def compute_correction(
    query: np.ndarray,
    signatures: QJLSignatures,
    jl_matrix: np.ndarray,
) -> float:
    """基于符号位计算注意力分数修正量（q·e 的估计）。

    使用简化的 1-bit JL 估计：correction = dot(sign(P·e), P·q) / m
    """
    q = query.astype(np.float64)
    m = signatures.jl_dim

    if signatures.residual_norm < 1e-30:
        return 0.0

    q_proj = jl_matrix @ q
    sign_values = 2.0 * signatures.signs.astype(np.float64) - 1.0
    correction = np.dot(sign_values, q_proj) / m

    return float(correction)
