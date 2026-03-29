"""核心数据类型定义。"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CompressedKV:
    """PolarQuant 压缩后的 KV 向量。"""

    radius: np.ndarray  # (...), FP32, 向量范数
    quantized_direction: np.ndarray  # (..., d_padded), uint8, 量化方向向量
    group_mins: np.ndarray  # (..., num_groups), FP32, 每组最小值
    group_scales: np.ndarray  # (..., num_groups), FP32, 每组缩放因子
    n_bits: int
    group_size: int
    original_dim: int  # 原始维度（用于 unpadding）


@dataclass
class QJLSignatures:
    """QJL 误差修正的符号位数据。"""

    signs: np.ndarray  # (..., jl_dim), bool
    jl_dim: int
    residual_norm: float = 0.0  # 残差范数，用于缩放修正量


@dataclass
class CompressedKVCache:
    """压缩的 KV Cache（Key + Value + 可选 QJL 修正）。"""

    compressed_keys: CompressedKV
    compressed_values: CompressedKV
    key_signatures: QJLSignatures | None = None
