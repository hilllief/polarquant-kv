"""PolarQuant 配置。"""

from dataclasses import dataclass


@dataclass
class PolarQuantConfig:
    n_bits: int = 4
    group_size: int = 32
    jl_dim: int = 64
    enable_qjl: bool = False
    enable_compression: bool = True
