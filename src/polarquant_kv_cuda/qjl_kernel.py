"""QJL 投影 Kernel（PyTorch GPU 实现）。"""

import torch


def compute_signatures_gpu(
    residual: torch.Tensor,
    jl_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算量化残差的 JL 投影符号位。

    Args:
        residual: [..., d], FP32, GPU
        jl_matrix: [jl_dim, d], FP32, GPU

    Returns:
        (packed_signs, residual_norms)
        packed_signs: [..., jl_dim_packed], uint8
        residual_norms: [...], FP16
    """
    original_shape = residual.shape[:-1]
    d = residual.shape[-1]
    jl_dim = jl_matrix.shape[0]

    # 投影: [..., d] @ [d, jl_dim] = [..., jl_dim]
    projected = residual.float() @ jl_matrix.T

    # 符号位
    signs = projected >= 0  # [..., jl_dim], bool

    # Bit packing
    packed = _batch_bit_pack(signs)

    # 残差范数
    norms = torch.norm(residual.float(), dim=-1).half()

    return packed, norms


def compute_correction_gpu(
    query: torch.Tensor,
    packed_signs: torch.Tensor,
    residual_norms: torch.Tensor,
    jl_matrix: torch.Tensor,
) -> torch.Tensor:
    """基于符号位计算注意力分数修正量。

    Args:
        query: [..., d], FP16/FP32
        packed_signs: [..., jl_dim_packed], uint8
        residual_norms: [...], FP16
        jl_matrix: [jl_dim, d], FP32

    Returns:
        修正量 tensor, shape [...]
    """
    jl_dim = jl_matrix.shape[0]

    # 解包符号位
    signs = _batch_bit_unpack(packed_signs, jl_dim)  # [..., jl_dim], bool
    sign_values = 2.0 * signs.float() - 1.0  # +1/-1

    # query 投影
    q_proj = query.float() @ jl_matrix.T  # [..., jl_dim]

    # 修正量 = dot(sign, q_proj) / m
    correction = (sign_values * q_proj).sum(dim=-1) / jl_dim

    return correction


def bit_pack(signs: torch.Tensor) -> torch.Tensor:
    """将 bool tensor 打包为 uint8。

    Args:
        signs: [N], bool, GPU

    Returns:
        [ceil(N/8)], uint8
    """
    n = signs.shape[0]
    # Pad to multiple of 8
    padded_n = ((n + 7) // 8) * 8
    if padded_n > n:
        signs = torch.nn.functional.pad(signs, (0, padded_n - n))

    signs_uint8 = signs.to(torch.uint8).reshape(-1, 8)
    # Pack: bit 0 is LSB
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                           dtype=torch.uint8, device=signs.device)
    packed = (signs_uint8 * weights).sum(dim=1).to(torch.uint8)
    return packed


def bit_unpack(packed: torch.Tensor, n: int) -> torch.Tensor:
    """将 uint8 解包为 bool tensor。

    Args:
        packed: [ceil(N/8)], uint8
        n: 原始长度

    Returns:
        [n], bool
    """
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                           dtype=torch.uint8, device=packed.device)
    # Unpack each byte to 8 bits
    unpacked = ((packed.unsqueeze(-1) & weights) > 0)  # [M, 8], bool
    return unpacked.reshape(-1)[:n]


def _batch_bit_pack(signs: torch.Tensor) -> torch.Tensor:
    """Batch bit packing: [..., jl_dim] bool -> [..., ceil(jl_dim/8)] uint8."""
    *batch_shape, jl_dim = signs.shape
    padded_jl = ((jl_dim + 7) // 8) * 8
    if padded_jl > jl_dim:
        signs = torch.nn.functional.pad(signs, (0, padded_jl - jl_dim))

    flat = signs.reshape(-1, padded_jl)
    grouped = flat.reshape(flat.shape[0], -1, 8).to(torch.uint8)
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                           dtype=torch.uint8, device=signs.device)
    packed = (grouped * weights).sum(dim=-1).to(torch.uint8)
    return packed.reshape(*batch_shape, -1)


def _batch_bit_unpack(packed: torch.Tensor, jl_dim: int) -> torch.Tensor:
    """Batch bit unpacking: [..., ceil(jl_dim/8)] uint8 -> [..., jl_dim] bool."""
    *batch_shape, packed_dim = packed.shape
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                           dtype=torch.uint8, device=packed.device)
    flat = packed.reshape(-1, packed_dim)
    unpacked = ((flat.unsqueeze(-1) & weights) > 0)  # [N, packed_dim, 8]
    unpacked = unpacked.reshape(flat.shape[0], -1)[:, :jl_dim]
    return unpacked.reshape(*batch_shape, jl_dim)
