"""PolarQuant 压缩 Kernel（含 bit packing 优化）。"""

import math
import torch

from polarquant_kv_cuda.types import CompressedKVCacheGPU
from polarquant_kv_cuda.rotation import rotate


def _validate_params(n_bits: int, group_size: int, d: int):
    if not (2 <= n_bits <= 8):
        raise ValueError(f"n_bits 必须在 [2, 8] 范围内，收到 {n_bits}")
    if group_size <= 0 or group_size > d:
        raise ValueError(f"group_size {group_size} 无效 (d={d})")


def _bit_pack_quantized(quantized: torch.Tensor, n_bits: int) -> torch.Tensor:
    """将量化值进行 bit packing。

    将多个 n_bits 值打包到 uint8 中。
    例如 4-bit: 2 个值/byte, 3-bit: 8 个值/3 bytes, 2-bit: 4 个值/byte。

    Args:
        quantized: [..., d_padded], uint8, 每个值 ∈ [0, 2^n_bits - 1]
        n_bits: 量化位数

    Returns:
        [..., packed_dim], uint8
    """
    if n_bits == 8:
        return quantized  # 无需打包

    *batch_shape, d = quantized.shape
    flat = quantized.reshape(-1, d).to(torch.int32)  # int32 支持位移

    if n_bits == 4:
        assert d % 2 == 0, f"4-bit packing 需要偶数维度，收到 {d}"
        low = flat[:, 0::2]
        high = flat[:, 1::2]
        packed = (high << 4) | low
        return packed.to(torch.uint8).reshape(*batch_shape, d // 2)

    elif n_bits == 2:
        assert d % 4 == 0, f"2-bit packing 需要 4 的倍数维度，收到 {d}"
        v0 = flat[:, 0::4]
        v1 = flat[:, 1::4]
        v2 = flat[:, 2::4]
        v3 = flat[:, 3::4]
        packed = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
        return packed.to(torch.uint8).reshape(*batch_shape, d // 4)

    elif n_bits == 3:
        # 8 个 3-bit 值打包到 3 bytes (24 bits)
        # 简化：用 uint8 存储，每个值占 1 byte（不做 3-bit packing，太复杂）
        # TODO: 实现真正的 3-bit packing
        return quantized

    elif n_bits == 6:
        # 4 个 6-bit 值打包到 3 bytes (24 bits)
        # 简化：用 uint8 存储
        return quantized

    return quantized


def _bit_unpack_quantized(packed: torch.Tensor, n_bits: int, original_dim: int) -> torch.Tensor:
    """Bit unpacking，恢复量化值。"""
    if n_bits == 8:
        return packed

    *batch_shape, packed_dim = packed.shape
    flat = packed.reshape(-1, packed_dim).to(torch.int32)

    if n_bits == 4:
        low = flat & 0x0F
        high = (flat >> 4) & 0x0F
        # 交错还原
        d = packed_dim * 2
        result = torch.zeros(flat.shape[0], d, dtype=torch.uint8, device=packed.device)
        result[:, 0::2] = low.to(torch.uint8)
        result[:, 1::2] = high.to(torch.uint8)
        return result.reshape(*batch_shape, d)

    elif n_bits == 2:
        v0 = flat & 0x03
        v1 = (flat >> 2) & 0x03
        v2 = (flat >> 4) & 0x03
        v3 = (flat >> 6) & 0x03
        d = packed_dim * 4
        result = torch.zeros(flat.shape[0], d, dtype=torch.uint8, device=packed.device)
        result[:, 0::4] = v0.to(torch.uint8)
        result[:, 1::4] = v1.to(torch.uint8)
        result[:, 2::4] = v2.to(torch.uint8)
        result[:, 3::4] = v3.to(torch.uint8)
        return result.reshape(*batch_shape, d)

    return packed


def compress_gpu(
    kv: torch.Tensor,
    rotation_matrix: torch.Tensor,
    n_bits: int = 4,
    group_size: int = 32,
    jl_matrix: torch.Tensor | None = None,
) -> CompressedKVCacheGPU:
    """GPU 上的 PolarQuant 压缩（含 bit packing）。"""
    if kv.numel() == 0:
        batch = kv.shape[0] if kv.dim() >= 1 else 0
        heads = kv.shape[1] if kv.dim() >= 2 else 0
        seq = kv.shape[2] if kv.dim() >= 3 else 0
        d = kv.shape[3] if kv.dim() >= 4 else 0
        num_groups = math.ceil(d / group_size) if d > 0 else 0
        d_padded = num_groups * group_size if d > 0 else 0
        packed_dim = _packed_dim(d_padded, n_bits)
        return CompressedKVCacheGPU(
            radius=torch.empty(batch, heads, seq, dtype=torch.float16, device=kv.device),
            quantized_direction=torch.empty(batch, heads, seq, packed_dim, dtype=torch.uint8, device=kv.device),
            group_mins=torch.empty(batch, heads, seq, num_groups, dtype=torch.float16, device=kv.device),
            group_scales=torch.empty(batch, heads, seq, num_groups, dtype=torch.float16, device=kv.device),
            qjl_signs=None, residual_norms=None,
            n_bits=n_bits, group_size=group_size, original_dim=d,
            seq_len=seq, max_seq_len=seq,
        )

    batch, num_heads, seq_len, head_dim = kv.shape
    _validate_params(n_bits, group_size, head_dim)
    device = kv.device

    num_groups = math.ceil(head_dim / group_size)
    d_padded = num_groups * group_size
    levels = (1 << n_bits) - 1

    # Step 1: 旋转
    kv_f32 = kv.float().clamp(-65000.0, 65000.0)
    kv_rotated = kv_f32 @ rotation_matrix.T

    # Step 2: 半径-方向分离
    radius = torch.norm(kv_rotated, dim=-1)
    safe_radius = radius.clamp(min=1e-30)
    direction = kv_rotated / safe_radius.unsqueeze(-1)

    zero_mask = (radius < 1e-30)
    if zero_mask.any():
        direction[zero_mask] = 0.0

    # Step 3: Padding
    if d_padded > head_dim:
        direction = torch.nn.functional.pad(direction, (0, d_padded - head_dim))

    # Step 4: 分组量化
    direction_grouped = direction.reshape(batch, num_heads, seq_len, num_groups, group_size)
    group_mins = direction_grouped.min(dim=-1).values
    group_maxs = direction_grouped.max(dim=-1).values
    group_range = group_maxs - group_mins
    group_scales = torch.where(group_range > 0, group_range / levels, torch.ones_like(group_range))

    normalized = (direction_grouped - group_mins.unsqueeze(-1)) / group_scales.unsqueeze(-1)
    quantized = normalized.clamp(0, levels).round().to(torch.uint8)
    quantized_flat = quantized.reshape(batch, num_heads, seq_len, d_padded)

    # Step 5: Bit packing
    packed = _bit_pack_quantized(quantized_flat, n_bits)

    # QJL
    qjl_signs = None
    residual_norms = None
    if jl_matrix is not None:
        dequant_grouped = quantized.float() * group_scales.unsqueeze(-1) + group_mins.unsqueeze(-1)
        dequant_flat = dequant_grouped.reshape(batch, num_heads, seq_len, d_padded)
        dequant_dir = dequant_flat[..., :head_dim]
        kv_hat_rotated = dequant_dir * safe_radius.unsqueeze(-1)
        residual = (kv_rotated - kv_hat_rotated) @ rotation_matrix
        from polarquant_kv_cuda.qjl_kernel import compute_signatures_gpu
        qjl_signs, residual_norms = compute_signatures_gpu(residual, jl_matrix)

    return CompressedKVCacheGPU(
        radius=radius.clamp(-65504.0, 65504.0).half(),
        quantized_direction=packed,
        group_mins=group_mins.half(),
        group_scales=group_scales.half(),
        qjl_signs=qjl_signs,
        residual_norms=residual_norms,
        n_bits=n_bits,
        group_size=group_size,
        original_dim=head_dim,
        seq_len=seq_len,
        max_seq_len=seq_len,
    )


def _packed_dim(d_padded: int, n_bits: int) -> int:
    if n_bits == 4:
        return d_padded // 2
    elif n_bits == 2:
        return d_padded // 4
    return d_padded
