"""需求 2 & 6: PolarQuant 极坐标量化 + Batch 操作。"""

import logging
import math

import numpy as np

from polarquant_kv.rotation import rotate, inverse_rotate
from polarquant_kv.types import CompressedKV
from polarquant_kv.utils import check_memory_warning, estimate_memory_bytes

logger = logging.getLogger(__name__)


def _validate_params(n_bits: int, group_size: int, d: int) -> None:
    """校验量化参数。"""
    if not (2 <= n_bits <= 8):
        raise ValueError(f"n_bits 必须在 [2, 8] 范围内，收到 {n_bits}")
    if group_size <= 0:
        raise ValueError(f"group_size 必须为正整数，收到 {group_size}")
    if group_size > d:
        raise ValueError(f"group_size ({group_size}) 不能超过向量维度 d ({d})")


def _group_quantize(
    direction: np.ndarray, n_bits: int, group_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对方向向量进行分组量化。

    Returns:
        (quantized, group_mins, group_scales)
    """
    d = direction.shape[-1]
    num_groups = math.ceil(d / group_size)
    # Pad if needed
    padded_d = num_groups * group_size
    if padded_d > d:
        pad_width = padded_d - d
        direction = np.concatenate(
            [direction, np.zeros(pad_width, dtype=np.float32)]
        )

    groups = direction.reshape(num_groups, group_size)
    group_mins = groups.min(axis=1).astype(np.float32)
    group_maxs = groups.max(axis=1).astype(np.float32)

    levels = (1 << n_bits) - 1  # 2^n_bits - 1
    group_scales = np.where(
        group_maxs - group_mins > 0,
        (group_maxs - group_mins) / levels,
        np.float32(1.0),
    ).astype(np.float32)

    # Quantize each group
    quantized_groups = np.zeros_like(groups, dtype=np.uint8)
    for g in range(num_groups):
        if group_scales[g] > 0 and (group_maxs[g] - group_mins[g]) > 0:
            vals = np.clip(
                np.round((groups[g] - group_mins[g]) / group_scales[g]),
                0,
                levels,
            )
            quantized_groups[g] = vals.astype(np.uint8)

    quantized = quantized_groups.reshape(-1)
    return quantized, group_mins, group_scales


def _group_dequantize(
    quantized: np.ndarray,
    group_mins: np.ndarray,
    group_scales: np.ndarray,
    group_size: int,
    original_dim: int,
) -> np.ndarray:
    """反量化方向向量。"""
    num_groups = len(group_mins)
    padded_d = num_groups * group_size
    groups = quantized[:padded_d].reshape(num_groups, group_size).astype(np.float32)

    for g in range(num_groups):
        groups[g] = groups[g] * group_scales[g] + group_mins[g]

    direction = groups.reshape(-1)[:original_dim]
    return direction.astype(np.float32)


def compress(
    v: np.ndarray,
    rotation_matrix: np.ndarray,
    n_bits: int = 4,
    group_size: int = 32,
) -> CompressedKV:
    """PolarQuant 压缩：旋转 → 半径分离 → 分组量化。"""
    d = v.shape[-1]
    _validate_params(n_bits, group_size, d)

    # 统一转为 float32
    v_f32 = v.astype(np.float32)

    # Step 1: 随机旋转
    v_rotated = rotate(v_f32, rotation_matrix)

    # Step 2: 半径-方向分离
    radius = np.float32(np.linalg.norm(v_rotated))

    if radius < 1e-30:
        # 零向量特殊处理
        num_groups = math.ceil(d / group_size)
        padded_d = num_groups * group_size
        return CompressedKV(
            radius=np.float32(0.0),
            quantized_direction=np.zeros(padded_d, dtype=np.uint8),
            group_mins=np.zeros(num_groups, dtype=np.float32),
            group_scales=np.ones(num_groups, dtype=np.float32),
            n_bits=n_bits,
            group_size=group_size,
            original_dim=d,
        )

    direction = (v_rotated / radius).astype(np.float32)

    # Step 3: 分组量化
    quantized, group_mins, group_scales = _group_quantize(direction, n_bits, group_size)

    return CompressedKV(
        radius=radius,
        quantized_direction=quantized,
        group_mins=group_mins,
        group_scales=group_scales,
        n_bits=n_bits,
        group_size=group_size,
        original_dim=d,
    )


def decompress(
    compressed: CompressedKV,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """PolarQuant 解压：反量化 → 半径恢复 → 逆旋转。"""
    if compressed.radius < 1e-30:
        return np.zeros(compressed.original_dim, dtype=np.float32)

    direction = _group_dequantize(
        compressed.quantized_direction,
        compressed.group_mins,
        compressed.group_scales,
        compressed.group_size,
        compressed.original_dim,
    )

    # 恢复半径
    v_rotated = direction * compressed.radius

    # 逆旋转
    v_recovered = inverse_rotate(v_rotated, rotation_matrix)
    return v_recovered.astype(np.float32)


def compress_batch(
    kv: np.ndarray,
    rotation_matrix: np.ndarray,
    n_bits: int = 4,
    group_size: int = 32,
) -> list:
    """Batch 压缩 [batch, num_heads, seq_len, head_dim]。"""
    if kv.size == 0:
        # 保存 shape 信息用于 decompress_batch
        return {"empty": True, "shape": kv.shape}

    batch, num_heads, seq_len, head_dim = kv.shape

    # 内存预估警告 (AC-6.4)
    est_bytes = estimate_memory_bytes(
        batch, num_heads, seq_len, head_dim, n_bits, group_size, jl_dim=0
    )
    if check_memory_warning(est_bytes):
        logger.warning(
            f"预估内存占用 {est_bytes / 1e9:.2f} GB 超过可用内存的 80%%，请注意"
        )

    results = []
    for b in range(batch):
        batch_results = []
        for h in range(num_heads):
            head_results = []
            for s in range(seq_len):
                c = compress(kv[b, h, s], rotation_matrix, n_bits, group_size)
                head_results.append(c)
            batch_results.append(head_results)
        results.append(batch_results)
    return results


def decompress_batch(
    compressed_list: list | dict,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """Batch 解压，返回 [batch, num_heads, seq_len, head_dim]。"""
    if isinstance(compressed_list, dict) and compressed_list.get("empty"):
        return np.empty(compressed_list["shape"], dtype=np.float32)

    if not compressed_list:
        # 需要从上下文推断 shape，但空列表无法推断
        # 返回空数组，shape 由调用者处理
        return np.empty((0,), dtype=np.float32)

    batch = len(compressed_list)
    num_heads = len(compressed_list[0])
    seq_len = len(compressed_list[0][0]) if num_heads > 0 else 0
    head_dim = compressed_list[0][0][0].original_dim if seq_len > 0 else 0

    result = np.zeros((batch, num_heads, seq_len, head_dim), dtype=np.float32)
    for b in range(batch):
        for h in range(num_heads):
            for s in range(seq_len):
                result[b, h, s] = decompress(compressed_list[b][h][s], rotation_matrix)
    return result
