"""压缩版 Flash Attention — 加载预编译 PTX kernel。"""

import os
import math
import cupy as cp
import torch
import numpy as np

from polarquant_kv_cuda.types import CompressedKVCacheGPU

# 加载预编译的 PTX
_PTX_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "csrc", "flash_compressed_attention.ptx")
_PTX_V2_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "csrc", "flash_compressed_v2.ptx")
_module = None
_module_v2 = None


def _get_kernel():
    global _module
    if _module is None:
        _module = cp.RawModule(path=_PTX_PATH)
    return _module.get_function("flash_compressed_attention_4bit")


def _get_kernel_v2():
    global _module_v2
    if _module_v2 is None:
        _module_v2 = cp.RawModule(path=_PTX_V2_PATH)
    return _module_v2.get_function("flash_compressed_v2")


def _to_cp(t: torch.Tensor) -> cp.ndarray:
    return cp.from_dlpack(t.contiguous())


def flash_compressed_attention(
    query: torch.Tensor,
    compressed_keys: CompressedKVCacheGPU,
    compressed_values: CompressedKVCacheGPU,
    rotation_matrix: torch.Tensor,
) -> torch.Tensor:
    """压缩版 Flash Attention。

    Args:
        query: [B, Hq, 1, D], FP16, CUDA
        compressed_keys/values: CompressedKVCacheGPU
        rotation_matrix: [D, D], FP32

    Returns:
        [B, Hq, 1, D], FP16
    """
    B, Hq, Sq, D = query.shape
    assert Sq == 1

    S = compressed_keys.seq_len
    Hkv = compressed_keys.radius.shape[1]
    gs = compressed_keys.group_size
    num_groups = (D + gs - 1) // gs
    scale = 1.0 / math.sqrt(D)

    if S == 0:
        return torch.zeros_like(query)

    kernel = _get_kernel()

    # 旋转空间的 query
    q_rot = (query.squeeze(2).float() @ rotation_matrix.T)
    q_flat = q_rot.reshape(B * Hq, D).contiguous()

    # Flatten compressed data
    total_heads = B * Hq
    total_kv = B * Hkv

    pk = compressed_keys.quantized_direction.reshape(total_kv, S, -1).contiguous()
    km = compressed_keys.group_mins.float().reshape(total_kv, S, num_groups).contiguous()
    ks = compressed_keys.group_scales.float().reshape(total_kv, S, num_groups).contiguous()
    kr = compressed_keys.radius.float().reshape(total_kv, S).contiguous()
    pv = compressed_values.quantized_direction.reshape(total_kv, S, -1).contiguous()
    vm = compressed_values.group_mins.float().reshape(total_kv, S, num_groups).contiguous()
    vs = compressed_values.group_scales.float().reshape(total_kv, S, num_groups).contiguous()
    vr = compressed_values.radius.float().reshape(total_kv, S).contiguous()

    out_cp = cp.zeros((total_heads, D), dtype=cp.float32)

    block_size = 128
    smem = (2 * D + block_size + 3) * 4  # q + acc + reduce + softmax_state

    kernel(
        (total_heads,), (block_size,),
        (_to_cp(q_flat), _to_cp(pk), _to_cp(km), _to_cp(ks), _to_cp(kr),
         _to_cp(pv), _to_cp(vm), _to_cp(vs), _to_cp(vr), out_cp,
         total_heads, total_kv, S, D, gs, num_groups, np.float32(scale)),
        shared_mem=smem,
    )

    outputs = torch.from_dlpack(out_cp).reshape(B, Hq, D)
    # 逆旋转
    outputs = outputs @ rotation_matrix
    return outputs.unsqueeze(2).half()


def flash_compressed_attention_v2(
    query: torch.Tensor,
    compressed_keys: CompressedKVCacheGPU,
    compressed_values: CompressedKVCacheGPU,
    rotation_matrix: torch.Tensor,
) -> torch.Tensor:
    """压缩版 Flash Attention V2（每 thread 一个维度）。"""
    B, Hq, Sq, D = query.shape
    assert Sq == 1

    S = compressed_keys.seq_len
    Hkv = compressed_keys.radius.shape[1]
    gs = compressed_keys.group_size
    num_groups = (D + gs - 1) // gs
    scale = 1.0 / math.sqrt(D)

    if S == 0:
        return torch.zeros_like(query)

    kernel = _get_kernel_v2()

    q_rot = (query.squeeze(2).float() @ rotation_matrix.T)
    q_flat = q_rot.reshape(B * Hq, D).contiguous()

    total_heads = B * Hq
    total_kv = B * Hkv

    pk = compressed_keys.quantized_direction.reshape(total_kv, S, -1).contiguous()
    km = compressed_keys.group_mins.float().reshape(total_kv, S, num_groups).contiguous()
    ks = compressed_keys.group_scales.float().reshape(total_kv, S, num_groups).contiguous()
    kr = compressed_keys.radius.float().reshape(total_kv, S).contiguous()
    pv = compressed_values.quantized_direction.reshape(total_kv, S, -1).contiguous()
    vm = compressed_values.group_mins.float().reshape(total_kv, S, num_groups).contiguous()
    vs = compressed_values.group_scales.float().reshape(total_kv, S, num_groups).contiguous()
    vr = compressed_values.radius.float().reshape(total_kv, S).contiguous()

    out_cp = cp.zeros((total_heads, D), dtype=cp.float32)

    block_size = D  # 128 threads, 每个 thread 一个维度
    smem = (block_size + 3) * 4  # reduce_buf + state

    kernel(
        (total_heads,), (block_size,),
        (_to_cp(q_flat), _to_cp(pk), _to_cp(km), _to_cp(ks), _to_cp(kr),
         _to_cp(pv), _to_cp(vm), _to_cp(vs), _to_cp(vr), out_cp,
         total_heads, total_kv, S, D, gs, num_groups, np.float32(scale)),
        shared_mem=smem,
    )

    outputs = torch.from_dlpack(out_cp).reshape(B, Hq, D)
    outputs = outputs @ rotation_matrix
    return outputs.unsqueeze(2).half()


_PTX_V3_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "csrc", "flash_compressed_v3.ptx")
_module_v3 = None


def _get_kernel_v3():
    global _module_v3
    if _module_v3 is None:
        _module_v3 = cp.RawModule(path=_PTX_V3_PATH)
    return _module_v3.get_function("flash_compressed_v3")


def flash_compressed_attention_v3(
    query: torch.Tensor,
    compressed_keys: CompressedKVCacheGPU,
    compressed_values: CompressedKVCacheGPU,
    rotation_matrix: torch.Tensor,
) -> torch.Tensor:
    """压缩版 Flash Attention V3（warp shuffle 优化）。"""
    B, Hq, Sq, D = query.shape
    assert Sq == 1

    S = compressed_keys.seq_len
    Hkv = compressed_keys.radius.shape[1]
    gs = compressed_keys.group_size
    num_groups = (D + gs - 1) // gs
    scale = 1.0 / math.sqrt(D)

    if S == 0:
        return torch.zeros_like(query)

    kernel = _get_kernel_v3()

    q_rot = (query.squeeze(2).float() @ rotation_matrix.T)
    q_flat = q_rot.reshape(B * Hq, D).contiguous()

    total_heads = B * Hq
    total_kv = B * Hkv

    pk = compressed_keys.quantized_direction.reshape(total_kv, S, -1).contiguous()
    km = compressed_keys.group_mins.float().reshape(total_kv, S, num_groups).contiguous()
    ks = compressed_keys.group_scales.float().reshape(total_kv, S, num_groups).contiguous()
    kr = compressed_keys.radius.float().reshape(total_kv, S).contiguous()
    pv = compressed_values.quantized_direction.reshape(total_kv, S, -1).contiguous()
    vm = compressed_values.group_mins.float().reshape(total_kv, S, num_groups).contiguous()
    vs = compressed_values.group_scales.float().reshape(total_kv, S, num_groups).contiguous()
    vr = compressed_values.radius.float().reshape(total_kv, S).contiguous()

    out_cp = cp.zeros((total_heads, D), dtype=cp.float32)

    block_size = D  # 128 threads
    smem = (4 + 4) * 4  # sh_warp_sums[4] + sh_max/sum/score/corr

    kernel(
        (total_heads,), (block_size,),
        (_to_cp(q_flat), _to_cp(pk), _to_cp(km), _to_cp(ks), _to_cp(kr),
         _to_cp(pv), _to_cp(vm), _to_cp(vs), _to_cp(vr), out_cp,
         total_heads, total_kv, S, D, gs, num_groups, np.float32(scale)),
        shared_mem=smem,
    )

    outputs = torch.from_dlpack(out_cp).reshape(B, Hq, D)
    outputs = outputs @ rotation_matrix
    return outputs.unsqueeze(2).half()


_PTX_V4_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "csrc", "flash_compressed_v4.ptx")
_module_v4 = None


def _get_kernel_v4():
    global _module_v4
    if _module_v4 is None:
        _module_v4 = cp.RawModule(path=_PTX_V4_PATH)
    return _module_v4.get_function("flash_compressed_v4")


def flash_compressed_attention_v4(
    query: torch.Tensor,
    compressed_keys: CompressedKVCacheGPU,
    compressed_values: CompressedKVCacheGPU,
    rotation_matrix: torch.Tensor,
) -> torch.Tensor:
    """压缩版 Flash Attention V4（shared memory tiling）。"""
    B, Hq, Sq, D = query.shape
    assert Sq == 1

    S = compressed_keys.seq_len
    Hkv = compressed_keys.radius.shape[1]
    gs = compressed_keys.group_size
    num_groups = (D + gs - 1) // gs
    scale = 1.0 / math.sqrt(D)
    packed_dim = D // 2
    BLOCK_K = 32

    if S == 0:
        return torch.zeros_like(query)

    kernel = _get_kernel_v4()

    q_rot = (query.squeeze(2).float() @ rotation_matrix.T)
    q_flat = q_rot.reshape(B * Hq, D).contiguous()

    total_heads = B * Hq
    total_kv = B * Hkv

    pk = compressed_keys.quantized_direction.reshape(total_kv, S, -1).contiguous()
    km = compressed_keys.group_mins.float().reshape(total_kv, S, num_groups).contiguous()
    ks = compressed_keys.group_scales.float().reshape(total_kv, S, num_groups).contiguous()
    kr = compressed_keys.radius.float().reshape(total_kv, S).contiguous()
    pv = compressed_values.quantized_direction.reshape(total_kv, S, -1).contiguous()
    vm = compressed_values.group_mins.float().reshape(total_kv, S, num_groups).contiguous()
    vs = compressed_values.group_scales.float().reshape(total_kv, S, num_groups).contiguous()
    vr = compressed_values.radius.float().reshape(total_kv, S).contiguous()

    out_cp = cp.zeros((total_heads, D), dtype=cp.float32)

    block_size = D  # 128
    # smem: 8 floats + 4*BLOCK_K*G floats + 2*BLOCK_K floats + 2*BLOCK_K*packed_dim bytes
    float_smem = 8 + 4 * BLOCK_K * num_groups + 2 * BLOCK_K
    byte_smem = 2 * BLOCK_K * packed_dim
    smem = float_smem * 4 + byte_smem

    kernel(
        (total_heads,), (block_size,),
        (_to_cp(q_flat), _to_cp(pk), _to_cp(km), _to_cp(ks), _to_cp(kr),
         _to_cp(pv), _to_cp(vm), _to_cp(vs), _to_cp(vr), out_cp,
         total_heads, total_kv, S, D, gs, num_groups, np.float32(scale)),
        shared_mem=smem,
    )

    outputs = torch.from_dlpack(out_cp).reshape(B, Hq, D)
    outputs = outputs @ rotation_matrix
    return outputs.unsqueeze(2).half()


_PTX_V5_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "csrc", "flash_compressed_v5.ptx")
_module_v5 = None

def _get_kernel_v5():
    global _module_v5
    if _module_v5 is None:
        _module_v5 = cp.RawModule(path=_PTX_V5_PATH)
    return _module_v5.get_function("flash_compressed_v5")

def flash_compressed_attention_v5(
    query: torch.Tensor,
    compressed_keys: CompressedKVCacheGPU,
    compressed_values: CompressedKVCacheGPU,
    rotation_matrix: torch.Tensor,
) -> torch.Tensor:
    """V5: 2-pass（score → softmax → V加权），score 存全局内存。"""
    B, Hq, Sq, D = query.shape
    assert Sq == 1
    S = compressed_keys.seq_len
    Hkv = compressed_keys.radius.shape[1]
    gs = compressed_keys.group_size
    num_groups = (D + gs - 1) // gs
    scale = 1.0 / math.sqrt(D)
    if S == 0:
        return torch.zeros_like(query)

    kernel = _get_kernel_v5()
    q_rot = (query.squeeze(2).float() @ rotation_matrix.T)
    q_flat = q_rot.reshape(B * Hq, D).contiguous()
    total_heads = B * Hq
    total_kv = B * Hkv

    pk = compressed_keys.quantized_direction.reshape(total_kv, S, -1).contiguous()
    km = compressed_keys.group_mins.float().reshape(total_kv, S, num_groups).contiguous()
    ks = compressed_keys.group_scales.float().reshape(total_kv, S, num_groups).contiguous()
    kr = compressed_keys.radius.float().reshape(total_kv, S).contiguous()
    pv = compressed_values.quantized_direction.reshape(total_kv, S, -1).contiguous()
    vm = compressed_values.group_mins.float().reshape(total_kv, S, num_groups).contiguous()
    vs = compressed_values.group_scales.float().reshape(total_kv, S, num_groups).contiguous()
    vr = compressed_values.radius.float().reshape(total_kv, S).contiguous()

    out_cp = cp.zeros((total_heads, D), dtype=cp.float32)
    score_buf = cp.zeros((total_heads, S), dtype=cp.float32)

    block_size = D
    smem = (4 + 2) * 4  # warp_sums + max + sum

    kernel(
        (total_heads,), (block_size,),
        (_to_cp(q_flat), _to_cp(pk), _to_cp(km), _to_cp(ks), _to_cp(kr),
         _to_cp(pv), _to_cp(vm), _to_cp(vs), _to_cp(vr), out_cp, score_buf,
         total_heads, total_kv, S, D, gs, num_groups, np.float32(scale)),
        shared_mem=smem,
    )

    outputs = torch.from_dlpack(out_cp).reshape(B, Hq, D)
    outputs = outputs @ rotation_matrix
    return outputs.unsqueeze(2).half()


def flash_compressed_attention_v6(
    query: torch.Tensor,
    compressed_keys: CompressedKVCacheGPU,
    compressed_values: CompressedKVCacheGPU,
    rotation_matrix: torch.Tensor,
    # 预计算的 CuPy 数组（避免每次转换）
    _precomputed: dict | None = None,
) -> torch.Tensor:
    """V6: 消除 Python overhead + FP16 计算。

    优化:
    1. 预计算 DLPack 转换（调用者缓存）
    2. 预分配 score_buf 和 output（不每次 alloc）
    3. q @ R^T 用 FP16（省一半带宽）
    """
    B, Hq, Sq, D = query.shape
    assert Sq == 1
    S = compressed_keys.seq_len
    Hkv = compressed_keys.radius.shape[1]
    gs = compressed_keys.group_size
    num_groups = (D + gs - 1) // gs
    scale = 1.0 / math.sqrt(D)
    if S == 0:
        return torch.zeros_like(query)

    kernel = _get_kernel_v5()
    total_heads = B * Hq
    total_kv = B * Hkv

    if _precomputed is not None:
        # 使用预计算的 CuPy 数组
        pk_cp = _precomputed["pk"]
        km_cp = _precomputed["km"]
        ks_cp = _precomputed["ks"]
        kr_cp = _precomputed["kr"]
        pv_cp = _precomputed["pv"]
        vm_cp = _precomputed["vm"]
        vs_cp = _precomputed["vs"]
        vr_cp = _precomputed["vr"]
        out_cp = _precomputed["out"]
        score_cp = _precomputed["score"]
    else:
        pk_cp = _to_cp(compressed_keys.quantized_direction.reshape(total_kv, S, -1).contiguous())
        km_cp = _to_cp(compressed_keys.group_mins.float().reshape(total_kv, S, num_groups).contiguous())
        ks_cp = _to_cp(compressed_keys.group_scales.float().reshape(total_kv, S, num_groups).contiguous())
        kr_cp = _to_cp(compressed_keys.radius.float().reshape(total_kv, S).contiguous())
        pv_cp = _to_cp(compressed_values.quantized_direction.reshape(total_kv, S, -1).contiguous())
        vm_cp = _to_cp(compressed_values.group_mins.float().reshape(total_kv, S, num_groups).contiguous())
        vs_cp = _to_cp(compressed_values.group_scales.float().reshape(total_kv, S, num_groups).contiguous())
        vr_cp = _to_cp(compressed_values.radius.float().reshape(total_kv, S).contiguous())
        out_cp = cp.zeros((total_heads, D), dtype=cp.float32)
        score_cp = cp.zeros((total_heads, S), dtype=cp.float32)

    # Q 旋转（这个每次都要算）
    q_rot = (query.squeeze(2).float() @ rotation_matrix.T).reshape(total_heads, D).contiguous()
    q_cp = _to_cp(q_rot)

    block_size = D
    smem = (4 + 2) * 4

    kernel(
        (total_heads,), (block_size,),
        (q_cp, pk_cp, km_cp, ks_cp, kr_cp,
         pv_cp, vm_cp, vs_cp, vr_cp, out_cp, score_cp,
         total_heads, total_kv, S, D, gs, num_groups, np.float32(scale)),
        shared_mem=smem,
    )

    outputs = torch.from_dlpack(out_cp).reshape(B, Hq, D)
    outputs = outputs @ rotation_matrix
    return outputs.unsqueeze(2).half()


def precompute_attention_data(
    compressed_keys: CompressedKVCacheGPU,
    compressed_values: CompressedKVCacheGPU,
    B: int, Hq: int, D: int,
) -> dict:
    """预计算 CuPy 数组，避免每次注意力调用时重复转换。"""
    S = compressed_keys.seq_len
    Hkv = compressed_keys.radius.shape[1]
    num_groups = (D + compressed_keys.group_size - 1) // compressed_keys.group_size
    total_heads = B * Hq
    total_kv = B * Hkv

    return {
        "pk": _to_cp(compressed_keys.quantized_direction.reshape(total_kv, S, -1).contiguous()),
        "km": _to_cp(compressed_keys.group_mins.float().reshape(total_kv, S, num_groups).contiguous()),
        "ks": _to_cp(compressed_keys.group_scales.float().reshape(total_kv, S, num_groups).contiguous()),
        "kr": _to_cp(compressed_keys.radius.float().reshape(total_kv, S).contiguous()),
        "pv": _to_cp(compressed_values.quantized_direction.reshape(total_kv, S, -1).contiguous()),
        "vm": _to_cp(compressed_values.group_mins.float().reshape(total_kv, S, num_groups).contiguous()),
        "vs": _to_cp(compressed_values.group_scales.float().reshape(total_kv, S, num_groups).contiguous()),
        "vr": _to_cp(compressed_values.radius.float().reshape(total_kv, S).contiguous()),
        "out": cp.zeros((total_heads, D), dtype=cp.float32),
        "score": cp.zeros((total_heads, S), dtype=cp.float32),
    }


_PTX_V6_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "csrc", "flash_compressed_v6.ptx")
_module_v6 = None

def _get_kernel_v6():
    global _module_v6
    if _module_v6 is None:
        _module_v6 = cp.RawModule(path=_PTX_V6_PATH)
    return _module_v6.get_function("flash_compressed_v6")

def flash_compressed_attention_v6_kernel(
    query, compressed_keys, compressed_values, rotation_matrix,
    _precomputed=None,
):
    """V6 kernel: 无分支解压 + __expf + 预计算。"""
    B, Hq, Sq, D = query.shape
    assert Sq == 1
    S = compressed_keys.seq_len
    Hkv = compressed_keys.radius.shape[1]
    gs = compressed_keys.group_size
    num_groups = (D + gs - 1) // gs
    scale = 1.0 / math.sqrt(D)
    if S == 0:
        return torch.zeros_like(query)

    kernel = _get_kernel_v6()
    total_heads = B * Hq
    total_kv = B * Hkv

    if _precomputed:
        pk_cp, km_cp, ks_cp, kr_cp = _precomputed["pk"], _precomputed["km"], _precomputed["ks"], _precomputed["kr"]
        pv_cp, vm_cp, vs_cp, vr_cp = _precomputed["pv"], _precomputed["vm"], _precomputed["vs"], _precomputed["vr"]
        out_cp, score_cp = _precomputed["out"], _precomputed["score"]
    else:
        pk_cp = _to_cp(compressed_keys.quantized_direction.reshape(total_kv, S, -1).contiguous())
        km_cp = _to_cp(compressed_keys.group_mins.float().reshape(total_kv, S, num_groups).contiguous())
        ks_cp = _to_cp(compressed_keys.group_scales.float().reshape(total_kv, S, num_groups).contiguous())
        kr_cp = _to_cp(compressed_keys.radius.float().reshape(total_kv, S).contiguous())
        pv_cp = _to_cp(compressed_values.quantized_direction.reshape(total_kv, S, -1).contiguous())
        vm_cp = _to_cp(compressed_values.group_mins.float().reshape(total_kv, S, num_groups).contiguous())
        vs_cp = _to_cp(compressed_values.group_scales.float().reshape(total_kv, S, num_groups).contiguous())
        vr_cp = _to_cp(compressed_values.radius.float().reshape(total_kv, S).contiguous())
        out_cp = cp.zeros((total_heads, D), dtype=cp.float32)
        score_cp = cp.zeros((total_heads, S), dtype=cp.float32)

    q_rot = (query.squeeze(2).float() @ rotation_matrix.T).reshape(total_heads, D).contiguous()
    q_cp = _to_cp(q_rot)

    kernel(
        (total_heads,), (D,),
        (q_cp, pk_cp, km_cp, ks_cp, kr_cp,
         pv_cp, vm_cp, vs_cp, vr_cp, out_cp, score_cp,
         total_heads, total_kv, S, D, gs, num_groups, np.float32(scale)),
        shared_mem=(4 + 1) * 4,
    )

    outputs = torch.from_dlpack(out_cp).reshape(B, Hq, D)
    outputs = outputs @ rotation_matrix
    return outputs.unsqueeze(2).half()
