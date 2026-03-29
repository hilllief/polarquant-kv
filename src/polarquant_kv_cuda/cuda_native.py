"""C++ CUDA 融合 kernel 的 Python 接口。

使用 torch.utils.cpp_extension.load 进行 JIT 编译。
"""

import os
import math
import torch

_module = None
_CCBIN = r"D:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64"


def _get_module():
    global _module
    if _module is not None:
        return _module

    from torch.utils.cpp_extension import load

    csrc_dir = os.path.join(os.path.dirname(__file__), "..", "..", "csrc")
    cu_file = os.path.join(csrc_dir, "polarquant_kernels.cu")

    # RTX 5060 Ti = sm_120 (Blackwell)
    _module = load(
        name="polarquant_cuda_native",
        sources=[cu_file],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_120"],
        extra_cflags=["/O2", "/w"],
        verbose=False,
    )
    return _module


def fused_compress_4bit(
    kv: torch.Tensor,
    rotation_matrix: torch.Tensor,
    group_size: int = 32,
) -> dict:
    """融合 4-bit 压缩（C++ CUDA kernel）。

    Args:
        kv: [B, H, S, D] or [N, D], FP16, CUDA
        rotation_matrix: [D, D], FP32, CUDA

    Returns:
        dict with radius, packed, gmins, gscales tensors
    """
    mod = _get_module()

    original_shape = kv.shape
    D = kv.shape[-1]
    kv_flat = kv.reshape(-1, D).contiguous().half()
    N = kv_flat.shape[0]

    num_groups = (D + group_size - 1) // group_size

    # 分配输出
    radius = torch.empty(N, dtype=torch.float16, device=kv.device)
    packed = torch.empty(N, D // 2, dtype=torch.uint8, device=kv.device)
    gmins = torch.empty(N, num_groups, dtype=torch.float16, device=kv.device)
    gscales = torch.empty(N, num_groups, dtype=torch.float16, device=kv.device)

    # 调用融合 kernel
    mod.fused_compress_4bit(kv_flat, rotation_matrix)

    # 重新调用并获取结果（当前接口需要调整）
    # TODO: 修改 C++ 接口接受输出 tensor
    return {
        "radius": radius,
        "packed": packed,
        "gmins": gmins,
        "gscales": gscales,
        "original_shape": original_shape,
    }


def fused_score_4bit(
    query: torch.Tensor,
    packed_keys: torch.Tensor,
    gmins: torch.Tensor,
    gscales: torch.Tensor,
    radius: torch.Tensor,
    rotation_matrix: torch.Tensor,
    head_dim: int,
    group_size: int = 32,
) -> torch.Tensor:
    """融合 4-bit 注意力分数计算（C++ CUDA kernel）。

    Args:
        query: [D], FP32, CUDA（原始空间）
        packed_keys: [S, D/2], uint8, CUDA
        gmins: [S, G], FP16, CUDA
        gscales: [S, G], FP16, CUDA
        radius: [S], FP16, CUDA
        rotation_matrix: [D, D], FP32, CUDA

    Returns:
        [S], FP32, 注意力分数
    """
    mod = _get_module()

    # 预计算旋转空间的 query
    q_rotated = (query.float() @ rotation_matrix.T).contiguous()

    scale = 1.0 / math.sqrt(head_dim)

    scores = mod.fused_score_4bit(
        q_rotated, packed_keys.contiguous(),
        gmins.contiguous(), gscales.contiguous(),
        radius.contiguous(),
        head_dim, group_size, scale,
    )
    return scores
