"""融合注意力 Kernel（PyTorch GPU 实现）。"""

import math
import torch

from polarquant_kv_cuda.types import CompressedKVCacheGPU
from polarquant_kv_cuda.decompress_kernel import decompress_gpu
from polarquant_kv_cuda.qjl_kernel import compute_correction_gpu, _batch_bit_unpack


def compressed_attention_gpu(
    query: torch.Tensor,
    compressed_keys: CompressedKVCacheGPU,
    compressed_values: CompressedKVCacheGPU,
    rotation_matrix: torch.Tensor,
    jl_matrix: torch.Tensor | None = None,
    enable_qjl: bool = True,
    num_kv_heads: int | None = None,
) -> torch.Tensor:
    """压缩注意力计算。

    Args:
        query: [batch, num_q_heads, seq_q, head_dim], FP16, GPU
        compressed_keys: 压缩的 Key Cache
        compressed_values: 压缩的 Value Cache
        rotation_matrix: [head_dim, head_dim], FP32
        jl_matrix: [jl_dim, head_dim], FP32（可选）
        enable_qjl: 是否启用 QJL 修正
        num_kv_heads: KV heads 数量（GQA 时使用）

    Returns:
        [batch, num_q_heads, seq_q, head_dim], FP16
    """
    batch, num_q_heads, seq_q, d = query.shape
    seq_kv = compressed_keys.seq_len
    device = query.device

    if seq_kv == 0:
        return torch.zeros_like(query)

    # 解压 K 和 V
    # 尝试使用融合 kernel（K 不需要完整解压）
    # 融合 kernel 在单 head 小 batch 时有优势，多 head 时 Python 循环开销大
    # 当前 fallback 到分步实现，融合 kernel 作为可选优化
    use_fused = False  # 默认关闭，等 kernel 优化到 batch 化后再启用
    if use_fused:
        try:
            from polarquant_kv_cuda.fused_attention import fused_compressed_attention
            return fused_compressed_attention(
                query, compressed_keys, compressed_values,
                rotation_matrix, enable_qjl,
                num_kv_heads=compressed_keys.radius.shape[1] if n_kv_heads is None else n_kv_heads,
            )
        except Exception:
            pass

    # 融合注意力 V2：K 不完整解压，直接在旋转空间算点积
    # 节省 K 的解压显存和带宽
    try:
        from polarquant_kv_cuda.fused_attention_v3 import fused_attention_v3
        return fused_attention_v3(
            query, compressed_keys, compressed_values,
            rotation_matrix, enable_qjl,
            num_kv_heads=n_kv_heads,
        )
    except Exception:
        pass

    # Fallback: 分步解压 + 计算
    K_hat = decompress_gpu(compressed_keys, rotation_matrix)   # [B, H_kv, S_kv, D]
    V_hat = decompress_gpu(compressed_values, rotation_matrix)  # [B, H_kv, S_kv, D]

    # GQA 扩展
    n_kv_heads = K_hat.shape[1]
    if n_kv_heads < num_q_heads:
        repeat = num_q_heads // n_kv_heads
        K_hat = K_hat.repeat_interleave(repeat, dim=1)
        V_hat = V_hat.repeat_interleave(repeat, dim=1)
        # QJL signs 也需要扩展
        if compressed_keys.qjl_signs is not None:
            qjl_signs = compressed_keys.qjl_signs.repeat_interleave(repeat, dim=1)
            residual_norms = compressed_keys.residual_norms.repeat_interleave(repeat, dim=1)
        else:
            qjl_signs = None
            residual_norms = None
    else:
        qjl_signs = compressed_keys.qjl_signs
        residual_norms = compressed_keys.residual_norms

    # 注意力分数: [B, H_q, S_q, S_kv]
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(query.float(), K_hat.float().transpose(-2, -1)) * scale

    # QJL 修正
    if enable_qjl and jl_matrix is not None and qjl_signs is not None:
        # query: [B, H, Sq, D], signs: [B, H, Skv, packed]
        # 对每个 query token 和每个 key token 计算修正
        jl_dim = jl_matrix.shape[0]
        signs_unpacked = _batch_bit_unpack(qjl_signs, jl_dim)  # [B, H, Skv, jl_dim]
        sign_values = 2.0 * signs_unpacked.float() - 1.0

        # query 投影: [B, H, Sq, jl_dim]
        q_proj = query.float() @ jl_matrix.T

        # 修正: [B, H, Sq, Skv] = [B, H, Sq, jl_dim] @ [B, H, jl_dim, Skv]
        correction = torch.matmul(q_proj, sign_values.transpose(-2, -1)) / jl_dim
        scores = scores + correction * scale

    # Softmax + 加权求和
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, V_hat.float())

    return output.half()
