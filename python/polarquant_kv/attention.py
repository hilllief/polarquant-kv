"""需求 4: 压缩注意力计算。"""

import numpy as np

from polarquant_kv.quantizer import compress, decompress
from polarquant_kv.qjl import compute_signatures, compute_correction
from polarquant_kv.utils import cosine_similarity


def _expand_kv_for_gqa(
    kv: np.ndarray, num_q_heads: int, num_kv_heads: int
) -> np.ndarray:
    """GQA: 将 KV heads 扩展到与 Q heads 数量一致。"""
    if num_kv_heads == num_q_heads:
        return kv
    repeat = num_q_heads // num_kv_heads
    return np.repeat(kv, repeat, axis=0)


def standard_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    num_kv_heads: int | None = None,
) -> np.ndarray:
    """标准注意力计算（参考实现）。

    支持的 shape:
    - 2D: query (n_q, d), key (seq, d), value (seq, d)
    - 3D (GQA): query (n_heads, seq_q, d), key (n_kv_heads, seq, d), value (n_kv_heads, seq, d)
    """
    if query.ndim == 2 and key.ndim == 2:
        return _standard_attention_2d(query, key, value)
    elif query.ndim == 3 and key.ndim == 3:
        return _standard_attention_gqa(query, key, value, num_kv_heads)
    else:
        raise ValueError(f"不支持的 shape: query {query.shape}, key {key.shape}")


def _standard_attention_2d(
    query: np.ndarray, key: np.ndarray, value: np.ndarray
) -> np.ndarray:
    """2D 标准注意力: query (n_q, d), key (seq, d), value (seq, d)。"""
    seq_len = key.shape[0]
    d = query.shape[-1]

    if seq_len == 0:
        return np.zeros((query.shape[0], d), dtype=np.float32)

    scores = query @ key.T / np.sqrt(d)  # (n_q, seq)
    weights = _softmax(scores)
    return (weights @ value).astype(np.float32)


def _standard_attention_gqa(
    query: np.ndarray, key: np.ndarray, value: np.ndarray,
    num_kv_heads: int | None,
) -> np.ndarray:
    """3D GQA 注意力: query (n_q_heads, seq_q, d)。"""
    n_q_heads = query.shape[0]
    n_kv_heads = key.shape[0]
    d = query.shape[-1]

    if num_kv_heads is not None and n_kv_heads != num_kv_heads:
        raise ValueError(f"key heads {n_kv_heads} != num_kv_heads {num_kv_heads}")

    key_expanded = _expand_kv_for_gqa(key, n_q_heads, n_kv_heads)
    value_expanded = _expand_kv_for_gqa(value, n_q_heads, n_kv_heads)

    outputs = np.zeros_like(query, dtype=np.float32)
    for h in range(n_q_heads):
        outputs[h] = _standard_attention_2d(query[h], key_expanded[h], value_expanded[h])
    return outputs


def _softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax。"""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def compressed_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    rotation_matrix: np.ndarray,
    jl_matrix: np.ndarray | None = None,
    n_bits: int = 4,
    group_size: int = 32,
    enable_qjl: bool = True,
    num_kv_heads: int | None = None,
    return_scores: bool = False,
) -> np.ndarray | dict:
    """压缩注意力计算。

    对 key/value 进行 PolarQuant 压缩，然后在压缩数据上计算注意力。
    """
    if query.ndim == 3 and key.ndim == 3:
        return _compressed_attention_gqa(
            query, key, value, rotation_matrix, jl_matrix,
            n_bits, group_size, enable_qjl, num_kv_heads,
        )

    return _compressed_attention_2d(
        query, key, value, rotation_matrix, jl_matrix,
        n_bits, group_size, enable_qjl, return_scores,
    )


def _compressed_attention_2d(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    rotation_matrix: np.ndarray,
    jl_matrix: np.ndarray | None,
    n_bits: int,
    group_size: int,
    enable_qjl: bool,
    return_scores: bool,
) -> np.ndarray | dict:
    """2D 压缩注意力。"""
    seq_len = key.shape[0]
    d = query.shape[-1]
    n_q = query.shape[0]

    if seq_len == 0:
        out = np.zeros((n_q, d), dtype=np.float32)
        if return_scores:
            return {"output": out, "scores": np.empty((n_q, 0), dtype=np.float32)}
        return out

    # 压缩 key 和 value
    K_hat = np.zeros_like(key, dtype=np.float32)
    V_hat = np.zeros_like(value, dtype=np.float32)
    residuals = []

    for s in range(seq_len):
        ck = compress(key[s], rotation_matrix, n_bits, group_size)
        K_hat[s] = decompress(ck, rotation_matrix)
        residuals.append(key[s] - K_hat[s])

        cv = compress(value[s], rotation_matrix, n_bits, group_size)
        V_hat[s] = decompress(cv, rotation_matrix)

    # 计算注意力分数
    scores = query @ K_hat.T / np.sqrt(d)  # (n_q, seq)

    # QJL 修正
    if enable_qjl and jl_matrix is not None:
        for qi in range(n_q):
            for s in range(seq_len):
                sigs = compute_signatures(residuals[s], jl_matrix)
                corr = compute_correction(query[qi], sigs, jl_matrix)
                scores[qi, s] += corr / np.sqrt(d)

    if return_scores:
        weights = _softmax(scores)
        output = (weights @ V_hat).astype(np.float32)
        return {"output": output, "scores": scores}

    weights = _softmax(scores)
    return (weights @ V_hat).astype(np.float32)


def _compressed_attention_gqa(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    rotation_matrix: np.ndarray,
    jl_matrix: np.ndarray | None,
    n_bits: int,
    group_size: int,
    enable_qjl: bool,
    num_kv_heads: int | None,
) -> np.ndarray:
    """3D GQA 压缩注意力。"""
    n_q_heads = query.shape[0]
    n_kv_heads = key.shape[0]

    key_expanded = _expand_kv_for_gqa(key, n_q_heads, n_kv_heads)
    value_expanded = _expand_kv_for_gqa(value, n_q_heads, n_kv_heads)

    outputs = np.zeros_like(query, dtype=np.float32)
    for h in range(n_q_heads):
        outputs[h] = _compressed_attention_2d(
            query[h], key_expanded[h], value_expanded[h],
            rotation_matrix, jl_matrix,
            n_bits, group_size, enable_qjl, return_scores=False,
        )
    return outputs
