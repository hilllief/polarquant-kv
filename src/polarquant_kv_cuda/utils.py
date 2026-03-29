"""工具函数。"""

import torch


def cosine_similarity_gpu(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算两个 tensor 的余弦相似度。"""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=0).item()
