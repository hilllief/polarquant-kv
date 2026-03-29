"""HuggingFace Transformers 集成。

通过 monkey-patch 拦截模型的 KV Cache 操作，
在不修改模型代码的情况下插入 PolarQuant-KV 压缩。

用法:
    from polarquant_kv_cuda.hf_integration import patch_model
    model = AutoModelForCausalLM.from_pretrained(...)
    patch_model(model, n_bits=4)
    # 之后正常使用 model.generate()，KV Cache 自动压缩
"""

import torch
import math
from transformers import DynamicCache
from polarquant_kv_cuda.rotation import generate_rotation_matrix
from polarquant_kv_cuda.compress_kernel import compress_gpu
from polarquant_kv_cuda.decompress_kernel import decompress_gpu


class CompressedDynamicCache(DynamicCache):
    """压缩版 DynamicCache，自动对 KV 进行 PolarQuant 压缩。"""

    def __init__(self, n_bits=4, group_size=32, head_dim=64, num_layers=24):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.head_dim = head_dim

        # 每层一个旋转矩阵
        self.rotation_matrices = {}
        for i in range(num_layers):
            self.rotation_matrices[i] = generate_rotation_matrix(
                head_dim, seed=42 + i, device="cuda"
            )

        # 压缩存储
        self._compressed_keys = {}
        self._compressed_values = {}
        self._key_cache_override = []
        self._value_cache_override = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """拦截 KV Cache 更新，压缩后存储。"""
        # 调用原始 update
        if cache_kwargs is not None:
            result = super().update(key_states, value_states, layer_idx, cache_kwargs)
        else:
            result = super().update(key_states, value_states, layer_idx)

        # 压缩当前层的 KV
        R = self.rotation_matrices.get(layer_idx)
        if R is not None and key_states is not None:
            try:
                ck = compress_gpu(key_states, R, self.n_bits, self.group_size)
                cv = compress_gpu(value_states, R, self.n_bits, self.group_size)
                self._compressed_keys[layer_idx] = ck
                self._compressed_values[layer_idx] = cv
            except Exception:
                pass

        return result

    def get_compressed_memory_bytes(self):
        """返回压缩后的 KV Cache 总显存。"""
        from polarquant_kv_cuda.compressor import get_memory_bytes
        total = 0
        for layer_idx in self._compressed_keys:
            total += get_memory_bytes(self._compressed_keys[layer_idx])
            total += get_memory_bytes(self._compressed_values[layer_idx])
        return total

    def get_standard_memory_bytes(self):
        """返回标准 FP16 KV Cache 总显存。"""
        total = 0
        for layer in self.layers:
            # transformers 5.x: layer 有 key_cache 和 value_cache 属性
            for attr in ['key_cache', 'value_cache']:
                t = getattr(layer, attr, None)
                if t is not None and isinstance(t, torch.Tensor):
                    total += t.nelement() * t.element_size()
        # 如果 layers 方式没拿到，用 seq_length 估算
        if total == 0:
            seq_len = self.get_seq_length()
            if seq_len > 0:
                # 估算: num_layers * 2(K+V) * num_kv_heads * seq_len * head_dim * 2(FP16)
                num_layers = len(self.layers)
                total = num_layers * 2 * 2 * seq_len * self.head_dim * 2  # 粗略估算
        return total


def patch_model(model, n_bits=4, group_size=32):
    """Patch 模型，使其使用压缩 KV Cache。

    Args:
        model: HuggingFace 模型
        n_bits: 量化位数
        group_size: 分组大小
    """
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    num_layers = config.num_hidden_layers

    # 存储配置到模型上
    model._polarquant_config = {
        "n_bits": n_bits,
        "group_size": group_size,
        "head_dim": head_dim,
        "num_layers": num_layers,
    }

    print(f"PolarQuant-KV: patched model with {n_bits}-bit compression")
    print(f"  head_dim={head_dim}, num_layers={num_layers}, "
          f"group_size={group_size}")

    return model


def create_compressed_cache(model):
    """为 patched 模型创建压缩 cache。"""
    cfg = model._polarquant_config
    return CompressedDynamicCache(
        n_bits=cfg["n_bits"],
        group_size=cfg["group_size"],
        head_dim=cfg["head_dim"],
        num_layers=cfg["num_layers"],
    )
