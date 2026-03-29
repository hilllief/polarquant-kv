# PolarQuant-KV Phase 3: 推理集成 Wrapper — 设计文档

## 引用

- 需求文档: #[[file:docs/specs/phase3-inference-wrapper/requirements.md]]

---

## 1. 模块架构

```
src/polarquant_kv_cuda/inference/
├── __init__.py
├── config.py           # PolarQuantConfig
├── kv_cache.py         # KVCacheManager
├── attention.py        # CompressedMultiHeadAttention
└── transformer.py      # SimpleTransformerLayer (测试用)
```

## 2. 核心类

### PolarQuantConfig
```python
@dataclass
class PolarQuantConfig:
    n_bits: int = 4
    group_size: int = 32
    jl_dim: int = 64
    enable_qjl: bool = False
    enable_compression: bool = True
```

### KVCacheManager
```python
class KVCacheManager:
    def __init__(self, config, max_seq_len, batch, num_heads, head_dim, device)
    def append(self, key, value)  # 压缩并追加
    def get_compressed_kv(self) -> (CompressedKVCacheGPU, CompressedKVCacheGPU)
    def reset(self)
    @property seq_len, memory_bytes, compression_ratio
```

### CompressedMultiHeadAttention
```python
class CompressedMultiHeadAttention(nn.Module):
    def __init__(self, config, num_heads, num_kv_heads, head_dim)
    def forward(self, query, key, value, cache_manager) -> output
```

## 3. 正确性属性

- P1: Prefill-Decode 一致性 — prefill 全量写入 vs decode 逐步写入，注意力输出一致
- P2: 开关等价性 — enable_compression=False 时与 torch SDPA 完全一致
- P3: 显存节省 — 压缩 KV Cache 显存 < 标准的 50%
- P4: 多层精度 — 2+ 层 Transformer 输出余弦相似度 ≥ 0.95
