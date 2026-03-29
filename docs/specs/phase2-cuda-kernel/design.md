# PolarQuant-KV Phase 2: CUDA Kernel 实现 — 设计文档

## 引用

- 需求文档: #[[file:docs/specs/phase2-cuda-kernel/requirements.md]]
- Phase 1 设计: #[[file:docs/specs/phase1-python-prototype/design.md]]
- PRD: #[[file:docs/polarquant-kvcache-prd.md]]

---

## 1. 模块架构

```
src/polarquant_kv_cuda/
├── __init__.py              # 公开 API 导出
├── types.py                 # GPU 数据结构（CompressedKVCache 等）
├── rotation.py              # 旋转矩阵（PyTorch 实现，非 Triton）
├── compress_kernel.py       # 需求 1: Triton 压缩 kernel
├── decompress_kernel.py     # 需求 2: Triton 解压 kernel
├── qjl_kernel.py            # 需求 3: Triton QJL 投影 kernel
├── attention_kernel.py      # 需求 4: Triton 融合注意力 kernel
├── compressor.py            # 高层 API（compress/decompress/attention）
├── benchmark.py             # 需求 5: 性能基准测试
└── utils.py                 # 工具函数
```

---

## 2. 核心数据结构

### 2.1 CompressedKVCacheGPU（需求 1, AC-1.1）

```python
@dataclass
class CompressedKVCacheGPU:
    radius: torch.Tensor          # [batch, num_heads, seq_len], FP16
    quantized_direction: torch.Tensor  # [batch, num_heads, seq_len, d_padded], uint8
    group_mins: torch.Tensor      # [batch, num_heads, seq_len, num_groups], FP16
    group_scales: torch.Tensor    # [batch, num_heads, seq_len, num_groups], FP16
    qjl_signs: torch.Tensor | None  # [batch, num_heads, seq_len, jl_dim_packed], uint8 (bit packed)
    residual_norms: torch.Tensor | None  # [batch, num_heads, seq_len], FP16
    n_bits: int
    group_size: int
    original_dim: int
    seq_len: int                  # 当前有效序列长度（支持增量追加）
    max_seq_len: int              # 预分配的最大序列长度
```

### 2.2 预分配策略（AC-1.6 增量追加）

```python
# 初始化时预分配 max_seq_len 的空间
cache = CompressedKVCacheGPU.allocate(
    batch=1, num_heads=32, max_seq_len=4096, head_dim=128,
    n_bits=4, group_size=32, jl_dim=64, device="cuda"
)
# 追加新 token 时只写入 cache.seq_len 位置
cache.append(compressed_kv_single_token)
```

---

## 3. Triton Kernel 设计

### 3.1 compress_kernel — PolarQuant 压缩

#### 并行策略
- 每个 program 处理一个向量（一个 head 的一个 token）
- grid = (batch * num_heads * seq_len,)
- BLOCK_SIZE = head_dim（128，一个 warp 可处理）

#### 算法步骤（单个 program 内）
1. 加载 FP16 向量 → 转 FP32
2. 矩阵-向量乘法：v_rotated = R @ v（R 预加载到 shared memory）
3. 计算半径：radius = norm(v_rotated)
4. 归一化：direction = v_rotated / radius（零向量特殊处理）
5. 分组量化：每 group_size 维计算 min/max → 线性量化到 n_bits
6. 写回：radius, quantized_direction, group_mins, group_scales

#### Autotune 配置
```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8),
    ],
    key=["head_dim"],
)
```

### 3.2 decompress_kernel — PolarQuant 解压

#### 并行策略
- 与压缩相同：每个 program 处理一个向量
- grid = (batch * num_heads * seq_len,)

#### 算法步骤
1. 加载量化数据 + group 参数
2. 反量化：val = quantized * scale + min
3. 恢复半径：v_rotated = direction * radius
4. 逆旋转：v = R^T @ v_rotated
5. 写回 FP16

### 3.3 qjl_kernel — QJL 投影

#### 并行策略
- 每个 program 处理一个向量的 JL 投影
- grid = (batch * num_heads * seq_len,)

#### 算法步骤
1. 加载残差向量（原始 - 解压后）
2. 矩阵-向量乘法：projected = P @ residual
3. 提取符号位：signs = projected >= 0
4. Bit packing：每 8 个符号位打包为 1 个 uint8
5. 计算残差范数
6. 写回 packed signs + residual_norm

### 3.4 attention_kernel — 融合注意力

#### 并行策略
- 每个 program 处理一个 query token 对所有 key tokens 的注意力
- grid = (batch * num_q_heads * seq_len_q,)
- 内部循环遍历 key tokens（分块处理）

#### 算法步骤（Flash Attention 风格）
```
for each key block (BLOCK_K tokens):
    1. 解压 K block: dequantize → inverse_rotate
    2. 计算 score = Q @ K_block^T / sqrt(d)
    3. (可选) QJL 修正: score += correction
    4. 在线 softmax 更新
    5. 解压 V block
    6. 累加 output += weights @ V_block
```

#### GQA 支持
- num_kv_heads < num_q_heads 时，多个 Q head 共享同一个 KV head
- 通过 head index 映射：kv_head_idx = q_head_idx // (num_q_heads // num_kv_heads)

---

## 4. 高层 API（compressor.py）

### 函数签名（与 Phase 1 对齐）

```python
def compress(
    kv: torch.Tensor,                    # [batch, num_heads, seq_len, head_dim], FP16, GPU
    rotation_matrix: torch.Tensor,       # [head_dim, head_dim], FP32, GPU
    n_bits: int = 4,
    group_size: int = 32,
    jl_matrix: torch.Tensor | None = None,  # [jl_dim, head_dim], FP32, GPU
) -> CompressedKVCacheGPU:
    """GPU 压缩。"""

def decompress(
    cache: CompressedKVCacheGPU,
    rotation_matrix: torch.Tensor,
) -> torch.Tensor:
    """GPU 解压，返回 [batch, num_heads, seq_len, head_dim] FP16。"""

def compressed_attention(
    query: torch.Tensor,                 # [batch, num_q_heads, seq_len_q, head_dim], FP16
    compressed_keys: CompressedKVCacheGPU,
    compressed_values: CompressedKVCacheGPU,
    rotation_matrix: torch.Tensor,
    jl_matrix: torch.Tensor | None = None,
    enable_qjl: bool = True,
    num_kv_heads: int | None = None,
) -> torch.Tensor:
    """融合注意力计算。"""

def get_memory_bytes(cache: CompressedKVCacheGPU) -> int:
    """返回压缩 KV Cache 的精确显存占用字节数。"""
```

---

## 5. rotation.py — 旋转矩阵（PyTorch）

不用 Triton，直接用 PyTorch 的矩阵乘法（已经很快）。

```python
def generate_rotation_matrix(d: int, seed: int | None = None, device="cuda") -> torch.Tensor:
    """生成正交旋转矩阵，放到 GPU 上。"""

def rotate(v: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """v @ R^T，PyTorch matmul。"""

def inverse_rotate(v: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """v @ R，PyTorch matmul。"""
```

---

## 6. 正确性属性（Hypothesis + PyTorch）

### P1: GPU-CPU 数值一致性（AC-1.2, AC-2.2, AC-6.1）

```
∀ v ∈ R^128 (FP16, 随机):
  compressed_gpu = gpu_compress(v.cuda(), R.cuda())
  compressed_cpu = cpu_compress(v.numpy(), R.numpy())
  cosine_sim(decompress_gpu, decompress_cpu) ≥ 0.999
```

### P2: 压缩-解压往返 GPU（AC-1.1, AC-2.1）

```
∀ v ∈ R^128 (FP16, GPU, 非零):
  v_hat = decompress(compress(v, R), R)
  cosine_sim(v, v_hat) ≥ 0.99 (4-bit)
```

### P3: 融合注意力等价性（AC-4.2）

```
∀ Q, K, V (FP16, GPU):
  out_standard = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
  out_compressed = compressed_attention(Q, compress(K), compress(V), R)
  cosine_sim(out_standard, out_compressed) ≥ 0.985
```

### P4: 零向量安全 GPU（AC-1.4）

```
∀ d ∈ {64, 128, 256}:
  v = torch.zeros(d, dtype=torch.float16, device="cuda")
  compressed = compress(v, R)
  v_hat = decompress(compressed, R)
  assert not torch.any(torch.isnan(v_hat))
  assert not torch.any(torch.isinf(v_hat))
```

### P5: 增量追加一致性（AC-1.6）

```
∀ KV_batch (seq_len tokens):
  # 全量压缩
  cache_full = compress(KV_batch)
  # 逐 token 追加
  cache_incr = allocate_empty()
  for t in range(seq_len):
    cache_incr.append(compress(KV_batch[:,:,t:t+1,:]))
  assert cache_full == cache_incr  # bit-exact
```

### P6: Bit Packing 往返（AC-3.3）

```
∀ signs ∈ {True, False}^64:
  packed = bit_pack(signs)
  unpacked = bit_unpack(packed)
  assert signs == unpacked  # bit-exact
```

---

## 7. 冗余消除分析

| 候选属性 | 保留/消除 | 理由 |
|---------|----------|------|
| P1 GPU-CPU 一致性 | 保留 | 跨平台验证，P2 不覆盖 |
| P2 压缩-解压往返 GPU | 保留 | GPU 特有的精度验证 |
| P3 融合注意力等价性 | 保留 | 端到端属性，P2 不覆盖注意力流程 |
| P4 零向量安全 GPU | 保留 | 边界条件，P2 的 hypothesis 会过滤零向量 |
| P5 增量追加一致性 | 保留 | 新增功能，其他属性不覆盖 |
| P6 Bit Packing 往返 | 保留 | QJL 特有，其他属性不覆盖 |

结论：6 个属性全部保留，无冗余。

---

## 8. Mock 策略

| 组件 | Mock 策略 |
|------|----------|
| GPU 硬件 | 不 Mock，直接在 GPU 上测试 |
| Phase 1 Python 原型 | 直接导入 polarquant_kv 包作为参考实现 |
| PyTorch | 直接使用 |
| Triton | 直接使用 |
| 性能测量 | torch.cuda.Event，不 Mock |

---

## 9. 测试依赖

```
torch >= 2.0
triton >= 3.0
pytest >= 8.0
hypothesis >= 6.100
numpy >= 1.26
scipy >= 1.12
polarquant-kv  # Phase 1 Python 原型（作为参考实现）
```
