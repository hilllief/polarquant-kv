# PolarQuant-KV Phase 1: Python 原型验证 — 设计文档

## 引用

- 需求文档: #[[file:docs/specs/phase1-python-prototype/requirements.md]]
- PRD: #[[file:docs/polarquant-kvcache-prd.md]]

---

## 1. 模块架构

```
python/polarquant_kv/
├── __init__.py              # 公开 API 导出
├── rotation.py              # 需求 1: 正交旋转矩阵生成
├── quantizer.py             # 需求 2: PolarQuant 极坐标量化
├── qjl.py                   # 需求 3: QJL 误差修正
├── attention.py             # 需求 4: 压缩注意力计算
├── benchmark.py             # 需求 5: 超参数搜索与精度验证
├── types.py                 # 数据类型定义（CompressedKV 等）
└── utils.py                 # 工具函数（压缩比计算等）
```

---

## 2. 核心数据结构

### 2.1 CompressedKV（需求 2, AC-2.9）

```python
@dataclass
class CompressedKV:
    radius: np.ndarray           # shape: (...), FP32, 向量范数
    quantized_direction: np.ndarray  # shape: (..., d), uint8, 量化后的方向向量
    group_mins: np.ndarray       # shape: (..., num_groups), FP32, 每组最小值
    group_scales: np.ndarray     # shape: (..., num_groups), FP32, 每组缩放因子
    n_bits: int                  # 量化位数
    group_size: int              # 分组大小
    original_dim: int            # 原始向量维度（用于 unpadding）
```

### 2.2 QJLSignatures（需求 3, AC-3.2）

```python
@dataclass
class QJLSignatures:
    signs: np.ndarray            # shape: (..., jl_dim), bool, 符号位
    jl_dim: int                  # JL 投影维度
```

### 2.3 CompressedKVCache（需求 4, 组合结构）

```python
@dataclass
class CompressedKVCache:
    compressed_keys: CompressedKV
    compressed_values: CompressedKV
    key_signatures: QJLSignatures | None  # QJL 修正数据（可选）
```

---

## 3. 模块设计

### 3.1 rotation.py — 正交旋转矩阵

#### 函数签名

```python
def generate_rotation_matrix(
    d: int,
    seed: int | None = None,
) -> np.ndarray:
    """生成 d×d 随机正交矩阵。使用 QR 分解。"""

def rotate(
    v: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """对向量（或 batch 向量）应用旋转。"""

def inverse_rotate(
    v: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """逆旋转（R^T · v）。"""
```

#### 实现策略
- 使用 `scipy.stats.ortho_group` 或 `np.linalg.qr` 生成正交矩阵
- 种子通过 `np.random.Generator(np.random.PCG64(seed))` 控制

### 3.2 quantizer.py — PolarQuant 极坐标量化

#### 函数签名

```python
def compress(
    v: np.ndarray,
    rotation_matrix: np.ndarray,
    n_bits: int = 4,
    group_size: int = 32,
) -> CompressedKV:
    """PolarQuant 压缩：旋转 → 半径分离 → 分组量化。"""

def decompress(
    compressed: CompressedKV,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """PolarQuant 解压：反量化 → 半径恢复 → 逆旋转。"""
```

#### 实现策略
- 输入统一转为 FP32（AC-2.1）
- 零向量特殊处理：半径为 0 时跳过方向量化，直接返回零（AC-2.8）
- 极端值处理：在旋转前 clip 到安全范围（AC-2.10）
- padding：当 d % group_size != 0 时，对方向向量末尾补零（AC-2.5）
- 分组量化：每组独立计算 min/max，线性映射到 [0, 2^n_bits - 1]

### 3.3 qjl.py — QJL 误差修正

#### 函数签名

```python
def generate_jl_matrix(
    jl_dim: int,
    d: int,
    seed: int | None = None,
) -> np.ndarray:
    """生成 JL 随机投影矩阵 P (jl_dim × d)。"""

def compute_signatures(
    residual: np.ndarray,
    jl_matrix: np.ndarray,
) -> QJLSignatures:
    """计算量化残差的 JL 投影符号位。"""

def compute_correction(
    query: np.ndarray,
    signatures: QJLSignatures,
    jl_matrix: np.ndarray,
) -> np.ndarray:
    """基于符号位计算注意力分数修正量。"""
```

#### 实现策略
- JL 矩阵元素 ~ N(0, 1/m)，通过 `rng.normal(0, 1/sqrt(m), size=(m, d))` 生成（AC-3.1）
- 符号位提取：`signs = (P @ e) >= 0`（AC-3.2）
- 修正公式：`correction = (1/m) * query @ P^T @ diag(2*signs - 1) @ ones`（简化的无偏估计）

### 3.4 attention.py — 压缩注意力计算

#### 函数签名

```python
def standard_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    num_kv_heads: int | None = None,
) -> np.ndarray:
    """标准注意力计算（参考实现）。"""

def compressed_attention(
    query: np.ndarray,
    compressed_keys: CompressedKV,
    compressed_values: CompressedKV,
    rotation_matrix: np.ndarray,
    key_signatures: QJLSignatures | None = None,
    jl_matrix: np.ndarray | None = None,
    enable_qjl: bool = True,
    num_kv_heads: int | None = None,
) -> np.ndarray:
    """压缩注意力计算。"""
```

#### 实现策略
- 标准注意力作为参考基准，用于精度对比
- GQA 支持：当 num_kv_heads < num_q_heads 时，对 KV 做 repeat_interleave（AC-4.6）
- enable_qjl=False 时跳过修正项（AC-4.7）
- seq_len=0 时返回空数组（AC-4.5）

### 3.5 benchmark.py — 超参数搜索

#### 函数签名

```python
def compute_compression_ratio(
    d: int,
    n_bits: int,
    group_size: int,
    jl_dim: int,
) -> float:
    """根据 AC-5.5 公式计算理论压缩比。"""

def evaluate_config(
    n_bits: int,
    group_size: int,
    jl_dim: int,
    d: int = 128,
    num_samples: int = 100,
    seed: int = 42,
) -> dict:
    """评估单个超参数配置的压缩比和精度。"""

def hyperparameter_search(
    n_bits_range: list[int],
    group_size_range: list[int],
    jl_dim_range: list[int],
    d: int = 128,
) -> dict:
    """遍历超参数组合，输出完整报告。"""

def generate_phase1_report(search_results: dict) -> str:
    """生成 Phase 1 完成判定报告（AC-5.7）。"""
```

### 3.6 utils.py — 工具函数

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度。"""

def attention_score_mse(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """计算注意力分数 MSE。"""

def estimate_memory_bytes(
    batch: int, num_heads: int, seq_len: int, head_dim: int,
    n_bits: int, group_size: int, jl_dim: int,
) -> int:
    """预估压缩后的内存占用（AC-6.4）。"""
```

---

## 4. 正确性属性（Hypothesis 属性测试）

### P1: 正交性保持（AC-1.1, AC-1.2）

```
∀ d ∈ [2, 512], seed ∈ Z:
  R = generate_rotation_matrix(d, seed)
  assert ||R^T · R - I||_F < 1e-6
  ∀ v ∈ R^d:
    assert | ||R·v|| - ||v|| | / ||v|| < 1e-6
```

- hypothesis 策略：`d = st.integers(2, 512)`, `v = st.arrays(np.float32, d, elements=st.floats(-1e3, 1e3))`

### P2: 压缩-解压往返（AC-2.1, AC-2.4）

```
∀ v ∈ R^128 (非零), n_bits ∈ {2,3,4,6,8}:
  compressed = compress(v, R, n_bits=n_bits)
  v_hat = decompress(compressed, R)
  cosine_sim(v, v_hat) ≥ threshold[n_bits]
```

- threshold: {2: 0.95, 3: 0.98, 4: 0.99, 6: 0.995, 8: 0.999}
- hypothesis 策略：`v = st.arrays(np.float32, 128, elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False))`
- 过滤零向量

### P3: 压缩比下界（AC-2.2, AC-2.3, AC-5.5）

```
∀ d ∈ {64, 128, 256}, n_bits ∈ {2,3,4,6,8}, group_size ∈ {16,32,64}:
  ratio = compute_compression_ratio(d, n_bits, group_size, jl_dim=64)
  assert ratio == (d * 16) / (16 + n_bits * d + ceil(d/group_size) * 32 + 64)
```

- 这是确定性属性，不需要随机输入
- 验证公式实现与 AC-5.5 定义一致

### P4: QJL 修正有效性（AC-3.3）

```
∀ Q ∈ R^{1×128}, K ∈ R^{seq×128} (标准正态分布), n_bits=4, jl_dim=64:
  score_true = Q · K^T / sqrt(128)
  score_quant = Q · decompress(compress(K))^T / sqrt(128)
  score_corrected = score_quant + correction(Q, signatures, P)
  MSE(score_corrected, score_true) < MSE(score_quant, score_true)
```

- 统计属性：在 100 个样本上，修正后的平均 MSE < 无修正的平均 MSE
- hypothesis 策略：`seq_len = st.integers(1, 256)`, Q/K ~ 标准正态

### P5: 注意力等价性（AC-4.2, AC-4.4）

```
∀ Q ∈ R^{1×128}, K,V ∈ R^{seq×128}:
  output_true = standard_attention(Q, K, V)
  output_compressed = compressed_attention(Q, compress(K), compress(V), ...)
  cosine_sim(output_true, output_compressed) ≥ 0.995
```

- hypothesis 策略：`seq_len = st.integers(1, 128)`, Q/K/V ~ 标准正态

### P6: Batch 一致性（AC-6.2）

```
∀ batch ∈ [1,4], num_heads ∈ [1,8], seq_len ∈ [1,32], head_dim=128:
  KV_batch = random(batch, num_heads, seq_len, head_dim)
  result_batch = compress_batch(KV_batch)
  for b, h, s:
    result_single = compress(KV_batch[b, h, s])
    assert result_batch[b, h, s] == result_single  # bit-exact
```

- hypothesis 策略：小规模 batch 参数组合

### P7: 零向量安全（AC-2.8, AC-3.5）

```
∀ d ∈ {64, 128, 256}:
  v = zeros(d)
  compressed = compress(v, R)
  assert compressed.radius == 0
  assert not any(isnan(decompress(compressed, R)))
  assert not any(isinf(decompress(compressed, R)))
  
  residual = zeros(d)
  sigs = compute_signatures(residual, P)
  assert all(sigs.signs == False)
  assert compute_correction(q, sigs, P) == 0
```

### P8: 极端值鲁棒性（AC-2.10）

```
∀ d ∈ {64, 128}:
  v_extreme = array with max_float16, min_float16, subnormal values
  compressed = compress(v_extreme, R)
  v_hat = decompress(compressed, R)
  assert not any(isnan(v_hat))
  assert not any(isinf(v_hat))
```

- hypothesis 策略：`elements=st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=True)`

---

## 5. 冗余消除分析

| 候选属性 | 保留/消除 | 理由 |
|---------|----------|------|
| P1 正交性保持 | 保留 | 独立的数学性质，不被其他属性覆盖 |
| P2 压缩-解压往返 | 保留 | 核心质量指标，P5 依赖它但不等价 |
| P3 压缩比下界 | 保留 | 确定性公式验证，与精度属性正交 |
| P4 QJL 修正有效性 | 保留 | 统计属性，P5 包含 QJL 但不单独验证其有效性 |
| P5 注意力等价性 | 保留 | 端到端属性，但不能替代 P2（P2 验证单向量，P5 验证注意力流程） |
| P6 Batch 一致性 | 保留 | 实现正确性属性，其他属性不覆盖 batch 逻辑 |
| P7 零向量安全 | 保留 | 边界条件，P2 的 hypothesis 策略会过滤零向量 |
| P8 极端值鲁棒性 | 保留 | 边界条件，P2 的 elements 范围不覆盖极端值 |

结论：8 个属性全部保留，无冗余。

---

## 6. Mock 策略

本项目为纯算法库，无外部依赖需要 Mock。

| 组件 | Mock 策略 |
|------|----------|
| 随机数生成 | 通过 seed 参数控制，不需要 Mock |
| NumPy/SciPy | 直接使用，不 Mock |
| 文件 I/O | benchmark 报告输出可 Mock，但建议直接测试字符串输出 |
| logging | 使用 `caplog` fixture 捕获日志断言 |

---

## 7. 测试依赖

```
pytest >= 8.0
hypothesis >= 6.100
scipy >= 1.12
numpy >= 1.26
psutil >= 5.9       # AC-6.4 内存检查
```
