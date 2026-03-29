# PolarQuant-KV Phase 1: Python 原型验证 — 需求文档

## 概述

用 NumPy/SciPy 实现 PolarQuant + QJL 完整算法的 Python 原型，在 CPU 上验证算法正确性和精度，确定最优超参数，为后续 CUDA 实现提供参考基准。

## 引用

- PRD: #[[file:docs/polarquant-kvcache-prd.md]]

---

## 需求 1: 正交旋转矩阵生成

### 用户故事
作为算法开发者，我需要生成随机正交旋转矩阵，用于消除 KV 向量 channel 间的方差差异，使量化误差均匀分布。

### 验收标准

- AC-1.1: WHEN 指定维度 d 生成旋转矩阵 R，SHALL 返回 d×d 正交矩阵，满足 R^T · R = I（误差 < 1e-6）
- AC-1.2: WHEN 用旋转矩阵 R 变换向量 v，SHALL 保持向量范数不变，即 ||R·v|| = ||v||（相对误差 < 1e-6）
- AC-1.3: WHEN 指定随机种子生成旋转矩阵，SHALL 每次生成相同的矩阵（可复现性）
- AC-1.4: WHEN 维度 d ≤ 0 或非整数，SHALL 抛出 ValueError 并给出明确错误信息

- AC-1.5: WHEN 维度 d 非常大（如 d=4096），SHALL 正常生成正交矩阵，且正交性误差仍 < 1e-5

---

## 需求 2: PolarQuant 极坐标量化

### 用户故事
作为算法开发者，我需要将 FP16/FP32 向量通过极坐标变换 + 分组量化压缩到低比特表示，实现 KV Cache 的高压缩比。

### 验收标准

- AC-2.1: WHEN 输入 FP16 或 FP32 向量 v ∈ R^d，SHALL 执行完整的极坐标量化流程：随机旋转 → 半径提取 → 方向向量分组量化（内部统一转为 FP32 计算）
- AC-2.2: WHEN 使用 4-bit 量化（n_bits=4, group_size=32），SHALL 实现理论压缩比 ≥ 3.8x（相对于 FP16 存储），目标值为 4.0x
- AC-2.3: WHEN 使用 3-bit 量化（n_bits=3, group_size=32），SHALL 实现理论压缩比 ≥ 5.0x，目标值为 5.3x
- AC-2.4: WHEN 对量化后的向量解压还原，SHALL 与原始向量的余弦相似度满足：4-bit 时 ≥ 0.99，3-bit 时 ≥ 0.98，2-bit 时 ≥ 0.95（d=128 基准，其他维度允许 ±0.005 浮动）
- AC-2.5: WHEN group_size 不能整除向量维度 d，SHALL 对最后一组进行 padding 处理，不丢失信息
- AC-2.6: WHEN n_bits 不在 [2, 8] 范围内，SHALL 抛出 ValueError
- AC-2.7: WHEN group_size ≤ 0 或 group_size > d，SHALL 抛出 ValueError
- AC-2.8: WHEN 输入零向量，SHALL 正确处理（半径为 0，方向向量为零向量），不产生 NaN 或 Inf
- AC-2.9: WHEN 压缩完成后，SHALL 返回结构化的 CompressedKV 对象，包含 radius、quantized_direction、group_params 字段，且每个字段可独立访问和检查
- AC-2.10: WHEN 输入向量包含极端值（如 max_float16、min_float16、subnormal），SHALL 正确处理，不产生 NaN、Inf 或溢出


---

## 需求 3: QJL 误差修正

### 用户故事
作为算法开发者，我需要对量化残差进行 JL 随机投影并保留符号位，以极低存储开销修正注意力分数的量化误差。

### 验收标准

- AC-3.1: WHEN 生成 JL 投影矩阵 P (m×d)，SHALL 每个元素服从 N(0, 1/m) 分布
- AC-3.2: WHEN 对量化残差 e 进行 JL 投影，SHALL 输出 m 维投影向量，并提取符号位（每个值 1 bit）
- AC-3.3: WHEN 使用符号位修正注意力分数，SHALL 在 100 个随机样本上，修正后的注意力分数平均 MSE 小于仅用 PolarQuant 的平均 MSE（统计显著性 p < 0.05）
- AC-3.4: WHEN jl_dim (m) ≤ 0，SHALL 抛出 ValueError
- AC-3.5: WHEN 量化残差为零向量（完美量化），SHALL 符号位全为 0，修正量为 0
- AC-3.6: WHEN 指定随机种子生成 JL 投影矩阵，SHALL 每次生成相同的矩阵（可复现性）

---

## 需求 4: 压缩注意力计算

### 用户故事
作为算法开发者，我需要在压缩的 KV Cache 上直接计算注意力分数，验证压缩注意力与标准注意力的等价性。

### 验收标准

- AC-4.1: WHEN 给定 Query 矩阵 Q 和压缩的 Key Cache，SHALL 计算注意力分数 score = Q · decompress(K_compressed)^T / sqrt(d)
- AC-4.2: WHEN 使用 QJL 修正（d=128, 4-bit, jl_dim=64, seq_len ≤ 4096），对标准正态分布的随机 Q/K 向量，SHALL 注意力分数的最大绝对误差 < 0.01（在 95% 的随机试验中成立）
- AC-4.3: WHEN 给定 Query、压缩的 Key 和压缩的 Value，SHALL 计算完整的注意力输出（含 softmax + 加权求和）
- AC-4.4: WHEN 压缩注意力输出与标准注意力输出对比，SHALL 余弦相似度 ≥ 0.995（d=128, 4-bit, jl_dim=64）
- AC-4.5: WHEN 序列长度为 0，SHALL 返回空的注意力输出，不报错
- AC-4.6: WHEN 支持 GQA（Grouped Query Attention），SHALL 正确处理 num_kv_heads < num_q_heads 的情况
- AC-4.7: WHEN 计算压缩注意力时，SHALL 支持可选的 enable_qjl 参数，默认为 True，设为 False 时跳过 QJL 修正，方便对比


---

## 需求 5: 超参数搜索与精度验证

### 用户故事
作为算法开发者，我需要系统地评估不同超参数组合下的压缩比和精度，确定最优配置。

### 验收标准

- AC-5.1: WHEN 指定超参数范围（n_bits ∈ {2,3,4,6,8}, group_size ∈ {16,32,64,128}, jl_dim ∈ {32,64,128}），SHALL 输出每种组合的压缩比和精度指标
- AC-5.2: WHEN 计算压缩比，SHALL 精确计算压缩后的存储字节数（含量化值、group 参数、半径、符号位），不遗漏任何开销
- AC-5.3: WHEN 评估精度，SHALL 报告以下指标：余弦相似度（向量级）、注意力分数 MSE、注意力输出余弦相似度
- AC-5.4: WHEN 生成验证报告的推荐配置，SHALL 基于"压缩比 ≥ 4x 且余弦相似度 ≥ 0.995"的帕累托最优准则选择，若无满足条件的配置，SHALL 明确提示
- AC-5.5: WHEN 计算压缩比，SHALL 使用公式：compression_ratio = (d × 16) / (16 + n_bits × d + num_groups × 32 + jl_dim)，其中 16 为半径的 FP16 位数，num_groups = ceil(d / group_size)，32 为每组 min+scale 的 FP16 位数，jl_dim 为 QJL 符号位开销
- AC-5.6: WHEN 执行超参数搜索，SHALL 通过 logging 模块输出当前进度（已完成/总数），每完成一个配置输出一次
- AC-5.7: WHEN Phase 1 验证完成，SHALL 输出 Phase 1 完成判定报告，包含：(1) 是否存在至少一组超参数满足压缩比 ≥ 4x 且余弦相似度 ≥ 0.995，(2) QJL 修正是否统计显著地降低了注意力误差，(3) 推荐进入 Phase 2 的超参数配置

---

## 需求 6: Batch 操作支持

### 用户故事
作为算法开发者，我需要支持批量压缩和解压操作，模拟实际推理中多 head、多 token 的场景。

### 验收标准

- AC-6.1: WHEN 输入 batch 的 KV 向量（shape: [batch, num_heads, seq_len, head_dim]），SHALL 对每个 head 的每个 token 独立执行压缩
- AC-6.2: WHEN batch 压缩结果与逐个压缩结果对比，SHALL 数值完全一致（bit-exact），实现时 SHALL 确保 batch 操作内部的计算顺序与逐个操作一致
- AC-6.3: WHEN batch_size = 0 或 seq_len = 0，SHALL 返回空的压缩结果，不报错
- AC-6.4: WHEN 执行 batch 压缩操作，SHALL 在压缩前检查预估内存占用，WHEN 预估内存超过可用内存的 80%，SHALL 发出警告（logging.warning）


---

## 正确性属性候选（供设计阶段使用）

| 属性 | 关联需求 | 描述 |
|------|---------|------|
| 正交性保持 | AC-1.1, AC-1.2 | 旋转矩阵保持向量范数和内积 |
| 压缩-解压往返 | AC-2.1, AC-2.4 | 压缩后解压的向量与原始向量高度相似 |
| 压缩比下界 | AC-2.2, AC-2.3, AC-5.5 | 给定 n_bits 和 group_size，压缩比有理论下界且计算公式确定 |
| QJL 修正有效性 | AC-3.3 | QJL 修正后的误差统计显著地小于无修正的误差 |
| 注意力等价性 | AC-4.2, AC-4.4 | 压缩注意力与标准注意力的输出高度一致 |
| Batch 一致性 | AC-6.2 | batch 操作与逐个操作结果一致 |
| 零向量安全 | AC-2.8, AC-3.5 | 零向量输入不产生 NaN/Inf |
| 极端值鲁棒性 | AC-2.10 | 极端值输入不产生 NaN/Inf/溢出 |

---

## 约束

- 仅使用 Python + NumPy + SciPy，不依赖 CUDA
- 不追求性能，追求正确性和可读性
- 所有随机操作支持种子控制，确保可复现
- 随机矩阵生成使用 numpy.random.Generator（非密码学安全），仅用于算法正确性验证
- 代码结构与 PRD 中的技术架构对齐（python/polarquant_kv/）
- Python 原型的核心函数签名（compress、decompress、compressed_attention）SHALL 与未来 CUDA 版本的 Python 绑定保持一致