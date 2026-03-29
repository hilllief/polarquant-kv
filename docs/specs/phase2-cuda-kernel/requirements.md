# PolarQuant-KV Phase 2: CUDA Kernel 实现 — 需求文档

## 概述

用 Triton 实现 PolarQuant + QJL 的 GPU kernel，在 GPU 上完成压缩/解压/融合注意力计算，验证与 Phase 1 Python 原型的数值一致性，并达到性能目标。

## 引用

- PRD: #[[file:docs/polarquant-kvcache-prd.md]]
- Phase 1 需求: #[[file:docs/specs/phase1-python-prototype/requirements.md]]
- Phase 1 报告: #[[file:docs/phase1-report.md]]

## 环境约束

- GPU: RTX 5060 Ti (Blackwell, sm_120)
- CUDA Toolkit: 13.2
- 实现工具: CuPy (Raw CUDA kernel via RawKernel/ElementwiseKernel)
- Python 绑定: 直接通过 Triton 的 Python API 调用

---

## 需求 1: Triton PolarQuant 压缩 Kernel

### 用户故事
作为推理框架开发者，我需要在 GPU 上高效执行 PolarQuant 压缩，避免 CPU-GPU 数据传输开销。

### 验收标准

- AC-1.1: WHEN 输入 GPU 上的 FP16 tensor (shape: [batch, num_heads, seq_len, head_dim])，SHALL 在 GPU 上完成压缩，输出包含 radius (FP16)、quantized_direction (uint8)、group_mins (FP16)、group_scales (FP16) 的结构化 tensor 组，不经过 CPU
- AC-1.2: WHEN 使用相同的旋转矩阵和量化参数，SHALL 与 Phase 1 Python 原型的压缩结果数值一致（余弦相似度 ≥ 0.999）
- AC-1.3: WHEN head_dim=128, n_bits=4, group_size=32，SHALL 单次压缩的 kernel 执行时间 < 10μs（per head, per token），使用 torch.cuda.Event 测量，取 100 次热启动的中位数
- AC-1.4: WHEN 输入零向量，SHALL 正确处理，不产生 NaN 或 Inf
- AC-1.5: WHEN 输入包含 FP16 极端值，SHALL 正确处理，不产生 NaN、Inf 或溢出
- AC-1.6: WHEN 新 token 到达时，SHALL 支持将单个 token 的 KV 向量追加压缩到已有的压缩 KV Cache 中，无需重新压缩全部历史
- AC-1.7: WHEN Triton kernel 使用 autotune，SHALL 在目标 GPU 上完成 autotuning 并缓存最优配置
- AC-1.8: WHEN GPU 显存不足以完成压缩操作，SHALL 抛出明确的 RuntimeError 并给出所需显存和可用显存的信息
- AC-1.9: WHEN 支持的量化配置，SHALL 至少支持 n_bits ∈ {2, 3, 4, 6, 8}，group_size ∈ {16, 32, 64, 128}，与 Phase 1 保持一致

---

## 需求 2: Triton PolarQuant 解压 Kernel

### 用户故事
作为推理框架开发者，我需要在 GPU 上高效执行 PolarQuant 解压，用于注意力计算。

### 验收标准

- AC-2.1: WHEN 输入压缩后的 tensor 组，SHALL 在 GPU 上完成解压，输出 FP16 tensor
- AC-2.2: WHEN 使用相同参数，SHALL 与 Phase 1 Python 原型的解压结果数值一致（余弦相似度 ≥ 0.999）
- AC-2.3: WHEN 解压后与原始向量对比，SHALL 余弦相似度满足 Phase 1 验证的阈值（4-bit ≥ 0.99）

---

## 需求 3: Triton QJL 投影 Kernel

### 用户故事
作为推理框架开发者，我需要在 GPU 上计算量化残差的 JL 投影符号位。

### 验收标准

- AC-3.1: WHEN 输入量化残差 tensor 和 JL 矩阵，SHALL 在 GPU 上计算投影并提取符号位
- AC-3.2: WHEN 使用相同参数（FP32 精度计算），SHALL 与 Phase 1 Python 原型的符号位结果一致率 ≥ 99%（允许浮点精度差异导致的边界值翻转）
- AC-3.3: WHEN 符号位用 bit packing 存储（64 个符号位 = 8 字节），SHALL 正确打包和解包

---

## 需求 4: Triton 融合注意力 Kernel

### 用户故事
作为推理框架开发者，我需要在一个函数调用中完成解压 + 注意力分数计算 + QJL 修正 + softmax + 加权求和，避免中间结果的显存开销。

### 验收标准

- AC-4.1: WHEN 给定 Query tensor (FP16, GPU) 和已压缩的 KV Cache tensor 组，SHALL 通过单个 Python 函数调用完成完整的注意力计算，内部可使用多个 Triton kernel 但对用户透明
- AC-4.2: WHEN 与标准注意力（未压缩）对比，SHALL 注意力输出的余弦相似度 ≥ 0.985（d=128, 4-bit）
- AC-4.3: WHEN 支持 GQA，SHALL 正确处理 num_kv_heads < num_q_heads
- AC-4.4: WHEN enable_qjl=False，SHALL 跳过 QJL 修正步骤
- AC-4.5: WHEN 序列长度为 0，SHALL 返回零 tensor，不报错
- AC-4.6: WHEN 注意力计算相比 PyTorch 标准 scaled_dot_product_attention（FP16, 未压缩），SHALL 在 seq_len ≥ 2048 时速度提升 ≥ 1.5x
- AC-4.7: WHEN Triton kernel 执行出错（如非法内存访问），SHALL 抛出明确的异常而非静默失败
- AC-4.8: WHEN seq_len ≥ 4096，SHALL 注意力计算加速目标为 ≥ 2x（长序列下内存带宽节省更显著）

---

## 需求 5: 性能基准测试

### 用户故事
作为算法开发者，我需要系统地测量 CUDA kernel 的性能，验证是否达到 PRD 的性能目标。

### 验收标准

- AC-5.1: WHEN 运行压缩基准测试，SHALL 报告：压缩延迟（μs/token/head）、吞吐量（tokens/s）
- AC-5.2: WHEN 运行注意力基准测试，SHALL 报告：注意力延迟（ms）、相对于标准注意力的加速比
- AC-5.3: WHEN 运行内存基准测试，SHALL 报告：KV Cache 显存占用（MB）、压缩比
- AC-5.4: WHEN 生成性能报告，SHALL 包含不同 seq_len（128, 512, 2048, 4096）下的性能数据
- AC-5.5: WHEN 查询压缩 KV Cache 的显存占用，SHALL 返回精确的字节数
- AC-5.6: WHEN 生成性能报告，SHALL 包含不同 batch_size（1, 4, 16）下的性能数据
- AC-5.7: WHEN 重复运行性能基准 10 次，SHALL 延迟的标准差 < 均值的 10%
- AC-5.8: WHEN Phase 2 完成，SHALL 输出完成判定报告，包含：(1) 所有 kernel 的性能数据，(2) 与 Phase 1 的数值一致性验证结果，(3) 是否满足 PRD 性能目标，(4) 推荐进入 Phase 3 的配置

---

## 需求 6: 数值一致性验证

### 用户故事
作为算法开发者，我需要验证 Triton 实现与 Phase 1 Python 原型的数值一致性。

### 验收标准

- AC-6.1: WHEN 对相同输入执行压缩，Triton 和 NumPy 的压缩结果 SHALL 余弦相似度 ≥ 0.999
- AC-6.2: WHEN 对相同输入执行注意力计算，Triton 和 NumPy 的输出 SHALL 余弦相似度 ≥ 0.99
- AC-6.3: WHEN 运行 Phase 1 的全部属性测试（P1~P8）的 GPU 版本，SHALL 全部通过

---

## 正确性属性候选

| 属性 | 关联需求 | 描述 |
|------|---------|------|
| GPU-CPU 数值一致性 | AC-1.2, AC-2.2, AC-6.1 | Triton 实现与 NumPy 原型数值一致 |
| 压缩-解压往返 (GPU) | AC-1.1, AC-2.1 | GPU 上压缩后解压的向量与原始向量高度相似 |
| 融合注意力等价性 | AC-4.2 | 融合 kernel 与分步计算结果一致 |
| 零向量安全 (GPU) | AC-1.4 | GPU 上零向量输入不产生 NaN/Inf |
| 性能下界 | AC-1.3, AC-4.6, AC-4.8 | 压缩延迟和注意力加速满足目标 |
| 增量追加一致性 | AC-1.6 | 追加压缩与全量压缩结果一致 |

---

## 约束

- 使用 Triton 编写 GPU kernel，不直接写 CUDA C++
- 所有 kernel 输入输出为 PyTorch tensor（GPU）
- 依赖：torch >= 2.0, triton >= 3.0
- 代码结构：src/polarquant_kv_cuda/（与 Phase 1 的 python/polarquant_kv/ 并列）
- 保持与 Phase 1 Python API 相同的函数签名
- Triton 实现的 Python API SHALL 与未来可能的 Raw CUDA + pybind11 实现保持相同的函数签名和 tensor 格式
- Triton JIT 缓存目录 SHALL 可通过环境变量配置，不硬编码路径
