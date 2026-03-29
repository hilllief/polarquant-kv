# PolarQuant-KV Phase 3: 推理集成 Wrapper — 需求文档

## 概述

用 PyTorch 实现完整的 Transformer 推理 Wrapper，将 PolarQuant-KV 集成到多头注意力的 KV Cache 管理中，验证在真实推理流程（prefill + decode）中的精度和显存节省效果。

## 引用

- PRD: #[[file:docs/polarquant-kvcache-prd.md]]
- Phase 2 报告: #[[file:docs/phase2-report.md]]

---

## 需求 1: KV Cache Manager

### 用户故事
作为推理框架开发者，我需要一个 KV Cache 管理器，自动对写入的 KV 向量进行压缩存储，读取时自动解压。

### 验收标准

- AC-1.1: WHEN 初始化 KV Cache Manager，SHALL 接受 max_seq_len、num_heads、head_dim、n_bits、group_size 参数，预分配 GPU 显存
- AC-1.2: WHEN 写入新 token 的 KV 向量（prefill 或 decode），SHALL 自动压缩并追加到缓存中，更新 seq_len 计数
- AC-1.3: WHEN 读取 KV Cache 用于注意力计算，SHALL 返回压缩的 KV Cache 对象，无需手动解压
- AC-1.4: WHEN KV Cache 已满（seq_len == max_seq_len），SHALL 抛出 RuntimeError 提示缓存已满
- AC-1.5: WHEN 查询当前状态，SHALL 返回 seq_len、显存占用字节数、压缩比

---

## 需求 2: Transformer 注意力层 Wrapper

### 用户故事
作为推理框架开发者，我需要一个可替换标准注意力层的 Wrapper，内部使用压缩 KV Cache。

### 验收标准

- AC-2.1: WHEN 替换标准 MultiHeadAttention，SHALL 提供相同的 forward 接口：输入 (query, key, value)，输出注意力结果
- AC-2.2: WHEN 执行 prefill（一次写入多个 token），SHALL 批量压缩所有 KV 并存入缓存
- AC-2.3: WHEN 执行 decode（逐 token 生成），SHALL 每次只压缩 1 个新 token 的 KV，与历史缓存拼接计算注意力
- AC-2.4: WHEN 与标准注意力对比，SHALL 注意力输出的余弦相似度 ≥ 0.98（4-bit, d=128）
- AC-2.5: WHEN 支持 GQA，SHALL 正确处理 num_kv_heads < num_q_heads

---

## 需求 3: 端到端推理流程

### 用户故事
作为算法开发者，我需要验证 PolarQuant-KV 在完整的 Transformer 推理流程中的效果。

### 验收标准

- AC-3.1: WHEN 构建一个简单的多层 Transformer 模型（≥ 2 层），SHALL 能用压缩 KV Cache 完成 prefill + decode 推理
- AC-3.2: WHEN 对比标准推理和压缩推理的输出 logits，SHALL 余弦相似度 ≥ 0.95（多层累积误差）
- AC-3.3: WHEN 对比显存占用，SHALL 压缩推理的 KV Cache 显存 < 标准推理的 50%（4-bit）
- AC-3.4: WHEN 运行 decode 循环生成 N 个 token，SHALL 每步的 KV Cache 正确增长，不出现显存泄漏

---

## 需求 4: 配置与开关

### 用户故事
作为用户，我需要能方便地启用/禁用 KV Cache 压缩，对比效果。

### 验收标准

- AC-4.1: WHEN 设置 enable_compression=False，SHALL 退化为标准注意力，不做任何压缩
- AC-4.2: WHEN 运行时切换 n_bits 配置，SHALL 在下次 prefill 时生效
- AC-4.3: WHEN 提供 PolarQuantConfig dataclass，SHALL 包含所有可配置参数（n_bits, group_size, jl_dim, enable_qjl, enable_compression）

---

## 正确性属性候选

| 属性 | 关联需求 | 描述 |
|------|---------|------|
| Prefill-Decode 一致性 | AC-2.2, AC-2.3 | prefill 写入的缓存与 decode 逐步写入的缓存等价 |
| 多层精度累积 | AC-3.2 | 多层 Transformer 的累积误差在可控范围内 |
| 显存节省 | AC-3.3 | 压缩 KV Cache 显存显著低于标准 |
| 开关等价性 | AC-4.1 | 禁用压缩时与标准注意力完全一致 |

---

## 约束

- 使用 PyTorch 实现，不依赖 llama.cpp
- 模型结构模拟真实 Transformer（含 RoPE、GQA 等），但参数随机初始化
- 代码结构：src/polarquant_kv_cuda/inference/
