# Phase 3: PolarQuant-KV llama.cpp 集成 — 需求文档

## 1. 概述

将 PolarQuant-KV 压缩引擎集成到 llama.cpp 推理框架中，使 GGUF 模型在推理时自动压缩 KV Cache，降低显存占用 4x+，同时保持零精度损失。

## 2. 功能需求

### FR-1: PolarQuant CUDA Kernel（ggml-cuda 层）
- FR-1.1: 压缩 kernel — 输入 FP16/FP32 KV 向量，输出 packed 4-bit indices + radius
- FR-1.2: 融合注意力 kernel — 直接从压缩格式计算 Q·K^T 和 softmax·V
- FR-1.3: Lloyd-Max codebook 预计算，存储在 constant/shared memory
- FR-1.4: 支持 GQA（Grouped Query Attention）

### FR-2: KV Cache 集成
- FR-2.1: 在 `cpy_k`/`cpy_v` 路径插入压缩操作
- FR-2.2: 在 `get_k`/`get_v` 路径支持压缩格式读取
- FR-2.3: 压缩 KV Cache 的内存布局：`packed_indices[n_kv, head_dim/2]` + `radius[n_kv]`
- FR-2.4: 保持与非压缩模式的 API 兼容

### FR-3: 编译与运行时配置
- FR-3.1: CMake 选项 `GGML_POLARQUANT=ON/OFF`（默认 OFF）
- FR-3.2: 运行时开关 `--polarquant` 或 `--pq` 命令行参数
- FR-3.3: 可配置参数：n_bits（默认 4）、codebook 类型

### FR-4: 端到端兼容性
- FR-4.1: 支持现有 GGUF 模型格式（无需重新量化）
- FR-4.2: 支持 Qwen 系列模型（Qwen2.5/Qwen3.5）
- FR-4.3: 支持 LLaMA 系列模型

## 3. 非功能需求

### NFR-1: 性能
- KV Cache 显存降低 ≥ 4x（FP16 → 4-bit）
- 压缩延迟 < 10μs/token/head
- 注意力计算不慢于标准 Flash Attention 的 2x

### NFR-2: 精度
- Token 匹配率 ≥ 99%（与未压缩推理对比）

### NFR-3: 硬件兼容性
- 支持 sm_86（Ampere）、sm_89（Ada）、sm_120（Blackwell）

### NFR-4: 构建兼容性
- Windows MSVC + CUDA 13.2
- Linux GCC + CUDA 12.x

## 4. 约束

- 最小侵入原则：尽量减少对 llama.cpp 现有代码的修改
- 所有新增代码通过 `#ifdef GGML_POLARQUANT` 条件编译
- 不修改 GGUF 文件格式

## 5. 验收标准

- AC-1: `cmake -DGGML_POLARQUANT=ON` 编译通过
- AC-2: `llama-cli --polarquant -m model.gguf -p "Hello"` 正常生成文本
- AC-3: KV Cache 显存占用降低 ≥ 4x（通过日志输出验证）
- AC-4: 生成文本与未压缩模式一致（前 32 token 匹配率 ≥ 99%）
