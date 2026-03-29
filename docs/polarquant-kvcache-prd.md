# PolarQuant-KV: LLM KV Cache 极坐标量化压缩引擎

## 项目需求文档（PRD）

---

## 1. 项目概述

### 1.1 项目名称
PolarQuant-KV（极坐标量化 KV Cache 压缩引擎）

### 1.2 项目定位
一个独立的 LLM 推理加速库，通过 PolarQuant + QJL 算法对 Transformer 的 KV Cache 进行实时压缩，在零精度损失的前提下将 KV Cache 内存降低 4-6 倍，推理速度提升 2-4 倍。

### 1.3 背景与动机
- KV Cache 是长上下文 LLM 推理的核心内存瓶颈。以 Qwen3.5-9B 为例，4096 上下文的 KV Cache 约占 1.5GB 显存，128K 上下文则需要 48GB
- Google TurboQuant（ICLR 2026）证明了极坐标量化 + JL 误差修正可以实现 6 倍压缩、零精度损失，但未开源实现
- 本项目旨在独立实现该算法，并集成到 llama.cpp 推理引擎中，使消费级 GPU 能够处理更长的上下文

### 1.4 与 DS-B2 的关系
PolarQuant-KV 是 DS-B2 架构的底层加速组件。DS-B2 的 System-A（应用脑）使用 llama.cpp 进行推理，PolarQuant-KV 集成到 llama.cpp 后，DS-B2 可直接受益：
- 更长的上下文窗口 → HMS 分段策略触发频率降低
- 更低的显存占用 → 同一张卡可跑更大的模型
- 更快的注意力计算 → DS-B2 整体延迟降低

---

## 2. 目标用户

| 用户类型 | 需求 |
|---------|------|
| 本地 LLM 部署者 | 在消费级 GPU（16-24GB）上跑长上下文推理 |
| 推理框架开发者 | 集成到 llama.cpp / vLLM / TensorRT-LLM |
| DS-B2 用户 | 降低 System-A 的显存和延迟开销 |
| 研究者 | 复现和改进 TurboQuant 论文 |

---

## 3. 核心功能需求

### 3.1 PolarQuant 极坐标量化模块

#### 功能描述
将 KV Cache 中的 FP16/FP32 向量通过极坐标变换 + 分组量化压缩到 4-8 bit。

#### 算法流程
```
输入: KV 向量 v ∈ R^d (FP16, d=128 per head)

Step 1: 随机旋转
  v' = R · v
  R: 预生成的 d×d 正交矩阵（QR 分解）
  目的: 消除 channel 间方差差异，使量化误差均匀分布

Step 2: 半径-方向分离
  r = ||v'||          (FP16, 2 字节)
  d = v' / r          (单位方向向量)

Step 3: 分组量化
  每 group_size 维一组（默认 32）
  每组独立计算 min/max
  量化到 n_bits（默认 4）
  存储: quantized_values (uint4) + group_min/scale (FP16)

输出: CompressedKV {radius, quantized_direction, group_params}
压缩比: FP16 → 4bit ≈ 4x; FP16 → 3bit ≈ 5.3x
```

#### 技术要求
- CUDA kernel 实现，支持 FP16 输入
- 压缩/解压必须在 GPU 上完成（不能走 CPU）
- 单次压缩延迟 < 10μs（per head, per token）
- 支持 batch 操作

### 3.2 QJL 误差修正模块

#### 功能描述
对 PolarQuant 的量化残差进行 Johnson-Lindenstrauss 投影，仅保留符号位，以极低开销修正注意力分数误差。

#### 算法流程
```
输入: 量化残差 e = v - v_quantized

Step 1: JL 随机投影
  e_proj = P · e
  P: m×d 随机投影矩阵 (m << d, 默认 m=64)
  P 的每个元素 ~ N(0, 1/m)

Step 2: 取符号位
  signs = sign(e_proj)    (每个值 1 bit)

Step 3: 注意力修正（在计算 attention score 时）
  score_corrected = score_quantized + correction_term(signs)

存储开销: m bits per token per head (64 bits = 8 bytes)
```

#### 技术要求
- 与 PolarQuant 共享 CUDA kernel（融合操作）
- JL 矩阵预生成，存储在 GPU constant memory
- 符号位用 bit packing 存储（64 个符号位 = 8 字节）

### 3.3 注意力计算适配

#### 功能描述
修改注意力计算逻辑，直接在压缩的 KV Cache 上计算注意力分数，避免先解压再计算的开销。

#### 技术要求
```
标准注意力: score = Q · K^T / sqrt(d)
压缩注意力: score = Q · decompress(K_compressed)^T / sqrt(d)
融合注意力: score = compressed_attention(Q, K_compressed) / sqrt(d)
                    ↑ 在一个 kernel 里完成解压+点积
```
- 实现融合 kernel：解压 + 矩阵乘法在同一个 CUDA kernel 中完成
- 支持 Flash Attention 风格的分块计算
- 支持 GQA（Grouped Query Attention）

### 3.4 llama.cpp 集成

#### 功能描述
将 PolarQuant-KV 集成到 llama.cpp 的推理流程中。

#### 集成点
```
llama.cpp 推理流程:
  1. Tokenize
  2. Embedding lookup
  3. For each layer:
     a. Compute Q, K, V from input
     b. Store K, V to KV Cache        ← 这里插入压缩
     c. Compute attention(Q, KV Cache) ← 这里用压缩版注意力
     d. FFN
  4. Output logits
```

#### 技术要求
- 修改 `ggml-cuda.cu` 中的 KV Cache 写入逻辑
- 修改 `ggml-cuda.cu` 中的注意力计算逻辑
- 新增编译选项 `GGML_POLARQUANT=ON`
- 保持与现有 GGUF 模型格式的兼容性（不需要重新量化模型）
- 提供开关：用户可选择启用/禁用 KV Cache 压缩

---

## 4. 非功能需求

### 4.1 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| KV Cache 压缩比 | ≥ 4x | FP16 → 4bit |
| 精度损失 | ≈ 0 | LongBench 等基准上无可测量的精度下降 |
| 压缩延迟 | < 10μs/token/head | 不能成为推理瓶颈 |
| 注意力计算加速 | ≥ 2x | 压缩后数据量更小，内存带宽需求降低 |
| 端到端推理加速 | ≥ 1.5x | 注意力只是推理的一部分 |

### 4.2 硬件兼容性

| GPU 架构 | 支持 | 说明 |
|---------|------|------|
| Ampere (sm_86) | 必须 | RTX 3090, A100 |
| Ada Lovelace (sm_89) | 必须 | RTX 4090 |
| Blackwell (sm_100) | 必须 | RTX 5060 Ti, RTX 5090 |
| Hopper (sm_90) | 可选 | H100 |

### 4.3 软件兼容性

| 组件 | 版本要求 |
|------|---------|
| CUDA Toolkit | ≥ 12.0 |
| llama.cpp | 最新 master 分支 |
| Python 绑定 | llama-cpp-python ≥ 0.3.x |

---

## 5. 技术架构

```
polarquant-kv/
├── include/
│   ├── polarquant.h          # C API 头文件
│   └── polarquant_cuda.h     # CUDA kernel 声明
├── src/
│   ├── polarquant.cu         # PolarQuant CUDA kernel
│   ├── qjl.cu               # QJL 误差修正 CUDA kernel
│   ├── attention_fused.cu    # 融合注意力 CUDA kernel
│   ├── polarquant.cpp        # C++ 封装层
│   └── python_binding.cpp    # Python 绑定（pybind11）
├── python/
│   └── polarquant_kv/
│       ├── __init__.py
│       ├── compressor.py     # Python 高层 API
│       └── benchmark.py      # 性能基准测试
├── integration/
│   └── llama_cpp_patch/      # llama.cpp 集成补丁
│       ├── ggml-polarquant.cu
│       └── README.md
├── tests/
│   ├── test_accuracy.py      # 精度测试
│   ├── test_speed.cu         # CUDA 性能测试
│   └── test_integration.py   # 集成测试
├── benchmarks/
│   ├── longbench.py          # LongBench 基准
│   └── memory_profile.py     # 内存分析
├── CMakeLists.txt
├── pyproject.toml
└── README.md
```

---

## 6. 开发计划

### Phase 1: Python 原型验证（2 周）
- 用 NumPy 实现 PolarQuant + QJL 完整算法
- 在 CPU 上验证精度（余弦相似度、注意力分数误差）
- 确定最优超参数（n_bits, group_size, jl_dim）
- 输出：Python 原型 + 精度验证报告

### Phase 2: CUDA Kernel 实现（4 周）
- 实现 PolarQuant 压缩/解压 CUDA kernel
- 实现 QJL 投影 + 符号位存储 CUDA kernel
- 实现融合注意力计算 kernel
- 性能优化（shared memory, warp-level primitives）
- 输出：独立 CUDA 库 + 性能基准

### Phase 3: llama.cpp 集成（3 周）
- 修改 llama.cpp 的 KV Cache 管理逻辑
- 修改注意力计算调用路径
- 添加编译选项和运行时开关
- 端到端测试（GGUF 模型 + 压缩 KV Cache）
- 输出：llama.cpp 补丁 + 集成文档

### Phase 4: 验证与发布（2 周）
- LongBench / Needle in a Haystack 精度验证
- 多 GPU 架构性能测试
- 文档编写
- 开源发布
- 输出：完整项目 + 论文（可选）

### 总计：约 11 周

---

## 7. 前置技能要求

| 技能 | 级别 | 说明 |
|------|------|------|
| CUDA 编程 | 中高级 | 需要写自定义 kernel，理解 shared memory、warp shuffle |
| C++ | 中级 | 修改 llama.cpp 源码 |
| 线性代数 | 中级 | 正交矩阵、JL 变换的数学原理 |
| LLM 推理原理 | 中级 | 理解 KV Cache、注意力机制、GQA |
| Python | 中级 | 原型验证和测试 |

---

## 8. 风险与挑战

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| CUDA kernel 性能不达标 | 压缩开销抵消了内存节省的收益 | 参考 Flash Attention 的优化技巧，使用 Triton 快速迭代 |
| llama.cpp 代码变动频繁 | 集成补丁可能与新版本不兼容 | 以最小侵入方式集成，抽象接口层 |
| 4-bit 精度在大模型上不够 | 70B+ 模型可能出现精度下降 | 提供 4/6/8 bit 多档位选择 |
| Blackwell 架构的 CUDA 特性 | sm_100 可能需要特殊优化 | 先在 Ampere/Ada 上开发，后适配 Blackwell |

---

## 9. 成功标准

| 标准 | 指标 |
|------|------|
| 压缩比 | KV Cache 内存降低 ≥ 4 倍 |
| 精度 | LongBench 分数下降 < 0.5% |
| 速度 | 端到端推理加速 ≥ 1.5 倍 |
| 兼容性 | 支持 Qwen3.5-9B/27B、LLaMA-3、Mistral 等主流模型 |
| 可用性 | pip install 即可使用，提供 llama.cpp 一键补丁 |

---

## 10. 参考资料

1. Google TurboQuant Blog: https://research.google/blog/turboquant/
2. TurboQuant Paper (ICLR 2026): arXiv:2504.19874
3. PolarQuant Paper: arXiv:2502.02617
4. Flash Attention: arXiv:2205.14135
5. llama.cpp 源码: https://github.com/ggml-org/llama.cpp
6. CUDA Programming Guide: https://docs.nvidia.com/cuda/
