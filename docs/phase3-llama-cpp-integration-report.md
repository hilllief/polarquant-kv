# Phase 3: PolarQuant-KV llama.cpp 集成报告

## 完成状态

22/24 任务完成。CUDA kernel 单元测试 534/534 通过，llama.cpp 完整编译 214/214 targets 成功，端到端 Qwen3.5-9B 生成文本验证通过。

## 集成架构

```
llama.cpp 推理流程:
  Tokenize → Embedding → For each layer:
    ├─ Compute Q, K, V
    ├─ cpy_k/cpy_v → 标准 KV Cache（ggml tensor）
    ├─ ★ process_ubatch hook → shadow compression → PolarQuant buffers
    ├─ build_attn_mha → 标准注意力（当前路径）
    ├─ ★ [未来] → polarquant_fused_attention → 从压缩格式直接计算
    └─ FFN
  → Output logits
```

## 新增文件

| 文件 | 说明 |
|------|------|
| `ggml/src/ggml-cuda/polarquant.cuh` | CUDA kernel 头文件、数据结构、Lloyd-Max codebook |
| `ggml/src/ggml-cuda/polarquant.cu` | 4 个 CUDA kernel + 状态管理 + extern "C" 导出 |
| `src/llama-polarquant.h` | 集成层 API |
| `src/llama-polarquant.cpp` | 动态加载 CUDA 函数、初始化、内存管理 |
| `tests/test-polarquant.cu` | 4 个测试用例，534 个断言 |

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `ggml/CMakeLists.txt` | `GGML_POLARQUANT` 选项 |
| `ggml/src/ggml-cuda/CMakeLists.txt` | 编译定义 |
| `CMakeLists.txt` | 全局传播 `GGML_POLARQUANT` |
| `include/llama.h` | context params 字段 |
| `common/common.h` | CLI params 字段 |
| `common/common.cpp` | 参数传递 |
| `common/arg.cpp` | `--polarquant` / `--pq` 参数 |
| `src/llama-cparams.h` | cparams polarquant 字段 |
| `src/llama-context.h` | `pq_state` 成员 |
| `src/llama-context.cpp` | 初始化/销毁/默认值/graph hook |
| `src/llama-graph.cpp` | PolarQuant include |
| `src/CMakeLists.txt` | 源文件列表 |

## 测试结果

### CUDA Kernel 单元测试
```
GPU: NVIDIA GeForce RTX 5060 Ti (sm_120, 16310 MB)
Test 1: Rotation invertibility...        PASSED
Test 2: Compress kernel correctness...   PASSED (7.53x compression)
Test 3: Fused attention kernel...        PASSED (GQA 2:1)
Test 4: Zero vector handling...          PASSED
Results: 534/534 tests passed
```

### 端到端测试（Qwen3.5-9B.Q4_K_M.gguf）
```
Context: 4096 tokens
PolarQuant: 132 MB compressed vs 1024 MB original (7.8x ratio)
Generated: "The user has started introducing themselves with"
Prompt: 120.7 tok/s | Generation: 50.3 tok/s
```

## 关键决策

- [AD-1] Codebook 硬编码：避免启动时计算开销
- [AD-2] ggml-cuda 层实现：纯 GPU 操作，不需要 CPU fallback
- [AD-3] 融合 kernel：减少 kernel launch 和中间内存
- [AD-4] `#ifdef GGML_POLARQUANT` 条件编译：零侵入，默认 OFF
- [AD-5] 动态加载 CUDA 函数：解决 llama.dll 和 ggml-cuda.dll 跨 DLL 链接问题
- [AD-6] Shadow compression：在 process_ubatch 后压缩，不修改 ggml 计算图

## 剩余工作

- Task 3.4: 实现 ggml tensor 数据提取 + 实际压缩调用
- Task 3.5: 添加 GGML_OP_POLARQUANT_ATTN 替换 build_attn_mha
