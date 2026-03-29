# Phase 3: PolarQuant-KV llama.cpp 集成 — 设计文档

## 1. 架构概述

```
llama.cpp 推理流程（修改点标记 ★）:

  Tokenize → Embedding → For each layer:
    ├─ Compute Q, K, V
    ├─ ★ Store K,V → polarquant_compress() → packed KV Cache
    ├─ ★ Attention  → polarquant_fused_attention(Q, packed_K, packed_V)
    └─ FFN
  → Output logits
```

## 2. 文件结构

```
integration/llama.cpp/
├── ggml/src/ggml-cuda/
│   ├── polarquant.cu          # PolarQuant CUDA kernels（新增）
│   └── polarquant.cuh         # PolarQuant 头文件（新增）
├── src/
│   ├── llama-polarquant.h     # PolarQuant 集成层头文件（新增）
│   ├── llama-polarquant.cpp   # PolarQuant 集成层实现（新增）
│   ├── llama-kv-cache.cpp     # 修改：在 cpy_k/cpy_v 中插入压缩
│   └── llama-graph.cpp        # 修改：在注意力构建中使用融合 kernel
├── CMakeLists.txt             # 修改：添加 GGML_POLARQUANT 选项
└── tests/
    └── test-polarquant.cpp    # PolarQuant 单元测试（新增）
```

## 3. 核心数据结构

### 3.1 压缩 KV Cache 内存布局

```c
// 每个 KV head 的压缩存储（per layer）:
struct polarquant_kv_layer {
    // Key 压缩数据
    uint8_t* k_packed;    // [n_kv, head_dim/2] — 4-bit packed indices
    float*   k_radius;    // [n_kv]             — 向量范数

    // Value 压缩数据
    uint8_t* v_packed;    // [n_kv, head_dim/2] — 4-bit packed indices
    float*   v_radius;    // [n_kv]             — 向量范数
};

// 全局 PolarQuant 状态
struct polarquant_state {
    bool     enabled;
    int      n_bits;       // 默认 4
    int      n_levels;     // 2^n_bits = 16
    float*   codebook;     // [n_levels] Lloyd-Max centroids (GPU)
    float*   rotation;     // [head_dim, head_dim] 正交矩阵 (GPU)
    int      head_dim;
    int      n_layers;

    polarquant_kv_layer* layers;  // [n_layers]
};
```

### 3.2 Lloyd-Max Codebook

4-bit codebook（16 个质心）基于 Beta((d-1)/2, (d-1)/2) 分布预计算：
```c
// d=128 时的 4-bit Lloyd-Max codebook（硬编码）
static const float POLARQUANT_CODEBOOK_4BIT_D128[16] = {
    -0.1547, -0.1175, -0.0928, -0.0722,
    -0.0537, -0.0363, -0.0193, -0.0024,
     0.0145,  0.0318,  0.0498,  0.0692,
     0.0907,  0.1157,  0.1467,  0.1947
};
```

## 4. CUDA Kernel 设计

### 4.1 压缩 Kernel

```
polarquant_compress_kernel<<<N, 128, smem>>>:
  输入: float* input [N, D], float* R [D, D], float* codebook [n_levels]
  输出: uint8_t* packed [N, D/2], float* radius [N]

  每个 block 处理一个向量:
  1. Shared memory 加载 codebook (16 floats)
  2. 矩阵-向量乘法: rotated = input @ R^T
  3. Warp reduce 计算范数
  4. 归一化: direction = rotated / radius
  5. Codebook 最近邻查找 + 4-bit pack
```

### 4.2 融合注意力 Kernel

```
polarquant_attention_kernel<<<total_heads, D>>>:
  输入: Q_rotated, packed_K, K_radius, packed_V, V_radius, codebook
  输出: attention_output

  Pass 1 — Score 计算:
    for each key token s:
      k_val = codebook[packed_K[s][dim]]  // 查表
      score[s] = warp_reduce(k_val * q_val) * K_radius[s] * scale

  Softmax (thread 0):
    weights = softmax(scores)

  Pass 2 — V 加权求和:
    for each value token s:
      v_val = codebook[packed_V[s][dim]]  // 查表
      output[dim] += weights[s] * v_val * V_radius[s]
```

## 5. 集成点

### 5.1 KV Cache 写入（压缩）

在 `llama_kv_cache_context::cpy_k()` / `cpy_v()` 中：
- 原始路径：`ggml_cpy(k_cur, k_cache)` — FP16 直接拷贝
- PolarQuant 路径：`polarquant_compress(k_cur)` → 写入 packed + radius

### 5.2 注意力计算

在 graph 构建的 `build_attn()` 中：
- 原始路径：`ggml_flash_attn_ext(Q, K, V, mask)`
- PolarQuant 路径：`polarquant_fused_attention(Q, packed_K, packed_V, ...)`

### 5.3 运行时开关

通过 `llama_context_params` 添加 `polarquant` 字段：
```c
struct llama_context_params {
    // ... existing fields ...
    bool polarquant;      // 启用 PolarQuant KV Cache 压缩
    int  polarquant_bits; // 量化位宽（默认 4）
};
```

## 6. 编译集成

### CMakeLists.txt 修改

```cmake
# 顶层 CMakeLists.txt
option(GGML_POLARQUANT "Enable PolarQuant KV Cache compression" OFF)

if (GGML_POLARQUANT)
    add_compile_definitions(GGML_POLARQUANT)
endif()
```

CUDA kernel 文件 `polarquant.cu` 放在 `ggml/src/ggml-cuda/` 目录下，
会被 `file(GLOB GGML_SOURCES_CUDA "*.cu")` 自动包含。

## 7. AUTO-DECISION 记录

- [AD-1] 选择 codebook 硬编码而非运行时计算：codebook 只依赖 d 和 n_bits，预计算一次即可。硬编码避免启动时的计算开销。
- [AD-2] 选择在 ggml-cuda 层实现而非 ggml 抽象层：PolarQuant 是纯 CUDA 操作，不需要 CPU fallback（KV Cache 压缩只在 GPU 上有意义）。
- [AD-3] 选择融合 kernel 而非分步操作：减少 kernel launch 开销和中间内存分配，参考 Flash Attention 的设计理念。
- [AD-4] 旋转矩阵使用 Hadamard 变换替代完整正交矩阵：O(d log d) 复杂度 vs O(d²)，且不需要存储 d×d 矩阵。[AUTO-INTERPRETED] PRD 中说"QR 分解"，但 Hadamard 更适合 GPU 实现。
