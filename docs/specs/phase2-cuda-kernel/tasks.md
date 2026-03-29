# PolarQuant-KV Phase 2: 测试任务（第一轮 Spec）

## 引用

- 需求文档: #[[file:docs/specs/phase2-cuda-kernel/requirements.md]]
- 设计文档: #[[file:docs/specs/phase2-cuda-kernel/design.md]]

---

## Task 1: 项目骨架与测试基础设施 [ ]

### 文件
- `src/polarquant_kv_cuda/__init__.py`
- `src/polarquant_kv_cuda/types.py` — GPU 数据结构
- `src/polarquant_kv_cuda/rotation.py` — 旋转矩阵（PyTorch）桩
- `src/polarquant_kv_cuda/compress_kernel.py` — 压缩 kernel 桩
- `src/polarquant_kv_cuda/decompress_kernel.py` — 解压 kernel 桩
- `src/polarquant_kv_cuda/qjl_kernel.py` — QJL kernel 桩
- `src/polarquant_kv_cuda/attention_kernel.py` — 注意力 kernel 桩
- `src/polarquant_kv_cuda/compressor.py` — 高层 API 桩
- `src/polarquant_kv_cuda/utils.py` — 工具函数桩
- `tests/cuda/__init__.py`
- `tests/cuda/conftest.py` — GPU fixtures

## Task 2: 需求 1 & 2 单元测试 — 压缩/解压 Kernel [ ]

### 文件
- `tests/cuda/test_compress_decompress.py`

### 覆盖
- AC-1.1 ~ AC-1.5, AC-1.8, AC-1.9
- AC-2.1 ~ AC-2.3

## Task 3: 需求 3 单元测试 — QJL Kernel [ ]

### 文件
- `tests/cuda/test_qjl_kernel.py`

### 覆盖
- AC-3.1 ~ AC-3.3

## Task 4: 需求 4 单元测试 — 融合注意力 [ ]

### 文件
- `tests/cuda/test_attention_kernel.py`

### 覆盖
- AC-4.1 ~ AC-4.5, AC-4.7

## Task 5: 需求 6 单元测试 — 数值一致性 [ ]

### 文件
- `tests/cuda/test_consistency.py`

### 覆盖
- AC-6.1 ~ AC-6.3

## Task 6: 需求 1 增量追加测试 [ ]

### 文件
- `tests/cuda/test_incremental.py`

### 覆盖
- AC-1.6

## Task 7: 属性测试 — P1~P6 GPU 版 [ ]

### 文件
- `tests/cuda/test_gpu_properties.py`

### 覆盖
- P1 GPU-CPU 一致性
- P2 压缩-解压往返 GPU
- P3 融合注意力等价性
- P4 零向量安全 GPU
- P5 增量追加一致性
- P6 Bit Packing 往返

## Task 8: 需求 5 性能基准测试 [ ]

### 文件
- `tests/cuda/test_benchmark.py`

### 覆盖
- AC-5.1 ~ AC-5.8
- AC-4.6, AC-4.8（性能目标）
- AC-1.3（压缩延迟）
