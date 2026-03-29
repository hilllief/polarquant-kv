# PolarQuant-KV Phase 1: 测试任务（第一轮 Spec）

## 引用

- 需求文档: #[[file:docs/specs/phase1-python-prototype/requirements.md]]
- 设计文档: #[[file:docs/specs/phase1-python-prototype/design.md]]

---

## Task 1: 项目骨架与测试基础设施 [x] 

### 描述
创建项目目录结构、pyproject.toml、测试基础设施文件（conftest.py、factories.py、strategies.py）。

### 文件
- `pyproject.toml` — 项目配置与依赖
- `python/polarquant_kv/__init__.py` — 包初始化（空）
- `python/polarquant_kv/types.py` — CompressedKV、QJLSignatures、CompressedKVCache 数据类
- `tests/__init__.py`
- `tests/conftest.py` — 共享 fixtures（rotation_matrix, jl_matrix, rng 等）
- `tests/factories.py` — 测试数据工厂函数
- `tests/strategies.py` — hypothesis 自定义策略

---

## Task 2: 需求 1 单元测试 — 正交旋转矩阵 [x]

### 描述
为 rotation.py 编写单元测试，覆盖 AC-1.1 ~ AC-1.5。

### 文件
- `tests/test_rotation.py`

### 验收标准覆盖
- AC-1.1: 正交性验证（R^T·R = I）
- AC-1.2: 范数保持验证
- AC-1.3: 种子可复现性
- AC-1.4: 无效维度 ValueError
- AC-1.5: 高维（d=4096）正交性

---

## Task 3: 需求 2 单元测试 — PolarQuant 量化 [x]

### 描述
为 quantizer.py 编写单元测试，覆盖 AC-2.1 ~ AC-2.10。

### 文件
- `tests/test_quantizer.py`

### 验收标准覆盖
- AC-2.1: 完整量化流程
- AC-2.2: 4-bit 压缩比 ≥ 3.8x
- AC-2.3: 3-bit 压缩比 ≥ 5.0x
- AC-2.4: 余弦相似度阈值（多 bit 档位）
- AC-2.5: padding 处理
- AC-2.6: n_bits 范围校验
- AC-2.7: group_size 范围校验
- AC-2.8: 零向量处理
- AC-2.9: CompressedKV 结构化返回
- AC-2.10: 极端值处理

---

## Task 4: 需求 3 单元测试 — QJL 误差修正 [x]

### 描述
为 qjl.py 编写单元测试，覆盖 AC-3.1 ~ AC-3.6。

### 文件
- `tests/test_qjl.py`

### 验收标准覆盖
- AC-3.1: JL 矩阵分布验证
- AC-3.2: 投影 + 符号位提取
- AC-3.3: 修正有效性（统计显著性）
- AC-3.4: jl_dim 校验
- AC-3.5: 零残差处理
- AC-3.6: 种子可复现性

---

## Task 5: 需求 4 单元测试 — 压缩注意力计算 [x]

### 描述
为 attention.py 编写单元测试，覆盖 AC-4.1 ~ AC-4.7。

### 文件
- `tests/test_attention.py`

### 验收标准覆盖
- AC-4.1: 压缩注意力分数计算
- AC-4.2: QJL 修正后最大绝对误差
- AC-4.3: 完整注意力输出
- AC-4.4: 注意力输出余弦相似度
- AC-4.5: 空序列处理
- AC-4.6: GQA 支持
- AC-4.7: enable_qjl 开关

---

## Task 6: 需求 5 单元测试 — 超参数搜索 [x]

### 描述
为 benchmark.py 编写单元测试，覆盖 AC-5.1 ~ AC-5.7。

### 文件
- `tests/test_benchmark.py`

### 验收标准覆盖
- AC-5.1: 超参数组合遍历
- AC-5.2: 存储字节数精确计算
- AC-5.3: 精度指标报告
- AC-5.4: 帕累托最优推荐
- AC-5.5: 压缩比公式验证
- AC-5.6: 进度日志输出
- AC-5.7: Phase 1 完成判定报告

---

## Task 7: 需求 6 单元测试 — Batch 操作 [x]

### 描述
为 batch 操作编写单元测试，覆盖 AC-6.1 ~ AC-6.4。

### 文件
- `tests/test_batch.py`

### 验收标准覆盖
- AC-6.1: batch 压缩（4D 输入）
- AC-6.2: batch vs 逐个一致性
- AC-6.3: 空 batch/空序列
- AC-6.4: 内存预估警告

---

## Task 8: 属性测试 — P1~P8 [x]

### 描述
为 8 个正确性属性编写 hypothesis 属性测试。

### 文件
- `tests/test_properties.py`

### 属性覆盖
- P1: 正交性保持
- P2: 压缩-解压往返
- P3: 压缩比下界
- P4: QJL 修正有效性
- P5: 注意力等价性
- P6: Batch 一致性
- P7: 零向量安全
- P8: 极端值鲁棒性

---

## Task 9: E2E 测试 — 端到端业务流程 [x]

### 描述
测试完整的业务流程：生成 KV → 压缩 → 压缩注意力计算 → 精度验证。

### 文件
- `tests/test_e2e.py`

### 场景
- 单 head 单 token 完整流程
- 多 head 多 token batch 流程
- GQA 场景完整流程
- 超参数搜索 + 报告生成完整流程
