# PolarQuant-KV Phase 1: 实现任务（第二轮 Spec）

## 目标

实现所有模块代码，使第一轮 Spec 的全部测试通过（红灯 → 绿灯）。

## 引用

- 需求文档: #[[file:docs/specs/phase1-python-prototype/requirements.md]]
- 设计文档: #[[file:docs/specs/phase1-python-prototype/design.md]]
- 测试任务: #[[file:docs/specs/phase1-python-prototype/tasks.md]]

---

- [x] 1. 实现 utils.py 工具函数
  - [x] 1.1 实现 cosine_similarity
  - [x] 1.2 实现 compute_compression_ratio（AC-5.5 公式）
  - [x] 1.3 实现 attention_score_mse
  - [x] 1.4 实现 estimate_memory_bytes（AC-6.4）

- [x] 2. 实现 rotation.py 正交旋转矩阵（需求 1）
  - [x] 2.1 实现 generate_rotation_matrix（QR 分解，种子控制，参数校验）
  - [x] 2.2 实现 rotate 和 inverse_rotate
  - [x] 2.3 跑 test_rotation.py 验证绿灯 ✅ 16 passed

- [x] 3. 实现 quantizer.py PolarQuant 量化（需求 2）
  - [x] 3.1 实现 compress（旋转 → 半径分离 → 分组量化，含零向量/极端值处理）
  - [x] 3.2 实现 decompress（反量化 → 半径恢复 → 逆旋转）
  - [x] 3.3 跑 test_quantizer.py 验证绿灯 ✅ 21 passed

- [x] 4. 实现 qjl.py QJL 误差修正（需求 3）
  - [x] 4.1 实现 generate_jl_matrix（N(0,1/m) 分布，种子控制）
  - [x] 4.2 实现 compute_signatures（JL 投影 + 符号位提取）
  - [x] 4.3 实现 compute_correction（注意力分数修正量）
  - [x] 4.4 跑 test_qjl.py 验证绿灯 ✅ 12 passed

- [x] 5. 实现 attention.py 压缩注意力计算（需求 4）
  - [x] 5.1 实现 standard_attention（含 GQA 支持）
  - [x] 5.2 实现 compressed_attention（含 enable_qjl 开关、return_scores、GQA）
  - [x] 5.3 跑 test_attention.py 验证绿灯 ✅ 9 passed

- [x] 6. 实现 quantizer.py batch 操作（需求 6）
  - [x] 6.1 实现 compress_batch（4D 输入，内存预估警告）
  - [x] 6.2 实现 decompress_batch
  - [x] 6.3 跑 test_batch.py 验证绿灯 ✅ 6 passed

- [x] 7. 实现 benchmark.py 超参数搜索（需求 5）
  - [x] 7.1 实现 evaluate_config
  - [x] 7.2 实现 hyperparameter_search（含进度日志）
  - [x] 7.3 实现 generate_phase1_report（Phase 1 完成判定）
  - [x] 7.4 跑 test_benchmark.py 验证绿灯 ✅ 16 passed

- [x] 8. 全量测试验证
  - [x] 8.1 跑 test_properties.py 属性测试验证绿灯 ✅ 10 passed
  - [x] 8.2 跑 test_e2e.py E2E 测试验证绿灯 ✅ 5 passed
  - [x] 8.3 跑全量 pytest 确认所有测试通过 ✅ 95 passed
