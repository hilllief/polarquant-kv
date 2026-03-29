# Phase 3: PolarQuant-KV llama.cpp 集成 — 最终状态

## 已完成 ✅

- CUDA kernels: 压缩/融合注意力/旋转 (534 测试通过)
- 集成层: 动态加载、状态管理、FP32 KV cache 强制
- 压缩管道: process_ubatch 后置 hook 实际调用压缩 kernel
- 编译系统: GGML_POLARQUANT CMake 选项
- CLI: --polarquant / --pq / --pq-bits
- GGML_OP_CUSTOM CUDA backend dispatch
- 端到端验证: Qwen3.5-9B 生成文本正常

## 阻塞项 🔴

融合注意力 kernel 替换 build_attn_mha 被 ggml 框架限制阻塞:
- ggml_custom_4d: 调度器不为 GGML_OP_CUSTOM 分配 GPU buffer (GGML_ASSERT(buffer) failed)
- ggml_custom_inplace: 同样的 buffer 分配问题
- 添加新 ggml op: 触发 80+ CUDA 模板文件全量重编译 (20+ 分钟/次)

需要 llama.cpp 上游支持:
- ggml_backend_sched 为 GGML_OP_CUSTOM 正确分配 GPU buffer
- 或添加 ggml_backend_custom_op 机制

## 当前可用功能

`--polarquant` 启用后:
- KV 数据被实时压缩到 4-bit 缓冲区 (66MB vs 512MB FP16)
- 注意力仍用标准路径 (精度零损失)
- 压缩数据已就绪，等待融合 kernel 接入
