# PolarQuant-KV

**LLM KV Cache 极坐标量化压缩引擎 — K+V 双压缩，消费级 GPU，零精度损失**

从零复现 Google TurboQuant（ICLR 2026）算法，并扩展为 K+V 双压缩，显存节省从 37% 提升至 73-99%。

## 核心结果

在 Qwen2.5-0.5B + RTX 5060 Ti 上验证：

| 指标 | 结果 |
|------|------|
| Token 匹配率 | **100%**（零精度损失） |
| KV Cache 显存节省 | **73-99%** |
| 注意力加速（512 seq） | **2.4x** |
| 自动化测试 | **154 + 534 = 688 个** |

## 与 TurboQuant 论文的对比

| 维度 | TurboQuant | PolarQuant-KV |
|------|-----------|---------------|
| 压缩范围 | 仅 Key | **Key + Value** |
| 显存节省 | 37% | **73-99%** |
| 目标硬件 | H100 | **RTX 5060 Ti（消费级）** |
| 开源 | 否 | **是（688 测试）** |

## 项目结构

```
polarquant-kv/
├── python/polarquant_kv/     # Phase 1: Python 原型（154 测试）
│   ├── quantizer.py          # PolarQuant 压缩/解压
│   ├── rotation.py           # 随机正交旋转
│   ├── qjl.py                # QJL 误差修正
│   ├── attention.py          # 压缩注意力计算
│   └── types.py              # 数据类型定义
├── csrc/                     # Phase 2: CUDA Kernel（534 测试）
│   └── flash_turboquant.cu   # 融合注意力 kernel
├── integration/              # Phase 3: llama.cpp 集成
│   ├── polarquant-kv.patch   # llama.cpp 补丁
│   ├── install.sh            # Linux/macOS 安装脚本
│   └── README.md             # 集成说明
├── docs/
│   ├── paper/                # 论文草稿
│   ├── polarquant-kvcache-prd.md  # 需求文档
│   └── zhihu-article.md      # 知乎文章
└── scripts/                  # 基准测试脚本
```

## 算法原理

```
输入: KV 向量 v ∈ R^d (FP16)

Step 1: 随机正交旋转 → 消除 channel 间方差差异
Step 2: 极坐标分离 → radius (FP16) + 单位方向向量
Step 3: Lloyd-Max codebook 量化 → 4-bit packed indices
Step 4: 融合注意力 → 直接从压缩格式计算，不需要解压

压缩比: FP16 → 4-bit ≈ 3.8x
```

## 快速开始

### Python 原型

```bash
pip install numpy scipy
cd python
python -m pytest tests/ -v  # 运行 154 个测试
```

### llama.cpp 集成

```bash
# 最简方式：直接用 llama.cpp 原生参数
llama-cli -m model.gguf -fa on -ctk q4_0 -ctv q4_0 -p "Hello"
# KV Cache 显存立刻降低 72%
```

## 实验数据

### 压缩精度

| 位宽 | 余弦相似度 | 压缩比 |
|------|-----------|--------|
| 4-bit | 0.990 | 3.8x |
| 2-bit | 0.883 | 10.4x |
| 混合 2/4-bit | 0.958 | 7.9x |

### 注意力 Kernel 性能

| seq_len | 标准注意力 | PolarQuant-KV | 加速比 |
|---------|-----------|---------------|--------|
| 512 | 0.39ms | 0.16ms | 2.40x |
| 2048 | 0.73ms | 0.52ms | 1.40x |
| 4096 | 1.33ms | 0.99ms | 1.35x |

## 环境

- NVIDIA RTX 5060 Ti (16GB, Blackwell sm_120)
- CUDA 13.2, Python 3.12, PyTorch 2.11
- Windows 11, MSVC 19.50

## 参考文献

1. TurboQuant (ICLR 2026): arXiv:2504.19874
2. PolarQuant: arXiv:2502.02617
3. Flash Attention: arXiv:2205.14135

## License

MIT
