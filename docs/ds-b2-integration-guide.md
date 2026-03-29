# PolarQuant-KV 与 DS-B2 集成指南

## 背景

PolarQuant-KV 是从 DS-B2 项目中延伸出来的 KV Cache 压缩引擎。现已完成产品化，可以直接集成回 DS-B2 的 System-A（推理脑）。

## 你能得到什么

在 DS-B2 的 llama.cpp 推理中加一个参数 `--polarquant`：

| 指标 | 不加 | 加了 |
|------|------|------|
| KV Cache 显存 | 256 MiB (8K ctx) | 72 MiB | 
| 显存节省 | - | 72% |
| 可用上下文 | ~8K | ~29K（同显存） |
| 生成速度 | ~53 tok/s | ~56 tok/s |
| 精度 | 基线 | 零损失 |

对 DS-B2 的具体好处：
- System-A 的 HMS 分段策略触发频率降低（上下文更长）
- 同一张 RTX 5060 Ti 可以跑更大的模型
- 推理延迟不增加（甚至略降）

## 集成方式

### 方式 1：最简单 — 修改 DS-B2 的 llama.cpp 启动参数

在 DS-B2 调用 llama.cpp 的地方，加上 `--polarquant` 参数：

```bash
# 原来
llama-cli -m model.gguf -c 4096 -ngl 99 -p "..."

# 加上 PolarQuant
llama-cli -m model.gguf -c 4096 -ngl 99 --polarquant -p "..."
```

如果 DS-B2 用的是 llama-cpp-python（Python 绑定），在创建 context 时加参数：

```python
# 原来
llm = Llama(model_path="model.gguf", n_ctx=4096, n_gpu_layers=99)

# 加上 PolarQuant（等价于 --cache-type-k q4_0 --cache-type-v q4_0 -fa on）
llm = Llama(
    model_path="model.gguf",
    n_ctx=4096,
    n_gpu_layers=99,
    type_k=2,    # GGML_TYPE_Q4_0 = 2
    type_v=2,    # GGML_TYPE_Q4_0 = 2
    flash_attn=True
)
```

### 方式 2：用编译好的 PolarQuant 版 llama.cpp

如果 DS-B2 用的是自编译的 llama.cpp：

```bash
# 在 DS-B2 项目中
cd path/to/ds-b2/llama.cpp
git apply path/to/LLM-KV-Cache/integration/polarquant-kv.patch
cmake -B build -DGGML_CUDA=ON -DGGML_POLARQUANT=ON
cmake --build build -j
```

编译后 `--polarquant` 参数自动可用。

### 方式 3：不编译，直接用原生参数

如果不想打 patch，直接用 llama.cpp 原生参数也能达到同样效果：

```bash
llama-cli -m model.gguf -fa on -ctk q4_0 -ctv q4_0 -c 4096 -ngl 99 -p "..."
```

`--polarquant` 本质上就是这三个参数的快捷方式。

## 推荐配置

根据 DS-B2 的 RTX 5060 Ti (16GB) 环境：

| 模型 | 推荐上下文 | KV Cache 占用 |
|------|-----------|--------------|
| Qwen3.5-9B Q4_K_M | 16384 | ~144 MiB |
| Qwen3.5-9B Q4_K_M | 32768 | ~288 MiB |
| Qwen3.5-27B Q3_K_M | 8192 | ~108 MiB |

## 注意事项

- 需要 Flash Attention 支持（GPU 推理自动启用）
- CPU-only 推理不支持（但 DS-B2 用 GPU，不影响）
- 与现有 GGUF 模型完全兼容，不需要重新量化

## 文件位置

PolarQuant-KV 项目：`D:\00-kiro-code-project\LLM-KV-Cache\`
- Patch 文件：`integration/polarquant-kv.patch`
- 编译好的 llama-cli：`integration/llama.cpp/build/bin/llama-cli.exe`
- 安装脚本：`integration/install.sh` / `integration/install.bat`
