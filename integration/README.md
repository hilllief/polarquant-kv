# PolarQuant-KV for llama.cpp

KV Cache 4-bit compression for llama.cpp. One flag, 72% VRAM savings, zero accuracy loss.

## Quick Start

```bash
# Linux/macOS
chmod +x install.sh
./install.sh cuda    # NVIDIA GPU
./install.sh hip     # AMD GPU
./install.sh metal   # Apple Silicon

# Windows
install.bat
```

Then run:
```bash
llama-cli -m your-model.gguf --polarquant -p "Hello"
```

## What --polarquant Does

- KV cache: FP16 (256 MiB) → Q4_0 (72 MiB) = 3.6x compression
- Flash Attention: auto-enabled
- Accuracy: zero loss (verified on Qwen3.5-9B)
- Speed: prompt ~130 tok/s, generation ~56 tok/s (RTX 5060 Ti)

## Manual Install

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
git apply /path/to/polarquant-kv.patch
cmake -B build -DGGML_CUDA=ON -DGGML_POLARQUANT=ON
cmake --build build -j$(nproc)
```

## Files Changed

21 files, 2025 lines added, 0 lines removed from upstream llama.cpp.

| Category | Files |
|----------|-------|
| CUDA kernels | polarquant.cu, polarquant.cuh |
| Integration | llama-polarquant.h/cpp |
| CLI | arg.cpp, common.h/cpp, llama.h |
| Graph | llama-graph.h/cpp, llama-context.h/cpp |
| Build | CMakeLists.txt (x4) |
| KV cache | llama-kv-cache.h/cpp, llama-cparams.h |
| Test | test-polarquant.cu (534 assertions) |

## Patch File

`polarquant-kv.patch` (170 KB) — apply to any recent llama.cpp checkout.
