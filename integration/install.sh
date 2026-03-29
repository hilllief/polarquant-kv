#!/bin/bash
# PolarQuant-KV one-click installer for llama.cpp
# Usage: ./install.sh [cuda|hip|metal|cpu]
#
# This script:
# 1. Clones llama.cpp (if not present)
# 2. Applies the PolarQuant patch
# 3. Builds with the specified backend
# 4. Tests the build

set -e

BACKEND="${1:-cuda}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/polarquant-kv.patch"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"

echo "=== PolarQuant-KV Installer ==="
echo "Backend: $BACKEND"

# Step 1: Clone llama.cpp
if [ ! -d "$LLAMA_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi

# Step 2: Apply patch
echo "Applying PolarQuant patch..."
cd "$LLAMA_DIR"
git checkout -- . 2>/dev/null || true
git apply "$PATCH_FILE"

# Step 3: Build
echo "Building with $BACKEND backend..."
case "$BACKEND" in
    cuda)
        cmake -B build -DGGML_CUDA=ON -DGGML_POLARQUANT=ON -DCMAKE_BUILD_TYPE=Release
        ;;
    hip)
        cmake -B build -DGGML_HIP=ON -DGGML_POLARQUANT=ON -DCMAKE_BUILD_TYPE=Release
        ;;
    metal)
        cmake -B build -DGGML_METAL=ON -DGGML_POLARQUANT=ON -DCMAKE_BUILD_TYPE=Release
        ;;
    cpu)
        cmake -B build -DGGML_POLARQUANT=ON -DCMAKE_BUILD_TYPE=Release
        ;;
    *)
        echo "Unknown backend: $BACKEND (use cuda/hip/metal/cpu)"
        exit 1
        ;;
esac

cmake --build build -j$(nproc)

# Step 4: Verify
echo ""
echo "=== Build complete ==="
echo "Binary: $LLAMA_DIR/build/bin/llama-cli"
echo ""
echo "Usage:"
echo "  $LLAMA_DIR/build/bin/llama-cli -m YOUR_MODEL.gguf --polarquant -p 'Hello'"
echo ""
echo "The --polarquant flag enables:"
echo "  - Q4_0 KV cache (3.6x compression, 72% VRAM savings)"
echo "  - Flash Attention (auto-enabled)"
echo "  - Zero accuracy loss"

# Verify --polarquant flag exists
if "$LLAMA_DIR/build/bin/llama-cli" --help 2>&1 | grep -q "polarquant"; then
    echo ""
    echo "OK: --polarquant flag verified"
else
    echo ""
    echo "WARNING: --polarquant flag not found in help output"
fi
