// Workaround for PyTorch 2.11 + MSVC 19.50 + /Zc:preprocessor
// CCCL requires /Zc:preprocessor, but PyTorch headers have std namespace ambiguity
#pragma once

#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING

// 在 valarray 被 include 之前，确保 std namespace 不会被污染
// valarray 在 line 20 有 namespace std { ... }，这和 CUDA 的 std 冲突
// 解决方案：先 include valarray，让 std 确定下来
#include <valarray>

// 然后 include torch
#include <torch/extension.h>
