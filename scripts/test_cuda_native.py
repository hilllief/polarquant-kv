"""测试 C++ CUDA 融合 kernel 的 JIT 编译和基本功能。"""

import os
import sys
import torch

# 设置 MSVC 路径
CCBIN = r"D:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64"
os.environ["PATH"] = CCBIN + ";" + os.environ.get("PATH", "")

# 设置 MSVC include/lib 路径
MSVC_ROOT = r"D:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717"
os.environ["INCLUDE"] = os.path.join(MSVC_ROOT, "include") + ";" + os.environ.get("INCLUDE", "")
os.environ["LIB"] = os.path.join(MSVC_ROOT, "lib", "x64") + ";" + os.environ.get("LIB", "")

# Windows SDK
SDK_ROOT = r"C:\Program Files (x86)\Windows Kits\10"
if os.path.exists(SDK_ROOT):
    # 找最新版本
    versions = os.listdir(os.path.join(SDK_ROOT, "Include"))
    versions = [v for v in versions if v.startswith("10.")]
    if versions:
        sdk_ver = sorted(versions)[-1]
        os.environ["INCLUDE"] += ";" + os.path.join(SDK_ROOT, "Include", sdk_ver, "ucrt")
        os.environ["INCLUDE"] += ";" + os.path.join(SDK_ROOT, "Include", sdk_ver, "shared")
        os.environ["INCLUDE"] += ";" + os.path.join(SDK_ROOT, "Include", sdk_ver, "um")
        os.environ["LIB"] += ";" + os.path.join(SDK_ROOT, "Lib", sdk_ver, "ucrt", "x64")
        os.environ["LIB"] += ";" + os.path.join(SDK_ROOT, "Lib", sdk_ver, "um", "x64")

from torch.utils.cpp_extension import load

csrc_dir = os.path.join(os.path.dirname(__file__), "..", "csrc")
cu_file = os.path.join(csrc_dir, "polarquant_kernels.cu")

print(f"Compiling {cu_file}...")
print(f"MSVC: {CCBIN}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    mod = load(
        name="polarquant_cuda_native",
        sources=[cu_file],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-Xcompiler", "/Zc:preprocessor"],
        extra_cflags=["/O2", "/w", "/Zc:preprocessor"],
        verbose=True,
    )
    print("\nCompilation SUCCESS!")
    print(f"Module: {mod}")
    print(f"Functions: {dir(mod)}")

    # 测试 fused_score_4bit
    D = 128
    S = 64
    G = D // 32

    q_rotated = torch.randn(D, dtype=torch.float32, device="cuda")
    packed_keys = torch.randint(0, 256, (S, D // 2), dtype=torch.uint8, device="cuda")
    gmins = torch.randn(S, G, dtype=torch.float16, device="cuda")
    gscales = torch.abs(torch.randn(S, G, dtype=torch.float16, device="cuda"))
    radius = torch.randn(S, dtype=torch.float16, device="cuda")

    scores = mod.fused_score_4bit(q_rotated, packed_keys, gmins, gscales, radius, D, 32, 1.0 / (D ** 0.5))
    print(f"\nfused_score_4bit OK: scores shape={scores.shape}, first 5={scores[:5]}")

except Exception as e:
    print(f"\nCompilation FAILED: {e}")
    import traceback
    traceback.print_exc()
