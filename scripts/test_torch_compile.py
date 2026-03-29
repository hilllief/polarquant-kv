"""测试 PyTorch C++ 扩展编译（含 CCCL workaround）。"""
import os
os.environ["PATH"] = (
    r"D:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;"
    + os.environ.get("PATH", "")
)

import torch
from torch.utils.cpp_extension import load

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

csrc = os.path.join(os.path.dirname(__file__), "..", "csrc")

mod = load(
    name="test_torch_ext",
    sources=[os.path.join(csrc, "test_torch_ext.cu")],
    extra_cuda_cflags=[
        "-O3", "--use_fast_math",
        "-Xcompiler", "/Zc:preprocessor",
        "-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING",
    ],
    extra_cflags=["/O2", "/Zc:preprocessor"],
    verbose=True,
)

print(f"\nCompile OK! Module: {mod}")

a = torch.randn(16, device="cuda")
b = torch.randn(16, device="cuda")
c = mod.test_add(a, b)
expected = a + b
print(f"test_add correct: {torch.allclose(c, expected)}")
print(f"Result: {c[:4]}")
