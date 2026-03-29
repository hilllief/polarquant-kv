"""分析融合注意力各步骤的耗时。"""
import torch
import cupy as cp
import math
import time
import numpy as np

from polarquant_kv_cuda.rotation import generate_rotation_matrix
from polarquant_kv_cuda.compress_kernel import compress_gpu
from polarquant_kv_cuda.fused_cuda_kernels import (
    _to_cp, _FUSED_ATTENTION_4BIT, fused_attention_4bit_cuda,
)

D = 128; Hq = 32; S = 4096; B = 1
R = generate_rotation_matrix(D, seed=42, device="cuda")
torch.manual_seed(42)

Q = torch.randn(B, Hq, 1, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, Hq, S, D, dtype=torch.float16, device="cuda")
V = torch.randn(B, Hq, S, D, dtype=torch.float16, device="cuda")
ck = compress_gpu(K, R, n_bits=4, group_size=32)
cv = compress_gpu(V, R, n_bits=4, group_size=32)

# Warmup
for _ in range(5):
    fused_attention_4bit_cuda(Q, ck, cv, R)
torch.cuda.synchronize()

# Profile each step
num_groups = D // 32
scale = 1.0 / math.sqrt(D)

def time_step(name, fn, repeat=50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) / repeat * 1000
    print(f"  {name:30s}: {ms:.3f} ms")

# Step 1: q_rot
time_step("q @ R^T", lambda: (Q.squeeze(2).float() @ R.T).contiguous())

# Step 2: reshape + float + contiguous
q_rot = (Q.squeeze(2).float() @ R.T).contiguous()
time_step("reshape+float+contiguous (9x)", lambda: (
    q_rot.reshape(B*Hq, D).contiguous(),
    ck.quantized_direction.reshape(B*Hq, S, -1).contiguous(),
    ck.group_mins.float().reshape(B*Hq, S, num_groups).contiguous(),
    ck.group_scales.float().reshape(B*Hq, S, num_groups).contiguous(),
    ck.radius.float().reshape(B*Hq, S).contiguous(),
))

# Step 3: DLPack conversion
q_flat = q_rot.reshape(B*Hq, D).contiguous()
pk_flat = ck.quantized_direction.reshape(B*Hq, S, -1).contiguous()
km_flat = ck.group_mins.float().reshape(B*Hq, S, num_groups).contiguous()
ks_flat = ck.group_scales.float().reshape(B*Hq, S, num_groups).contiguous()
kr_flat = ck.radius.float().reshape(B*Hq, S).contiguous()
pv_flat = cv.quantized_direction.reshape(B*Hq, S, -1).contiguous()
vm_flat = cv.group_mins.float().reshape(B*Hq, S, num_groups).contiguous()
vs_flat = cv.group_scales.float().reshape(B*Hq, S, num_groups).contiguous()
vr_flat = cv.radius.float().reshape(B*Hq, S).contiguous()

time_step("DLPack _to_cp (9x)", lambda: (
    _to_cp(q_flat), _to_cp(pk_flat), _to_cp(km_flat),
    _to_cp(ks_flat), _to_cp(kr_flat), _to_cp(pv_flat),
    _to_cp(vm_flat), _to_cp(vs_flat), _to_cp(vr_flat),
))

# Step 4: kernel only
q_cp = _to_cp(q_flat)
pk_cp = _to_cp(pk_flat); km_cp = _to_cp(km_flat)
ks_cp = _to_cp(ks_flat); kr_cp = _to_cp(kr_flat)
pv_cp = _to_cp(pv_flat); vm_cp = _to_cp(vm_flat)
vs_cp = _to_cp(vs_flat); vr_cp = _to_cp(vr_flat)
out_cp = cp.zeros((B*Hq, D), dtype=cp.float32)
block_size = 128
smem = (D + block_size + 4) * 4

time_step("CUDA kernel only", lambda: _FUSED_ATTENTION_4BIT(
    (B*Hq,), (block_size,),
    (q_cp, pk_cp, km_cp, ks_cp, kr_cp,
     pv_cp, vm_cp, vs_cp, vr_cp, out_cp,
     B*Hq, B*Hq, S, D, 32, num_groups, np.float32(scale)),
    shared_mem=smem,
))

# Step 5: inverse rotate
outputs = torch.zeros(B, Hq, D, dtype=torch.float32, device="cuda")
time_step("inverse rotate (matmul)", lambda: outputs @ R)

# Total
time_step("TOTAL fused_attention_4bit_cuda", lambda: fused_attention_4bit_cuda(Q, ck, cv, R))

# Baseline
time_step("SDPA baseline", lambda: torch.nn.functional.scaled_dot_product_attention(
    Q.float(), K.float(), V.float()
))
