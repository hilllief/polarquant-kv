"""TurboQuant (Lloyd-Max codebook) vs 当前实现 vs SDPA benchmark。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc"))

import torch, math, numpy as np
import polarquant_turboquant
import polarquant_full

from polarquant_kv_cuda.rotation import generate_rotation_matrix
from polarquant_kv_cuda.lloyd_max import get_codebook_torch
from polarquant_kv_cuda.compress_kernel import compress_gpu
from polarquant_kv_cuda.compressor import get_memory_bytes

D = 128; SEED = 42; DEVICE = "cuda"
torch.manual_seed(SEED)
R = generate_rotation_matrix(D, seed=SEED, device=DEVICE)

# 预计算 Lloyd-Max codebook
k_cb = get_codebook_torch(D, 4, device=DEVICE)
v_cb = get_codebook_torch(D, 4, device=DEVICE)
print(f"4-bit codebook: {k_cb.shape[0]} centroids")
print(f"Centroid range: [{k_cb.min():.4f}, {k_cb.max():.4f}]")

def measure(fn, warmup=20, repeat=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]

Hq = 32
print(f"\nGPU: {torch.cuda.get_device_name(0)}")

print(f"\n{'seq':>6}|{'SDPA':>7}|{'OldC++':>7}|{'Turbo':>7}|{'T/SDPA':>7}|{'Cos':>6}|{'CompR':>6}")
print(f"{'-'*6}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*6}|{'-'*6}")

for S in [512, 2048, 4096, 8192, 16384]:
    Q = torch.randn(1, Hq, 1, D, dtype=torch.float16, device=DEVICE)
    K = torch.randn(1, Hq, S, D, dtype=torch.float16, device=DEVICE)
    V = torch.randn(1, Hq, S, D, dtype=torch.float16, device=DEVICE)

    # SDPA baseline
    sdpa = measure(lambda: torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float()))

    # Old C++ (per-group min/max)
    ck = compress_gpu(K, R, n_bits=4, group_size=32)
    cv = compress_gpu(V, R, n_bits=4, group_size=32)
    ng = D // 32; scale = 1.0 / math.sqrt(D)
    q_f = Q.squeeze(2).float().reshape(Hq, D).contiguous()
    pk_old = ck.quantized_direction.reshape(Hq, S, -1).contiguous()
    km = ck.group_mins.float().reshape(Hq, S, ng).contiguous()
    ks = ck.group_scales.float().reshape(Hq, S, ng).contiguous()
    kr_old = ck.radius.float().reshape(Hq, S).contiguous()
    pv_old = cv.quantized_direction.reshape(Hq, S, -1).contiguous()
    vm = cv.group_mins.float().reshape(Hq, S, ng).contiguous()
    vs = cv.group_scales.float().reshape(Hq, S, ng).contiguous()
    vr_old = cv.radius.float().reshape(Hq, S).contiguous()
    out_old = torch.zeros(Hq, D, dtype=torch.float32, device=DEVICE)
    qr_old = torch.zeros(Hq, D, dtype=torch.float32, device=DEVICE)
    sc_old = torch.zeros(Hq, S, dtype=torch.float32, device=DEVICE)
    ao_old = torch.zeros(Hq, D, dtype=torch.float32, device=DEVICE)

    def run_old():
        polarquant_full.full_compressed_attention(
            q_f.data_ptr(), R.data_ptr(),
            pk_old.data_ptr(), km.data_ptr(), ks.data_ptr(), kr_old.data_ptr(),
            pv_old.data_ptr(), vm.data_ptr(), vs.data_ptr(), vr_old.data_ptr(),
            out_old.data_ptr(), qr_old.data_ptr(), sc_old.data_ptr(), ao_old.data_ptr(),
            Hq, Hq, S, D, 32, ng, scale)
    old_lat = measure(run_old)

    # TurboQuant (Lloyd-Max codebook)
    # 压缩 K 和 V
    K_flat = K.reshape(-1, D).float().contiguous()
    V_flat = V.reshape(-1, D).float().contiguous()
    N = K_flat.shape[0]
    kr_tq = torch.zeros(N, dtype=torch.float32, device=DEVICE)
    pk_tq = torch.zeros(N, D // 2, dtype=torch.uint8, device=DEVICE)
    vr_tq = torch.zeros(N, dtype=torch.float32, device=DEVICE)
    pv_tq = torch.zeros(N, D // 2, dtype=torch.uint8, device=DEVICE)

    polarquant_turboquant.turboquant_compress(
        K_flat.data_ptr(), R.data_ptr(), k_cb.data_ptr(),
        kr_tq.data_ptr(), pk_tq.data_ptr(), N, D, 16)
    polarquant_turboquant.turboquant_compress(
        V_flat.data_ptr(), R.data_ptr(), v_cb.data_ptr(),
        vr_tq.data_ptr(), pv_tq.data_ptr(), N, D, 16)
    torch.cuda.synchronize()

    # Reshape for attention
    pk_tq = pk_tq.reshape(Hq, S, -1).contiguous()
    kr_tq = kr_tq.reshape(Hq, S).contiguous()
    pv_tq = pv_tq.reshape(Hq, S, -1).contiguous()
    vr_tq = vr_tq.reshape(Hq, S).contiguous()

    q_rot = (Q.squeeze(2).float() @ R.T).reshape(Hq, D).contiguous()
    out_tq = torch.zeros(Hq, D, dtype=torch.float32, device=DEVICE)
    sc_tq = torch.zeros(Hq, S, dtype=torch.float32, device=DEVICE)

    def run_turbo():
        polarquant_turboquant.turboquant_full_attention(
            q_rot.data_ptr(), R.data_ptr(),
            pk_tq.data_ptr(), kr_tq.data_ptr(),
            pv_tq.data_ptr(), vr_tq.data_ptr(),
            k_cb.data_ptr(), v_cb.data_ptr(),
            out_tq.data_ptr(), q_rot.data_ptr(), sc_tq.data_ptr(), out_tq.data_ptr(),
            Hq, Hq, S, D, 16, scale)
    tq_lat = measure(run_turbo)

    # 精度
    run_turbo(); torch.cuda.synchronize()
    out_std = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float()).half()
    # TurboQuant 输出在旋转空间，需要逆旋转
    out_tq_final = (out_tq.reshape(1, Hq, D) @ R).half()
    cos = torch.nn.functional.cosine_similarity(
        out_std.flatten().float(), out_tq_final.flatten().float(), dim=0).item()

    # 压缩比
    orig_bytes = K.nelement() * 2 + V.nelement() * 2
    tq_bytes = pk_tq.nelement() + kr_tq.nelement() * 4 + pv_tq.nelement() + vr_tq.nelement() * 4
    comp_ratio = orig_bytes / tq_bytes

    print(f"{S:>6}|{sdpa:>5.2f}ms|{old_lat:>5.2f}ms|{tq_lat:>5.2f}ms|{sdpa/tq_lat:>5.2f}x|{cos:>5.3f}|{comp_ratio:>4.1f}x")

    del Q, K, V; torch.cuda.empty_cache()
