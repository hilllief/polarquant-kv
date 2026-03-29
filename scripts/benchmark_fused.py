"""融合 kernel 性能 benchmark。"""

import math
import torch
import cupy as cp

from polarquant_kv_cuda.rotation import generate_rotation_matrix
from polarquant_kv_cuda.compress_kernel import compress_gpu
from polarquant_kv_cuda.fused_cuda_kernels import (
    fused_compress_4bit_cuda,
    fused_attention_scores_4bit,
    fused_attention_4bit_cuda,
)

D = 128
SEED = 42
DEVICE = "cuda"
torch.manual_seed(SEED)


def measure(fn, warmup=20, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def benchmark_compress():
    print("=" * 60)
    print("1. 压缩延迟: 融合 kernel vs PyTorch 分步")
    print("=" * 60)

    R = generate_rotation_matrix(D, seed=SEED, device=DEVICE)

    print(f"\n  {'N vectors':>12} | {'PyTorch':>10} | {'Fused':>10} | {'Speedup':>8}")
    print(f"  {'-'*12} | {'-'*10} | {'-'*10} | {'-'*8}")

    for N in [1, 32, 2048, 32*4096]:
        kv = torch.randn(1, 1, N, D, dtype=torch.float16, device=DEVICE)

        pt_lat = measure(lambda: compress_gpu(kv, R, n_bits=4, group_size=32))
        fused_lat = measure(lambda: fused_compress_4bit_cuda(kv, R, group_size=32))

        speedup = pt_lat / fused_lat if fused_lat > 0 else 0
        print(f"  {N:>12} | {pt_lat:>8.3f}ms | {fused_lat:>8.3f}ms | {speedup:>6.1f}x")


def benchmark_attention():
    print("\n" + "=" * 60)
    print("2. 注意力延迟: 融合 kernel vs 标准 SDPA")
    print("=" * 60)

    R = generate_rotation_matrix(D, seed=SEED, device=DEVICE)
    Hq = 32

    print(f"\n  {'seq_len':>8} | {'SDPA':>8} | {'Fused':>8} | {'Speedup':>8}")
    print(f"  {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")

    for S in [128, 512, 2048, 4096]:
        Q = torch.randn(1, Hq, 1, D, dtype=torch.float16, device=DEVICE)
        K = torch.randn(1, Hq, S, D, dtype=torch.float16, device=DEVICE)
        V = torch.randn(1, Hq, S, D, dtype=torch.float16, device=DEVICE)

        # 标准 SDPA
        sdpa_lat = measure(lambda: torch.nn.functional.scaled_dot_product_attention(
            Q.float(), K.float(), V.float()
        ))

        # 融合注意力
        ck = compress_gpu(K, R, n_bits=4, group_size=32)
        cv = compress_gpu(V, R, n_bits=4, group_size=32)

        fused_lat = measure(lambda: fused_attention_4bit_cuda(Q, ck, cv, R))

        speedup = sdpa_lat / fused_lat if fused_lat > 0 else 0
        print(f"  {S:>8} | {sdpa_lat:>6.3f}ms | {fused_lat:>6.3f}ms | {speedup:>6.2f}x")

    # 精度验证
    print("\n  精度验证 (seq=512):")
    Q = torch.randn(1, 4, 1, D, dtype=torch.float16, device=DEVICE)
    K = torch.randn(1, 4, 512, D, dtype=torch.float16, device=DEVICE)
    V = torch.randn(1, 4, 512, D, dtype=torch.float16, device=DEVICE)

    out_std = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float()
    ).half()

    ck = compress_gpu(K, R, n_bits=4, group_size=32)
    cv = compress_gpu(V, R, n_bits=4, group_size=32)
    out_fused = fused_attention_4bit_cuda(Q, ck, cv, R)

    cos = torch.nn.functional.cosine_similarity(
        out_std.flatten().float(), out_fused.flatten().float(), dim=0
    ).item()
    print(f"  余弦相似度: {cos:.4f}")


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    benchmark_compress()
    benchmark_attention()
