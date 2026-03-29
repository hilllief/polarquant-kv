"""Flash Attention V3 vs V4 benchmark。"""
import torch, math
from polarquant_kv_cuda.rotation import generate_rotation_matrix
from polarquant_kv_cuda.compress_kernel import compress_gpu
from polarquant_kv_cuda.flash_attention_native import (
    flash_compressed_attention_v3, flash_compressed_attention_v4,
)

D = 128; SEED = 42; DEVICE = "cuda"
torch.manual_seed(SEED)
R = generate_rotation_matrix(D, seed=SEED, device=DEVICE)

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

Hq = 32
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"\n{'seq':>6} | {'SDPA':>8} | {'V3':>8} | {'V4tile':>8} | {'V4/SDPA':>8} | {'Cos':>6}")
print(f"{'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*6}")

for S in [512, 2048, 4096, 8192, 16384]:
    Q = torch.randn(1, Hq, 1, D, dtype=torch.float16, device=DEVICE)
    K = torch.randn(1, Hq, S, D, dtype=torch.float16, device=DEVICE)
    V = torch.randn(1, Hq, S, D, dtype=torch.float16, device=DEVICE)

    sdpa = measure(lambda: torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float()))
    ck = compress_gpu(K, R, n_bits=4, group_size=32)
    cv = compress_gpu(V, R, n_bits=4, group_size=32)
    v3 = measure(lambda: flash_compressed_attention_v3(Q, ck, cv, R))
    v4 = measure(lambda: flash_compressed_attention_v4(Q, ck, cv, R))

    out_std = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float()).half()
    out_v4 = flash_compressed_attention_v4(Q, ck, cv, R)
    cos = torch.nn.functional.cosine_similarity(
        out_std.flatten().float(), out_v4.flatten().float(), dim=0).item()

    print(f"{S:>6} | {sdpa:>6.3f}ms | {v3:>6.3f}ms | {v4:>6.3f}ms | {sdpa/v4:>6.2f}x | {cos:>5.3f}")
    del Q, K, V, ck, cv; torch.cuda.empty_cache()
