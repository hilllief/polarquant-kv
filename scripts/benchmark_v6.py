"""V6: 快速解压 kernel + PyTorch matmul。
存储压缩格式（省显存），注意力时快速解压到临时 FP16 buffer + matmul。
"""
import torch, math
from polarquant_kv_cuda.rotation import generate_rotation_matrix
from polarquant_kv_cuda.compress_kernel import compress_gpu
from polarquant_kv_cuda.decompress_kernel import decompress_gpu
from polarquant_kv_cuda.fused_cuda_kernels import fast_dequant_direction_4bit
from polarquant_kv_cuda.compressor import get_memory_bytes

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


def attention_v6(Q, ck, cv, R):
    """V6: 快速解压 + PyTorch matmul。"""
    B, Hq, Sq, D_dim = Q.shape
    S = ck.seq_len

    # Q 旋转
    q_rot = Q.squeeze(2).float() @ R.T  # [B, Hq, D]

    # K: 快速解压 direction（FP16，不乘 radius，不逆旋转）
    K_dir = fast_dequant_direction_4bit(
        ck.quantized_direction, ck.group_mins, ck.group_scales, D_dim, ck.group_size
    )  # [B, Hq, S, D], FP16

    # Score = q_rot @ K_dir^T * radius * scale
    scores = torch.matmul(q_rot.unsqueeze(2), K_dir.transpose(-2, -1)).squeeze(2)
    scores = scores * ck.radius.float() * (1.0 / math.sqrt(D_dim))

    # Softmax
    weights = torch.softmax(scores, dim=-1)

    # V: 完整解压
    V_hat = decompress_gpu(cv, R)

    # Output
    output = torch.matmul(weights.unsqueeze(2), V_hat.float()).squeeze(2)
    return output.unsqueeze(2).half()


print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"\n{'seq':>6} | {'SDPA':>8} | {'V6':>8} | {'Speedup':>8} | {'Cos':>6} | {'Mem Save':>8}")
print(f"{'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*6} | {'-'*8}")

for S in [512, 2048, 4096, 8192, 16384]:
    Hq = 32
    Q = torch.randn(1, Hq, 1, D, dtype=torch.float16, device=DEVICE)
    K = torch.randn(1, Hq, S, D, dtype=torch.float16, device=DEVICE)
    V = torch.randn(1, Hq, S, D, dtype=torch.float16, device=DEVICE)

    sdpa_lat = measure(lambda: torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float()
    ))

    ck = compress_gpu(K, R, n_bits=4, group_size=32)
    cv = compress_gpu(V, R, n_bits=4, group_size=32)

    v6_lat = measure(lambda: attention_v6(Q, ck, cv, R))

    out_std = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float()).half()
    out_v6 = attention_v6(Q, ck, cv, R)
    cos = torch.nn.functional.cosine_similarity(
        out_std.flatten().float(), out_v6.flatten().float(), dim=0
    ).item()

    std_mem = K.nelement() * 2 + V.nelement() * 2
    comp_mem = get_memory_bytes(ck) + get_memory_bytes(cv)
    mem_save = (1 - comp_mem / std_mem) * 100

    speedup = sdpa_lat / v6_lat
    print(f"{S:>6} | {sdpa_lat:>6.3f}ms | {v6_lat:>6.3f}ms | {speedup:>6.2f}x | {cos:>5.3f} | {mem_save:>5.0f}%")

    del Q, K, V, ck, cv
    torch.cuda.empty_cache()
