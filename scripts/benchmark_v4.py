"""V4 策略: 不写自定义 attention kernel，
只优化 K 的解压路径（在旋转空间解压，省掉逆旋转），
然后用 PyTorch 的高效 GEMV 算 score。

核心洞察: SDPA 慢在 seq_len 大时的内存带宽。
如果 K 用 4-bit 存储，解压后用 PyTorch matmul 算 score，
解压 + matmul 的总时间可能比直接读 FP16 K 做 matmul 更快，
因为从全局内存读的数据量少了 3x。
"""

import torch
import math
import time

from polarquant_kv_cuda.rotation import generate_rotation_matrix
from polarquant_kv_cuda.compress_kernel import compress_gpu, _bit_unpack_quantized
from polarquant_kv_cuda.decompress_kernel import decompress_gpu

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


def attention_v4(Q, ck, cv, R):
    """V4: 旋转空间解压 K（不逆旋转）+ PyTorch matmul。"""
    B, Hq, Sq, D_dim = Q.shape
    S = ck.seq_len
    gs = ck.group_size
    num_groups = D_dim // gs
    d_padded = num_groups * gs

    # Q 旋转到旋转空间
    q_rot = Q.squeeze(2).float() @ R.T  # [B, Hq, D]

    # K: 解压 direction（不乘 radius，不逆旋转）
    quantized = _bit_unpack_quantized(ck.quantized_direction, 4, d_padded).float()
    quantized_grouped = quantized.reshape(B, Hq, S, num_groups, gs)
    gm = ck.group_mins.float()
    gsc = ck.group_scales.float()
    direction = (quantized_grouped * gsc.unsqueeze(-1) + gm.unsqueeze(-1))
    direction = direction.reshape(B, Hq, S, d_padded)[..., :D_dim]
    radius = ck.radius.float()

    # Score = q_rot @ direction^T * radius * scale
    scores = torch.matmul(q_rot.unsqueeze(2), direction.transpose(-2, -1)).squeeze(2)
    scores = scores * radius * (1.0 / math.sqrt(D_dim))

    # Softmax
    weights = torch.softmax(scores, dim=-1)

    # V: 完整解压
    V_hat = decompress_gpu(cv, R)

    # Output
    output = torch.matmul(weights.unsqueeze(2), V_hat.float()).squeeze(2)
    return output.unsqueeze(2).half()


def attention_v5(Q, K_dir, K_rad, V_hat, R):
    """V5: 预解压 K direction（缓存），只做 matmul。
    
    关键: 在 KV Cache append 时就解压 direction 并缓存，
    注意力计算时直接用缓存的 direction 做 matmul。
    这样注意力计算只有 matmul，没有解压开销。
    """
    B, Hq, Sq, D_dim = Q.shape
    q_rot = Q.squeeze(2).float() @ R.T
    scores = torch.matmul(q_rot.unsqueeze(2), K_dir.transpose(-2, -1)).squeeze(2)
    scores = scores * K_rad * (1.0 / math.sqrt(D_dim))
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights.unsqueeze(2), V_hat.float()).squeeze(2)
    return output.unsqueeze(2).half()


print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"\n{'seq':>6} | {'SDPA':>8} | {'V4':>8} | {'V5':>8} | {'V5 vs SDPA':>10} | {'Cos':>6}")
print(f"{'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*10} | {'-'*6}")

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

    v4_lat = measure(lambda: attention_v4(Q, ck, cv, R))

    # V5: 预解压 direction 并缓存
    d_padded = D
    quantized = _bit_unpack_quantized(ck.quantized_direction, 4, d_padded).float()
    quantized_grouped = quantized.reshape(1, Hq, S, 4, 32)
    direction = (quantized_grouped * ck.group_scales.float().unsqueeze(-1)
                 + ck.group_mins.float().unsqueeze(-1)).reshape(1, Hq, S, D)
    K_dir = direction.contiguous()
    K_rad = ck.radius.float()
    V_hat = decompress_gpu(cv, R)

    v5_lat = measure(lambda: attention_v5(Q, K_dir, K_rad, V_hat, R))

    # 精度
    out_std = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float()).half()
    out_v5 = attention_v5(Q, K_dir, K_rad, V_hat, R)
    cos = torch.nn.functional.cosine_similarity(
        out_std.flatten().float(), out_v5.flatten().float(), dim=0
    ).item()

    v5_speedup = sdpa_lat / v5_lat
    print(f"{S:>6} | {sdpa_lat:>6.3f}ms | {v4_lat:>6.3f}ms | {v5_lat:>6.3f}ms | {v5_speedup:>8.2f}x | {cos:>5.3f}")

    # 显存对比
    if S == 4096:
        std_mem = K.nelement() * 2 + V.nelement() * 2  # FP16 K+V
        comp_mem = (ck.quantized_direction.nelement() + ck.group_mins.nelement() * 2
                    + ck.group_scales.nelement() * 2 + ck.radius.nelement() * 2)
        v5_mem = K_dir.nelement() * 4 + K_rad.nelement() * 4 + V_hat.nelement() * 2
        print(f"\n  显存 (seq=4096): 标准={std_mem/1e6:.1f}MB, 压缩={comp_mem/1e6:.1f}MB, V5缓存={v5_mem/1e6:.1f}MB")

    del Q, K, V, ck, cv
    torch.cuda.empty_cache()
