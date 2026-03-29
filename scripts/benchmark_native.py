"""纯 C++ CUDA kernel benchmark（零 Python overhead）。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc"))

import torch, math
import polarquant_native  # 编译好的 .pyd

from polarquant_kv_cuda.rotation import generate_rotation_matrix
from polarquant_kv_cuda.compress_kernel import compress_gpu
from polarquant_kv_cuda.compressor import get_memory_bytes

D = 128; SEED = 42; DEVICE = "cuda"
torch.manual_seed(SEED)
R = generate_rotation_matrix(D, seed=SEED, device=DEVICE)

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

def native_attention(Q, ck, cv, R, out, score_buf):
    """纯 C++ kernel 调用。"""
    B, Hq, _, D_dim = Q.shape
    S = ck.seq_len
    Hkv = ck.radius.shape[1]
    gs = ck.group_size
    ng = (D_dim + gs - 1) // gs
    scale = 1.0 / math.sqrt(D_dim)

    q_rot = (Q.squeeze(2).float() @ R.T).reshape(B * Hq, D_dim).contiguous()

    # 保持引用，防止临时 tensor 被 GC
    pk = ck.quantized_direction.reshape(B*Hkv, S, -1).contiguous()
    km = ck.group_mins.float().reshape(B*Hkv, S, ng).contiguous()
    ks = ck.group_scales.float().reshape(B*Hkv, S, ng).contiguous()
    kr = ck.radius.float().reshape(B*Hkv, S).contiguous()
    pv = cv.quantized_direction.reshape(B*Hkv, S, -1).contiguous()
    vm = cv.group_mins.float().reshape(B*Hkv, S, ng).contiguous()
    vs = cv.group_scales.float().reshape(B*Hkv, S, ng).contiguous()
    vr = cv.radius.float().reshape(B*Hkv, S).contiguous()

    polarquant_native.launch_flash_attn(
        q_rot.data_ptr(), pk.data_ptr(), km.data_ptr(), ks.data_ptr(), kr.data_ptr(),
        pv.data_ptr(), vm.data_ptr(), vs.data_ptr(), vr.data_ptr(),
        out.data_ptr(), score_buf.data_ptr(),
        B * Hq, B * Hkv, S, D_dim, gs, ng, scale,
    )
    result = out.reshape(B, Hq, D_dim) @ R
    return result.unsqueeze(2).half()

def native_attention_precomputed(Q, R, q_rot_buf, out, score_buf, ptrs):
    """预计算版：KV 数据指针只算一次。"""
    B, Hq, _, D_dim = Q.shape
    scale = 1.0 / math.sqrt(D_dim)
    torch.matmul(Q.squeeze(2).float(), R.T, out=q_rot_buf)
    polarquant_native.launch_flash_attn(
        q_rot_buf.data_ptr(), *ptrs, out.data_ptr(), score_buf.data_ptr(),
        ptrs[-3], ptrs[-2], ptrs[-1], D_dim, 32, D_dim // 32, scale,
    )
    return (out.reshape(B, Hq, D_dim) @ R).unsqueeze(2).half()

Hq = 32
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"\n{'seq':>6}|{'SDPA':>7}|{'Native':>7}|{'Speedup':>8}|{'Cos':>6}|{'MemSave':>7}")
print(f"{'-'*6}|{'-'*7}|{'-'*7}|{'-'*8}|{'-'*6}|{'-'*7}")

for S in [512, 2048, 4096, 8192, 16384]:
    Q = torch.randn(1, Hq, 1, D, dtype=torch.float16, device=DEVICE)
    K = torch.randn(1, Hq, S, D, dtype=torch.float16, device=DEVICE)
    V = torch.randn(1, Hq, S, D, dtype=torch.float16, device=DEVICE)

    sdpa = measure(lambda: torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float()))

    ck = compress_gpu(K, R, n_bits=4, group_size=32)
    cv = compress_gpu(V, R, n_bits=4, group_size=32)

    out = torch.zeros(Hq, D, dtype=torch.float32, device=DEVICE)
    score_buf = torch.zeros(Hq, S, dtype=torch.float32, device=DEVICE)

    nat = measure(lambda: native_attention(Q, ck, cv, R, out, score_buf))

    # 跳过预计算版（简化测试）
    natp = nat

    out_std = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float()).half()
    out_nat = native_attention(Q, ck, cv, R, out, score_buf)
    cos = torch.nn.functional.cosine_similarity(
        out_std.flatten().float(), out_nat.flatten().float(), dim=0).item()

    std_mem = K.nelement() * 2 + V.nelement() * 2
    comp_mem = get_memory_bytes(ck) + get_memory_bytes(cv)
    save = (1 - comp_mem / std_mem) * 100
    best = nat
    print(f"{S:>6}|{sdpa:>5.2f}ms|{nat:>5.2f}ms|{sdpa/best:>7.2f}x|{cos:>5.3f}|{save:>5.0f}%")

    del Q, K, V, ck, cv; torch.cuda.empty_cache()
