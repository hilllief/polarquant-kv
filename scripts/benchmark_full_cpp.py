"""全 C++ 版本 benchmark。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc"))

import torch, math
import polarquant_full

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

Hq = 32
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"\n{'seq':>6}|{'SDPA':>7}|{'FullC++':>8}|{'Speedup':>8}|{'Cos':>6}|{'Mem':>5}")
print(f"{'-'*6}|{'-'*7}|{'-'*8}|{'-'*8}|{'-'*6}|{'-'*5}")

for S in [512, 2048, 4096, 8192, 16384]:
    Q = torch.randn(1, Hq, 1, D, dtype=torch.float16, device=DEVICE)
    K = torch.randn(1, Hq, S, D, dtype=torch.float16, device=DEVICE)
    V = torch.randn(1, Hq, S, D, dtype=torch.float16, device=DEVICE)

    # 标准 SDPA
    sdpa = measure(lambda: torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float()))

    # 压缩
    ck = compress_gpu(K, R, n_bits=4, group_size=32)
    cv = compress_gpu(V, R, n_bits=4, group_size=32)
    ng = D // 32
    scale = 1.0 / math.sqrt(D)

    # 预分配所有 buffer（只做一次）
    q_f32 = Q.squeeze(2).float().reshape(Hq, D).contiguous()
    pk = ck.quantized_direction.reshape(Hq, S, -1).contiguous()
    km = ck.group_mins.float().reshape(Hq, S, ng).contiguous()
    ks = ck.group_scales.float().reshape(Hq, S, ng).contiguous()
    kr = ck.radius.float().reshape(Hq, S).contiguous()
    pv = cv.quantized_direction.reshape(Hq, S, -1).contiguous()
    vm = cv.group_mins.float().reshape(Hq, S, ng).contiguous()
    vs = cv.group_scales.float().reshape(Hq, S, ng).contiguous()
    vr = cv.radius.float().reshape(Hq, S).contiguous()
    out = torch.zeros(Hq, D, dtype=torch.float32, device=DEVICE)
    q_rot = torch.zeros(Hq, D, dtype=torch.float32, device=DEVICE)
    score = torch.zeros(Hq, S, dtype=torch.float32, device=DEVICE)
    attn_out = torch.zeros(Hq, D, dtype=torch.float32, device=DEVICE)

    def run_full():
        polarquant_full.full_compressed_attention(
            q_f32.data_ptr(), R.data_ptr(),
            pk.data_ptr(), km.data_ptr(), ks.data_ptr(), kr.data_ptr(),
            pv.data_ptr(), vm.data_ptr(), vs.data_ptr(), vr.data_ptr(),
            out.data_ptr(), q_rot.data_ptr(), score.data_ptr(),
            attn_out.data_ptr(),
            Hq, Hq, S, D, 32, ng, scale,
        )

    cpp = measure(run_full)

    # 精度
    q_f32_check = Q.squeeze(2).float().reshape(Hq, D).contiguous()
    polarquant_full.full_compressed_attention(
        q_f32_check.data_ptr(), R.data_ptr(),
        pk.data_ptr(), km.data_ptr(), ks.data_ptr(), kr.data_ptr(),
        pv.data_ptr(), vm.data_ptr(), vs.data_ptr(), vr.data_ptr(),
        out.data_ptr(), q_rot.data_ptr(), score.data_ptr(),
        attn_out.data_ptr(),
        Hq, Hq, S, D, 32, ng, scale,
    )
    torch.cuda.synchronize()
    out_std = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float()).half()
    out_cpp = out.reshape(1, Hq, D).half()
    cos = torch.nn.functional.cosine_similarity(
        out_std.flatten().float(), out_cpp.flatten().float(), dim=0).item()

    std_mem = K.nelement() * 2 + V.nelement() * 2
    comp_mem = get_memory_bytes(ck) + get_memory_bytes(cv)
    save = (1 - comp_mem / std_mem) * 100

    print(f"{S:>6}|{sdpa:>5.2f}ms|{cpp:>6.2f}ms|{sdpa/cpp:>6.2f}x|{cos:>5.3f}|{save:>3.0f}%")

    del Q, K, V, ck, cv; torch.cuda.empty_cache()
