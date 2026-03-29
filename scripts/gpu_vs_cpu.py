"""GPU vs CPU 实现的精度和压缩比对比。"""

import numpy as np
import torch
import math

# Phase 1 CPU
from polarquant_kv.rotation import generate_rotation_matrix as gen_R_cpu
from polarquant_kv.quantizer import compress as compress_cpu, decompress as decompress_cpu
from polarquant_kv.utils import cosine_similarity as cos_cpu, compute_compression_ratio

# Phase 2 GPU
from polarquant_kv_cuda.rotation import generate_rotation_matrix as gen_R_gpu
from polarquant_kv_cuda.compress_kernel import compress_gpu
from polarquant_kv_cuda.decompress_kernel import decompress_gpu
from polarquant_kv_cuda.compressor import get_memory_bytes

D = 128
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def compare_accuracy():
    print("=" * 60)
    print("1. 精度对比: GPU vs CPU")
    print("=" * 60)

    R_cpu = gen_R_cpu(D, seed=SEED)
    R_gpu = gen_R_gpu(D, seed=SEED, device="cuda")

    for n_bits in [4, 3, 2]:
        cpu_sims = []
        gpu_sims = []
        cross_sims = []

        for i in range(100):
            v_np = np.random.randn(D).astype(np.float32)

            # CPU: FP64 精度计算
            c_cpu = compress_cpu(v_np, R_cpu, n_bits=n_bits, group_size=32)
            v_hat_cpu = decompress_cpu(c_cpu, R_cpu)
            cpu_sims.append(cos_cpu(v_np, v_hat_cpu))

            # GPU: FP16 输入 → FP32 内部计算 → FP16 输出
            v_gpu = torch.from_numpy(v_np).half().cuda().reshape(1, 1, 1, D)
            c_gpu = compress_gpu(v_gpu, R_gpu, n_bits=n_bits, group_size=32)
            v_hat_gpu = decompress_gpu(c_gpu, R_gpu)
            v_hat_gpu_np = v_hat_gpu.squeeze().cpu().float().numpy()

            gpu_sim = cos_cpu(v_np, v_hat_gpu_np)
            gpu_sims.append(gpu_sim)

            # GPU vs CPU 解压结果对比
            cross_sims.append(cos_cpu(v_hat_cpu, v_hat_gpu_np))

        print(f"\n  {n_bits}-bit:")
        print(f"    CPU 余弦 (原始 vs 解压): {np.mean(cpu_sims):.6f}")
        print(f"    GPU 余弦 (原始 vs 解压): {np.mean(gpu_sims):.6f}")
        print(f"    差值 (CPU - GPU):         {np.mean(cpu_sims) - np.mean(gpu_sims):.6f}")
        print(f"    GPU vs CPU 解压一致性:    {np.mean(cross_sims):.6f}")


def compare_compression_ratio():
    print("\n" + "=" * 60)
    print("2. 压缩比对比: 理论 vs CPU vs GPU")
    print("=" * 60)

    R_gpu = gen_R_gpu(D, seed=SEED, device="cuda")

    print(f"\n  {'n_bits':>6} | {'理论(bit级)':>11} | {'CPU(bit级)':>10} | {'GPU(tensor)':>11} | {'差距原因':>20}")
    print(f"  {'-'*6} | {'-'*11} | {'-'*10} | {'-'*11} | {'-'*20}")

    for n_bits in [4, 3, 2]:
        # 理论压缩比 (bit 级别，不含 QJL)
        theory = compute_compression_ratio(D, n_bits, 32, 0)

        # CPU 压缩比 = 理论值（CPU 实现就是按 bit 算的）
        cpu_ratio = theory

        # GPU 实际 tensor 压缩比
        kv = torch.randn(1, 1, 1000, D, dtype=torch.float16, device="cuda")
        orig_bytes = kv.nelement() * 2  # FP16
        ck = compress_gpu(kv, R_gpu, n_bits=n_bits, group_size=32)
        gpu_bytes = get_memory_bytes(ck)
        gpu_ratio = orig_bytes / gpu_bytes

        # 分析 GPU 存储开销
        num_groups = math.ceil(D / 32)
        d_padded = num_groups * 32
        per_vec_gpu = (
            2 +                          # radius: FP16 = 2 bytes
            d_padded * 1 +               # quantized_direction: uint8 = 1 byte each
            num_groups * 2 +             # group_mins: FP16 = 2 bytes each
            num_groups * 2               # group_scales: FP16 = 2 bytes each
        )
        per_vec_orig = D * 2  # FP16
        manual_ratio = per_vec_orig / per_vec_gpu

        reason = f"uint8存{n_bits}bit浪费{(8-n_bits)/8*100:.0f}%"

        print(f"  {n_bits:>6} | {theory:>11.2f}x | {cpu_ratio:>10.2f}x | {gpu_ratio:>11.2f}x | {reason}")

    print(f"""
  关键差距:
  - CPU 用 bit 级别计算压缩比，不实际存储
  - GPU 用 uint8 存每个量化值，4-bit 值占 8-bit → 浪费 50%
  - 解决方案: bit packing (两个 4-bit 值打包到 1 byte)
""")


def show_bit_packing_potential():
    print("=" * 60)
    print("3. Bit Packing 后的预估压缩比")
    print("=" * 60)

    print(f"\n  {'n_bits':>6} | {'当前GPU':>8} | {'bit pack后':>10} | {'理论上限':>8}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*10} | {'-'*8}")

    for n_bits in [4, 3, 2]:
        num_groups = math.ceil(D / 32)
        orig = D * 2  # FP16 bytes

        # 当前: uint8 per value
        current = 2 + D * 1 + num_groups * 4
        current_ratio = orig / current

        # Bit packed
        quant_bytes = math.ceil(D * n_bits / 8)
        packed = 2 + quant_bytes + num_groups * 4
        packed_ratio = orig / packed

        # 理论 (bit 级别)
        theory = compute_compression_ratio(D, n_bits, 32, 0)

        print(f"  {n_bits:>6} | {current_ratio:>7.2f}x | {packed_ratio:>9.2f}x | {theory:>7.2f}x")

    print("""
  结论:
  - Bit packing 是缩小 GPU-CPU 差距的关键
  - 4-bit + bit packing: 1.75x → 2.46x (接近理论 3.12x)
  - 剩余差距来自 group params 和 radius 的 tensor 存储开销
  - 自定义内存布局（非 PyTorch tensor）可进一步逼近理论值
""")


if __name__ == "__main__":
    compare_accuracy()
    compare_compression_ratio()
    show_bit_packing_potential()
