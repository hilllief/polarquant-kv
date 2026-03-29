"""测试 CuPy RawKernel 融合注意力分数计算。"""

import cupy as cp
import torch
import math
import time

D = 128
GROUP_SIZE = 32
NUM_GROUPS = D // GROUP_SIZE

# 融合注意力分数 kernel: 从 4-bit packed 格式直接计算 score
fused_score_kernel = cp.RawKernel(r'''
extern "C" __global__
void fused_score_4bit(
    const float* __restrict__ q_rotated,
    const unsigned char* __restrict__ packed_keys,
    const float* __restrict__ gmins,
    const float* __restrict__ gscales,
    const float* __restrict__ radius,
    float* __restrict__ scores,
    int S, int D, int group_size, int num_groups, float scale
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;

    float dot = 0.0f;
    int packed_dim = D / 2;

    for (int g = 0; g < num_groups; g++) {
        float gmin = gmins[s * num_groups + g];
        float gsc = gscales[s * num_groups + g];
        int g_start = g * group_size;

        for (int i = 0; i < group_size; i += 2) {
            int idx = g_start + i;
            int byte_idx = idx / 2;
            unsigned char packed = packed_keys[s * packed_dim + byte_idx];

            float v0 = (float)(packed & 0x0F) * gsc + gmin;
            float v1 = (float)((packed >> 4) & 0x0F) * gsc + gmin;

            dot += v0 * q_rotated[idx];
            dot += v1 * q_rotated[idx + 1];
        }
    }

    scores[s] = dot * radius[s] * scale;
}
''', 'fused_score_4bit')


def benchmark():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"D={D}, group_size={GROUP_SIZE}\n")

    scale = 1.0 / math.sqrt(D)

    for S in [128, 512, 2048, 4096, 8192]:
        # 准备数据
        q_rot = cp.random.randn(D).astype(cp.float32)
        packed = cp.random.randint(0, 256, (S, D // 2), dtype=cp.uint8)
        gmins = cp.random.randn(S, NUM_GROUPS).astype(cp.float32) * 0.1
        gscales = cp.abs(cp.random.randn(S, NUM_GROUPS).astype(cp.float32)) * 0.01
        radius = cp.abs(cp.random.randn(S).astype(cp.float32)) * 10
        scores = cp.zeros(S, dtype=cp.float32)

        threads = 256
        blocks = (S + threads - 1) // threads

        # Warmup
        for _ in range(20):
            fused_score_kernel((blocks,), (threads,),
                (q_rot, packed, gmins, gscales, radius, scores,
                 S, D, GROUP_SIZE, NUM_GROUPS, cp.float32(scale)))
        cp.cuda.Stream.null.synchronize()

        # Benchmark
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        times = []
        for _ in range(100):
            start.record()
            fused_score_kernel((blocks,), (threads,),
                (q_rot, packed, gmins, gscales, radius, scores,
                 S, D, GROUP_SIZE, NUM_GROUPS, cp.float32(scale)))
            end.record()
            end.synchronize()
            times.append(cp.cuda.get_elapsed_time(start, end))

        times.sort()
        median_ms = times[len(times) // 2]

        # 对比: PyTorch matmul (标准注意力的 score 计算部分)
        q_torch = torch.randn(1, D, dtype=torch.float32, device="cuda")
        k_torch = torch.randn(S, D, dtype=torch.float16, device="cuda")

        for _ in range(20):
            _ = q_torch @ k_torch.float().T
        torch.cuda.synchronize()

        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        torch_times = []
        for _ in range(100):
            t_start.record()
            _ = q_torch @ k_torch.float().T
            t_end.record()
            torch.cuda.synchronize()
            torch_times.append(t_start.elapsed_time(t_end))
        torch_times.sort()
        torch_median = torch_times[len(torch_times) // 2]

        speedup = torch_median / median_ms if median_ms > 0 else 0
        print(f"  S={S:>5}: fused={median_ms:.4f}ms, matmul={torch_median:.4f}ms, speedup={speedup:.2f}x")


if __name__ == "__main__":
    benchmark()
