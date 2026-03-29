"""用 printf 调试 kernel。"""
import cupy as cp
import numpy as np
import math

D = 8  # 用小维度方便调试
GROUP_SIZE = 4
NUM_GROUPS = D // GROUP_SIZE

debug_kernel = cp.RawKernel(r'''
extern "C" __global__
void debug_score(
    const float* q, const unsigned char* pk,
    const float* gm, const float* gs, const float* rad,
    float* scores, int S, int D, int group_size, int num_groups, float scale
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;

    float dot = 0.0f;
    int packed_dim = D / 2;

    printf("s=%d, D=%d, packed_dim=%d, num_groups=%d, group_size=%d\n",
           s, D, packed_dim, num_groups, group_size);

    for (int g = 0; g < num_groups; g++) {
        float gmin = gm[s * num_groups + g];
        float gsc = gs[s * num_groups + g];
        int g_start = g * group_size;

        printf("  g=%d, gmin=%f, gsc=%f, g_start=%d\n", g, gmin, gsc, g_start);

        for (int i = 0; i < group_size; i += 2) {
            int idx = g_start + i;
            int byte_idx = idx / 2;
            unsigned char packed = pk[s * packed_dim + byte_idx];
            unsigned char q0 = packed & 0x0F;
            unsigned char q1 = (packed >> 4) & 0x0F;

            float v0 = (float)q0 * gsc + gmin;
            float v1 = (float)q1 * gsc + gmin;

            printf("    i=%d, byte_idx=%d, packed=0x%02x, q0=%d, q1=%d, v0=%f, v1=%f, qr0=%f, qr1=%f\n",
                   i, byte_idx, packed, q0, q1, v0, v1, q[idx], q[idx+1]);

            dot += v0 * q[idx];
            dot += v1 * q[idx + 1];
        }
    }

    float r = rad[s];
    printf("  dot=%f, rad=%f, scale=%f, score=%f\n", dot, r, scale, dot * r * scale);
    scores[s] = dot * r * scale;
}
''', 'debug_score')

# 构造简单测试数据
q = cp.array([1.0, 0.5, -0.3, 0.2, 0.8, -0.1, 0.4, -0.6], dtype=cp.float32)
# 4-bit packed: 每 byte 存 2 个值
# group 0: values [5, 10, 3, 8] → packed bytes: [0xA5, 0x83]
# group 1: values [7, 2, 12, 1] → packed bytes: [0x27, 0x1C]
pk = cp.array([[0xA5, 0x83, 0x27, 0x1C]], dtype=cp.uint8)
gm = cp.array([[0.0, 0.0]], dtype=cp.float32)  # 2 groups
gs = cp.array([[0.1, 0.1]], dtype=cp.float32)
rad = cp.array([2.0], dtype=cp.float32)
scores = cp.zeros(1, dtype=cp.float32)

debug_kernel((1,), (1,), (q, pk, gm, gs, rad, scores, 1, D, GROUP_SIZE, NUM_GROUPS, cp.float32(0.5)))
cp.cuda.Stream.null.synchronize()

print(f"\nKernel score: {float(scores[0]):.6f}")

# 手动验证
# pk[0] = 0xA5: low=5, high=10 → v0=0.5, v1=1.0
# pk[1] = 0x83: low=3, high=8  → v0=0.3, v1=0.8
# pk[2] = 0x27: low=7, high=2  → v0=0.7, v1=0.2
# pk[3] = 0x1C: low=12,high=1  → v0=1.2, v1=0.1
direction = [0.5, 1.0, 0.3, 0.8, 0.7, 0.2, 1.2, 0.1]
q_np = [1.0, 0.5, -0.3, 0.2, 0.8, -0.1, 0.4, -0.6]
dot = sum(d * qv for d, qv in zip(direction, q_np))
expected = dot * 2.0 * 0.5
print(f"Expected:     {expected:.6f}")
