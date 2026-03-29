/*
 * PolarQuant-KV 纯 CUDA 实现（不依赖 torch headers）。
 * 通过 pybind11 暴露给 Python，接收 raw GPU 指针。
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>

// ============================================================
// Kernel: 融合 4-bit 注意力（score + online softmax + V 加权）
// 每个 block 处理一个 query head
// ============================================================

extern "C" __global__
void fused_attention_4bit_kernel(
    const float* __restrict__ q_rotated,     // [Hq, D]
    const unsigned char* __restrict__ pk,    // [Hkv, S, D/2]
    const float* __restrict__ k_gmins,       // [Hkv, S, G]
    const float* __restrict__ k_gscales,     // [Hkv, S, G]
    const float* __restrict__ k_radius,      // [Hkv, S]
    const unsigned char* __restrict__ pv,    // [Hkv, S, D/2]
    const float* __restrict__ v_gmins,       // [Hkv, S, G]
    const float* __restrict__ v_gscales,     // [Hkv, S, G]
    const float* __restrict__ v_radius,      // [Hkv, S]
    float* __restrict__ output,              // [Hq, D]
    int Hq, int Hkv, int S, int D,
    int group_size, int num_groups, float scale
) {
    int hq = blockIdx.x;
    if (hq >= Hq) return;
    int tid = threadIdx.x;
    int hkv = hq * Hkv / Hq;
    int packed_dim = D / 2;

    extern __shared__ float smem[];
    float* acc = smem;
    float* reduce_buf = smem + D;
    // shared scalars
    float* sh_score = smem + D + blockDim.x;
    float* sh_max = smem + D + blockDim.x + 1;
    float* sh_sum = smem + D + blockDim.x + 2;
    float* sh_corr = smem + D + blockDim.x + 3;

    if (tid < D) acc[tid] = 0.0f;
    if (tid == 0) { *sh_max = -1e30f; *sh_sum = 0.0f; }
    __syncthreads();

    for (int s = 0; s < S; s++) {
        // --- Score: parallel dot product ---
        float my_dot = 0.0f;
        for (int i = tid; i < D; i += blockDim.x) {
            int g = i / group_size;
            float gmin = k_gmins[(hkv * S + s) * num_groups + g];
            float gsc = k_gscales[(hkv * S + s) * num_groups + g];
            int byte_idx = i / 2;
            unsigned char p = pk[(hkv * S + s) * packed_dim + byte_idx];
            float val = (i % 2 == 0) ?
                (float)(p & 0x0F) * gsc + gmin :
                (float)((p >> 4) & 0x0F) * gsc + gmin;
            my_dot += val * q_rotated[hq * D + i];
        }
        reduce_buf[tid] = my_dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduce_buf[tid] += reduce_buf[tid + stride];
            __syncthreads();
        }

        // --- Online softmax ---
        if (tid == 0) {
            float score = reduce_buf[0] * k_radius[hkv * S + s] * scale;
            float old_max = *sh_max;
            float new_max = fmaxf(old_max, score);
            float corr = expf(old_max - new_max);
            *sh_sum = (*sh_sum) * corr + expf(score - new_max);
            *sh_max = new_max;
            *sh_score = score;
            *sh_corr = corr;
        }
        __syncthreads();

        float corr = *sh_corr;
        float weight = expf(*sh_score - *sh_max);

        // --- V 解压 + 加权累加 ---
        if (tid < D) {
            acc[tid] *= corr;
            int g = tid / group_size;
            float vgmin = v_gmins[(hkv * S + s) * num_groups + g];
            float vgsc = v_gscales[(hkv * S + s) * num_groups + g];
            int byte_idx = tid / 2;
            unsigned char vp = pv[(hkv * S + s) * packed_dim + byte_idx];
            float vval = (tid % 2 == 0) ?
                (float)(vp & 0x0F) * vgsc + vgmin :
                (float)((vp >> 4) & 0x0F) * vgsc + vgmin;
            acc[tid] += weight * vval * v_radius[hkv * S + s];
        }
        __syncthreads();
    }

    // 归一化 + 逆旋转（逆旋转在 host 端做）
    if (tid < D) {
        output[hq * D + tid] = acc[tid] / (*sh_sum);
    }
}
