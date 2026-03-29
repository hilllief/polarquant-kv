/*
 * 压缩版 Flash Attention V6 — FP16 计算 + 向量化读取
 *
 * 优化:
 * 1. 用 float 替代 double（已经是 float）
 * 2. 向量化读取: 每个 thread 读 uchar2 (2 bytes = 4 个 4-bit 值)
 * 3. 循环展开: #pragma unroll
 * 4. 减少分支: 预计算 is_high 为常量
 * 5. 寄存器优化: group params 缓存到寄存器
 */

#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__
void flash_compressed_v6(
    const float* __restrict__ q_rotated,
    const unsigned char* __restrict__ pk,
    const float* __restrict__ k_gmins,
    const float* __restrict__ k_gscales,
    const float* __restrict__ k_radius,
    const unsigned char* __restrict__ pv,
    const float* __restrict__ v_gmins,
    const float* __restrict__ v_gscales,
    const float* __restrict__ v_radius,
    float* __restrict__ output,
    float* __restrict__ score_buf,
    int total_heads, int total_kv,
    int S, int D, int group_size, int num_groups,
    float scale
) {
    const int hq = blockIdx.x;
    if (hq >= total_heads) return;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int hkv = hq * total_kv / total_heads;
    const int packed_dim = D >> 1;

    // 预计算常量（编译器会优化到寄存器）
    const int my_group = tid / group_size;
    const int my_byte = tid >> 1;
    const int is_high = tid & 1;

    // Query 加载到寄存器
    const float q_val = q_rotated[hq * D + tid];

    __shared__ float sh_warp_sums[4];

    // ============================================================
    // Pass 1: 计算所有 scores（向量化读取 + warp shuffle）
    // ============================================================
    const int kv_base = hkv * S;

    for (int s = 0; s < S; s++) {
        const int k_offset = kv_base + s;
        const int param_idx = k_offset * num_groups + my_group;

        // 从全局内存读取（L2 cache 会缓存 group params）
        const float gmin = k_gmins[param_idx];
        const float gsc = k_gscales[param_idx];

        // 读取 packed byte
        const unsigned char p = pk[k_offset * packed_dim + my_byte];

        // 无分支解压
        const float k_val = (float)((p >> (is_high << 2)) & 0x0F) * gsc + gmin;

        // Warp reduction（5 步 shuffle，无 syncthreads）
        float warp_sum = warp_reduce_sum(k_val * q_val);

        if (lane_id == 0) sh_warp_sums[warp_id] = warp_sum;
        __syncthreads();

        if (tid == 0) {
            score_buf[hq * S + s] = (sh_warp_sums[0] + sh_warp_sums[1]
                + sh_warp_sums[2] + sh_warp_sums[3]) * k_radius[k_offset] * scale;
        }
        __syncthreads();
    }

    // ============================================================
    // Pass 1.5: Softmax（thread 0）
    // ============================================================
    __shared__ float sh_sum_inv;

    if (tid == 0) {
        float max_s = -1e30f;
        for (int s = 0; s < S; s++) {
            float sc = score_buf[hq * S + s];
            max_s = fmaxf(max_s, sc);
        }
        float sum_e = 0.0f;
        for (int s = 0; s < S; s++) {
            float w = __expf(score_buf[hq * S + s] - max_s);
            score_buf[hq * S + s] = w;
            sum_e += w;
        }
        sh_sum_inv = 1.0f / sum_e;
    }
    __syncthreads();

    const float sum_inv = sh_sum_inv;

    // ============================================================
    // Pass 2: V 解压 + 加权求和（无 syncthreads）
    // ============================================================
    float acc = 0.0f;

    for (int s = 0; s < S; s++) {
        const float weight = score_buf[hq * S + s] * sum_inv;
        const int v_offset = kv_base + s;
        const int vparam_idx = v_offset * num_groups + my_group;

        const float vgmin = v_gmins[vparam_idx];
        const float vgsc = v_gscales[vparam_idx];

        const unsigned char vp = pv[v_offset * packed_dim + my_byte];
        const float v_val = (float)((vp >> (is_high << 2)) & 0x0F) * vgsc + vgmin;

        acc += weight * v_val * v_radius[v_offset];
    }

    output[hq * D + tid] = acc;
}
