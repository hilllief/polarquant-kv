/*
 * 压缩版 Flash Attention V4 — Shared Memory Tiling
 *
 * V3 的基础上增加:
 * 1. 分块加载 BLOCK_K 个 token 的 packed data 到 shared memory
 * 2. Group params 预加载到 shared memory（32 个 thread 共享）
 * 3. 减少全局内存访问次数
 *
 * 内存访问模式:
 *   V3: 每个 token 每个 thread 读 1 byte (packed) + 2 float (gmin,gscale) = 9 bytes
 *       128 threads × S tokens = 128 × S × 9 bytes 全局内存读
 *   V4: 每块 BLOCK_K tokens，先加载到 shared memory，再从 shared memory 读
 *       全局内存读: BLOCK_K × (D/2 + G*2 + 1) bytes per block
 *       shared memory 读: 128 × BLOCK_K × 1 byte (快 ~100x)
 */

#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_K 32  // 每次处理 32 个 key tokens

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__
void flash_compressed_v4(
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
    int total_heads, int total_kv,
    int S, int D, int group_size, int num_groups,
    float scale
) {
    int hq = blockIdx.x;
    if (hq >= total_heads) return;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int hkv = hq * total_kv / total_heads;
    int packed_dim = D / 2;

    int my_group = tid / group_size;
    int my_byte = tid / 2;
    int is_high = tid % 2;

    float q_val = (tid < D) ? q_rotated[hq * D + tid] : 0.0f;

    // Shared memory layout:
    // [0]: sh_warp_sums[4]
    // [4]: sh_max, sh_sum_exp, sh_score, sh_corr
    // [8]: k_gmin_shared[BLOCK_K * num_groups]  — group params 缓存
    // [8 + BLOCK_K*G]: k_gscale_shared[BLOCK_K * num_groups]
    // [8 + 2*BLOCK_K*G]: k_radius_shared[BLOCK_K]
    // 后面: v 的 group params
    extern __shared__ float smem[];
    float* sh_warp_sums = smem;
    float* sh_max = smem + 4;
    float* sh_sum_exp = smem + 5;
    float* sh_score = smem + 6;
    float* sh_corr = smem + 7;

    int params_offset = 8;
    float* k_gmin_sh = smem + params_offset;
    float* k_gscale_sh = k_gmin_sh + BLOCK_K * num_groups;
    float* k_radius_sh = k_gscale_sh + BLOCK_K * num_groups;
    float* v_gmin_sh = k_radius_sh + BLOCK_K;
    float* v_gscale_sh = v_gmin_sh + BLOCK_K * num_groups;
    float* v_radius_sh = v_gscale_sh + BLOCK_K * num_groups;

    // 用 unsigned char shared memory 存 packed data
    // 放在 float shared memory 之后
    int float_smem_size = params_offset + 2 * BLOCK_K * num_groups + BLOCK_K
                        + 2 * BLOCK_K * num_groups + BLOCK_K;
    unsigned char* pk_sh = (unsigned char*)(smem + float_smem_size);
    unsigned char* pv_sh = pk_sh + BLOCK_K * packed_dim;

    float acc = 0.0f;
    if (tid == 0) { *sh_max = -1e30f; *sh_sum_exp = 0.0f; }
    __syncthreads();

    for (int block_start = 0; block_start < S; block_start += BLOCK_K) {
        int block_end = min(block_start + BLOCK_K, S);
        int block_len = block_end - block_start;

        // --- 协作加载: 所有 thread 一起加载这块数据到 shared memory ---
        // 加载 packed keys: BLOCK_K * packed_dim bytes
        int total_pk_bytes = block_len * packed_dim;
        for (int i = tid; i < total_pk_bytes; i += blockDim.x) {
            int local_s = i / packed_dim;
            int byte_off = i % packed_dim;
            int global_s = block_start + local_s;
            pk_sh[i] = pk[(hkv * S + global_s) * packed_dim + byte_off];
        }
        // 加载 packed values
        for (int i = tid; i < total_pk_bytes; i += blockDim.x) {
            int local_s = i / packed_dim;
            int byte_off = i % packed_dim;
            int global_s = block_start + local_s;
            pv_sh[i] = pv[(hkv * S + global_s) * packed_dim + byte_off];
        }
        // 加载 group params + radius
        int total_params = block_len * num_groups;
        for (int i = tid; i < total_params; i += blockDim.x) {
            int local_s = i / num_groups;
            int g = i % num_groups;
            int global_s = block_start + local_s;
            k_gmin_sh[i] = k_gmins[(hkv * S + global_s) * num_groups + g];
            k_gscale_sh[i] = k_gscales[(hkv * S + global_s) * num_groups + g];
            v_gmin_sh[i] = v_gmins[(hkv * S + global_s) * num_groups + g];
            v_gscale_sh[i] = v_gscales[(hkv * S + global_s) * num_groups + g];
        }
        for (int i = tid; i < block_len; i += blockDim.x) {
            k_radius_sh[i] = k_radius[hkv * S + block_start + i];
            v_radius_sh[i] = v_radius[hkv * S + block_start + i];
        }
        __syncthreads();  // 确保数据加载完成

        // --- 处理这块中的每个 token（从 shared memory 读取）---
        for (int local_s = 0; local_s < block_len; local_s++) {
            // 解压 K 并计算 partial dot（从 shared memory 读）
            float gmin = k_gmin_sh[local_s * num_groups + my_group];
            float gsc = k_gscale_sh[local_s * num_groups + my_group];

            unsigned char p = pk_sh[local_s * packed_dim + my_byte];
            float k_val;
            if (is_high)
                k_val = (float)((p >> 4) & 0x0F) * gsc + gmin;
            else
                k_val = (float)(p & 0x0F) * gsc + gmin;

            float partial = k_val * q_val;

            // Warp reduction
            float warp_sum = warp_reduce_sum(partial);
            if (lane_id == 0) sh_warp_sums[warp_id] = warp_sum;
            __syncthreads();

            // Thread 0 汇总
            if (tid == 0) {
                float total_dot = sh_warp_sums[0] + sh_warp_sums[1]
                                + sh_warp_sums[2] + sh_warp_sums[3];
                float score = total_dot * k_radius_sh[local_s] * scale;

                float old_max = *sh_max;
                float new_max = fmaxf(old_max, score);
                float corr = expf(old_max - new_max);
                *sh_sum_exp = (*sh_sum_exp) * corr + expf(score - new_max);
                *sh_max = new_max;
                *sh_score = score;
                *sh_corr = corr;
            }
            __syncthreads();

            float corr = *sh_corr;
            float weight = expf(*sh_score - *sh_max);

            // 解压 V 并累加（从 shared memory 读）
            float vgmin = v_gmin_sh[local_s * num_groups + my_group];
            float vgsc = v_gscale_sh[local_s * num_groups + my_group];

            unsigned char vp = pv_sh[local_s * packed_dim + my_byte];
            float v_val;
            if (is_high)
                v_val = (float)((vp >> 4) & 0x0F) * vgsc + vgmin;
            else
                v_val = (float)(vp & 0x0F) * vgsc + vgmin;

            acc = acc * corr + weight * v_val * v_radius_sh[local_s];
        }
        __syncthreads();  // 确保所有 thread 处理完这块再加载下一块
    }

    if (tid < D) {
        output[hq * D + tid] = acc / (*sh_sum_exp);
    }
}
