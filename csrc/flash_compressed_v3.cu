/*
 * 压缩版 Flash Attention V3 — Warp Shuffle 优化
 *
 * 关键优化:
 * 1. 用 __shfl_down_sync 替代 shared memory reduction（无 syncthreads）
 * 2. 每个 warp (32 threads) 处理 32 个维度的 partial dot product
 * 3. 4 个 warp 覆盖 D=128 维度
 * 4. Warp 0 的 lane 0 做 softmax 更新，通过 __shfl_sync 广播
 * 5. 向量化内存访问: 每个 thread 读 4 bytes (2 个 4-bit 值)
 *
 * 并行策略:
 *   grid = (total_heads,)
 *   block = 128 threads = 4 warps × 32 lanes
 *   每个 thread 负责 1 个维度 (dim = threadIdx.x)
 */

#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction: 求和
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;  // 只有 lane 0 有正确结果
}

extern "C" __global__
void flash_compressed_v3(
    const float* __restrict__ q_rotated,     // [total_heads, D]
    const unsigned char* __restrict__ pk,    // [total_kv, S, D/2]
    const float* __restrict__ k_gmins,       // [total_kv, S, G]
    const float* __restrict__ k_gscales,     // [total_kv, S, G]
    const float* __restrict__ k_radius,      // [total_kv, S]
    const unsigned char* __restrict__ pv,    // [total_kv, S, D/2]
    const float* __restrict__ v_gmins,       // [total_kv, S, G]
    const float* __restrict__ v_gscales,     // [total_kv, S, G]
    const float* __restrict__ v_radius,      // [total_kv, S]
    float* __restrict__ output,              // [total_heads, D]
    int total_heads, int total_kv,
    int S, int D, int group_size, int num_groups,
    float scale
) {
    int hq = blockIdx.x;
    if (hq >= total_heads) return;

    int tid = threadIdx.x;       // 0..127, 每个 thread 负责一个维度
    int warp_id = tid / 32;      // 0..3
    int lane_id = tid % 32;      // 0..31
    int hkv = hq * total_kv / total_heads;
    int packed_dim = D / 2;

    // 预计算: 我的维度对应的 group 和 byte 位置
    int my_group = tid / group_size;
    int my_byte = tid / 2;
    int is_high = tid % 2;

    // 加载 query 到寄存器（只做一次）
    float q_val = (tid < D) ? q_rotated[hq * D + tid] : 0.0f;

    // Online softmax 状态（用 shared memory 在 warp 间同步）
    __shared__ float sh_warp_sums[4];  // 每个 warp 的 partial sum
    __shared__ float sh_max;
    __shared__ float sh_sum_exp;
    __shared__ float sh_score;
    __shared__ float sh_corr;

    // 累加器（每个 thread 一个维度）
    float acc = 0.0f;

    if (tid == 0) {
        sh_max = -1e30f;
        sh_sum_exp = 0.0f;
    }
    __syncthreads();

    // 遍历所有 key tokens
    for (int s = 0; s < S; s++) {
        int k_offset = hkv * S + s;

        // --- Step 1: 解压 K[s][my_dim] 并计算 partial dot ---
        float gmin = k_gmins[k_offset * num_groups + my_group];
        float gsc = k_gscales[k_offset * num_groups + my_group];

        unsigned char p = pk[k_offset * packed_dim + my_byte];
        float k_val;
        if (is_high)
            k_val = (float)((p >> 4) & 0x0F) * gsc + gmin;
        else
            k_val = (float)(p & 0x0F) * gsc + gmin;

        float partial = k_val * q_val;

        // --- Step 2: Warp-level reduction (无 syncthreads!) ---
        float warp_sum = warp_reduce_sum(partial);

        // 每个 warp 的 lane 0 写入 shared memory
        if (lane_id == 0) sh_warp_sums[warp_id] = warp_sum;
        __syncthreads();  // 只需 1 次 syncthreads（4 个 warp 间同步）

        // --- Step 3: Warp 0 汇总 4 个 warp 的结果 ---
        if (tid == 0) {
            float total_dot = sh_warp_sums[0] + sh_warp_sums[1]
                            + sh_warp_sums[2] + sh_warp_sums[3];
            float score = total_dot * k_radius[k_offset] * scale;

            // Online softmax
            float old_max = sh_max;
            float new_max = fmaxf(old_max, score);
            float corr = expf(old_max - new_max);
            sh_sum_exp = sh_sum_exp * corr + expf(score - new_max);
            sh_max = new_max;
            sh_score = score;
            sh_corr = corr;
        }
        __syncthreads();  // 广播 softmax 状态

        float corr = sh_corr;
        float weight = expf(sh_score - sh_max);

        // --- Step 4: 解压 V[s][my_dim] + 加权累加 ---
        float vgmin = v_gmins[k_offset * num_groups + my_group];
        float vgsc = v_gscales[k_offset * num_groups + my_group];

        unsigned char vp = pv[k_offset * packed_dim + my_byte];
        float v_val;
        if (is_high)
            v_val = (float)((vp >> 4) & 0x0F) * vgsc + vgmin;
        else
            v_val = (float)(vp & 0x0F) * vgsc + vgmin;

        acc = acc * corr + weight * v_val * v_radius[k_offset];
        // 不需要 syncthreads! 每个 thread 独立更新自己的 acc
    }

    // 归一化并写回
    if (tid < D) {
        output[hq * D + tid] = acc / sh_sum_exp;
    }
}
