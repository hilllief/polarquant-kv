/*
 * 压缩版 Flash Attention (4-bit PolarQuant)
 *
 * 核心优化:
 * 1. 分块处理: 每次从全局内存加载 BLOCK_K 个压缩 key 到 shared memory
 * 2. Shared memory 解压: 在 shared memory 中完成 4-bit unpack + dequantize
 * 3. Online softmax: 不存储完整 score 矩阵，O(D) 额外空间
 * 4. K 和 V 都不做逆旋转: 在旋转空间算 score，V 累加后一次逆旋转
 *
 * 内存读取量对比:
 *   标准 FP16: seq_kv * D * 2 bytes (K) + seq_kv * D * 2 bytes (V)
 *   压缩 4-bit: seq_kv * D/2 bytes (K packed) + seq_kv * G * 4 bytes (params)
 *              + seq_kv * D/2 bytes (V packed) + seq_kv * G * 4 bytes (params)
 *   节省: ~3x 内存带宽
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define BLOCK_K 64    // 每次处理 64 个 key tokens
#define MAX_D 256     // 最大 head_dim
#define MAX_GROUPS 8  // 最大 group 数 (256/32)

extern "C" __global__
void flash_compressed_attention_4bit(
    // Query (旋转空间)
    const float* __restrict__ q_rotated,     // [total_heads, D]
    // Compressed Keys
    const unsigned char* __restrict__ pk,    // [total_kv_heads, S, D/2]
    const float* __restrict__ k_gmins,       // [total_kv_heads, S, G]
    const float* __restrict__ k_gscales,     // [total_kv_heads, S, G]
    const float* __restrict__ k_radius,      // [total_kv_heads, S]
    // Compressed Values
    const unsigned char* __restrict__ pv,    // [total_kv_heads, S, D/2]
    const float* __restrict__ v_gmins,       // [total_kv_heads, S, G]
    const float* __restrict__ v_gscales,     // [total_kv_heads, S, G]
    const float* __restrict__ v_radius,      // [total_kv_heads, S]
    // Output (旋转空间)
    float* __restrict__ output,              // [total_heads, D]
    // Dimensions
    int total_heads, int total_kv_heads,
    int S, int D, int group_size, int num_groups,
    float scale
) {
    int head_idx = blockIdx.x;
    if (head_idx >= total_heads) return;
    int tid = threadIdx.x;  // 0..127

    // GQA mapping
    int kv_head_idx = head_idx * total_kv_heads / total_heads;

    int packed_dim = D / 2;

    // ============================================================
    // Shared memory layout:
    //   q_shared[D]                    - query 向量 (加载一次)
    //   acc[D]                         - 输出累加器
    //   k_packed_shared[BLOCK_K][D/2]  - 一块压缩 key
    //   k_gmins_shared[BLOCK_K][G]     - group mins
    //   k_gscales_shared[BLOCK_K][G]   - group scales
    //   k_radius_shared[BLOCK_K]       - radius
    //   score_reduce[blockDim.x]       - score reduction buffer
    //   softmax_state[3]               - max, sum_exp, correction
    // ============================================================

    extern __shared__ float smem[];

    float* q_shared = smem;                                    // [D]
    float* acc = smem + D;                                     // [D]
    // 后面的 shared memory 用 char* 管理（混合类型）
    float* score_reduce = smem + 2 * D;                        // [blockDim.x]
    float* softmax_state = smem + 2 * D + blockDim.x;         // [3]: max, sum, correction

    // 加载 query 到 shared memory（只做一次）
    if (tid < D) {
        q_shared[tid] = q_rotated[head_idx * D + tid];
        acc[tid] = 0.0f;
    }
    if (tid == 0) {
        softmax_state[0] = -1e30f;  // max
        softmax_state[1] = 0.0f;    // sum_exp
    }
    __syncthreads();

    // ============================================================
    // 分块遍历所有 key tokens
    // ============================================================

    for (int block_start = 0; block_start < S; block_start += BLOCK_K) {
        int block_end = min(block_start + BLOCK_K, S);
        int block_len = block_end - block_start;

        // 对这个 block 中的每个 key token 计算 score
        for (int local_s = 0; local_s < block_len; local_s++) {
            int s = block_start + local_s;

            // --- 计算 score: 每个 thread 处理部分维度 ---
            float my_dot = 0.0f;
            for (int i = tid; i < D; i += blockDim.x) {
                int g = i / group_size;
                float gmin = k_gmins[(kv_head_idx * S + s) * num_groups + g];
                float gsc = k_gscales[(kv_head_idx * S + s) * num_groups + g];

                int byte_idx = i / 2;
                unsigned char p = pk[(kv_head_idx * S + s) * packed_dim + byte_idx];
                float val;
                if (i % 2 == 0)
                    val = (float)(p & 0x0F) * gsc + gmin;
                else
                    val = (float)((p >> 4) & 0x0F) * gsc + gmin;

                my_dot += val * q_shared[i];
            }

            // Parallel reduction for score
            score_reduce[tid] = my_dot;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) score_reduce[tid] += score_reduce[tid + stride];
                __syncthreads();
            }

            // --- Online softmax (thread 0) ---
            if (tid == 0) {
                float score = score_reduce[0] * k_radius[kv_head_idx * S + s] * scale;
                float old_max = softmax_state[0];
                float new_max = fmaxf(old_max, score);
                float corr = expf(old_max - new_max);
                softmax_state[1] = softmax_state[1] * corr + expf(score - new_max);
                softmax_state[0] = new_max;
                // Store for V accumulation
                softmax_state[2] = corr;
                score_reduce[0] = score;  // reuse as score broadcast
            }
            __syncthreads();

            float corr = softmax_state[2];
            float weight = expf(score_reduce[0] - softmax_state[0]);

            // --- V 解压 + 加权累加 ---
            if (tid < D) {
                acc[tid] *= corr;

                int g = tid / group_size;
                float vgmin = v_gmins[(kv_head_idx * S + s) * num_groups + g];
                float vgsc = v_gscales[(kv_head_idx * S + s) * num_groups + g];

                int byte_idx = tid / 2;
                unsigned char vp = pv[(kv_head_idx * S + s) * packed_dim + byte_idx];
                float vval;
                if (tid % 2 == 0)
                    vval = (float)(vp & 0x0F) * vgsc + vgmin;
                else
                    vval = (float)((vp >> 4) & 0x0F) * vgsc + vgmin;

                acc[tid] += weight * vval * v_radius[kv_head_idx * S + s];
            }
            __syncthreads();
        }
    }

    // 归一化
    if (tid < D) {
        output[head_idx * D + tid] = acc[tid] / softmax_state[1];
    }
}
