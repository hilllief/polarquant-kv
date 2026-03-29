/*
 * 压缩版 Flash Attention V2
 *
 * 关键优化: 每个 thread 独立处理所有 key tokens 的部分维度，
 * 只在最后做一次 reduction。大幅减少 __syncthreads 次数。
 *
 * 并行策略:
 *   blockDim.x = D (128 threads, 每个 thread 负责一个维度)
 *   gridDim.x = total_heads
 *
 * 每个 thread:
 *   1. 加载 q[my_dim]
 *   2. 遍历所有 key tokens:
 *      - 从全局内存读 packed_key[s][my_dim/2]
 *      - 解压得到 direction[my_dim]
 *      - partial_score[my_dim] = direction * q[my_dim]
 *      - reduction 得到 full score (需要 syncthreads)
 *      - online softmax 更新
 *      - 解压 V[s][my_dim], 累加到 acc[my_dim]
 *   3. 归一化 acc[my_dim]
 */

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void flash_compressed_v2(
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
    int dim = threadIdx.x;  // 每个 thread 负责一个维度
    if (dim >= D) return;

    int hkv = hq * total_kv / total_heads;
    int packed_dim = D / 2;
    int my_group = dim / group_size;
    int my_byte = dim / 2;
    int is_high_nibble = dim % 2;

    // 加载 query 到寄存器
    float q_val = q_rotated[hq * D + dim];

    // Online softmax 状态（每个 thread 独立维护 partial）
    // 但 softmax 需要 full score，所以还是需要 reduction
    // 优化: 用 shared memory 做 warp-level reduction

    extern __shared__ float smem[];
    float* reduce_buf = smem;           // [blockDim.x]
    float* sh_state = smem + blockDim.x; // [3]: max, sum, score

    float acc_val = 0.0f;  // 每个 thread 的累加器（一个维度）

    if (dim == 0) {
        sh_state[0] = -1e30f;  // max
        sh_state[1] = 0.0f;    // sum
    }
    __syncthreads();

    for (int s = 0; s < S; s++) {
        // --- 解压 K[s][dim] 并计算 partial score ---
        int k_offset = (hkv * S + s);
        float gmin = k_gmins[k_offset * num_groups + my_group];
        float gsc = k_gscales[k_offset * num_groups + my_group];

        unsigned char p = pk[k_offset * packed_dim + my_byte];
        float k_val;
        if (is_high_nibble)
            k_val = (float)((p >> 4) & 0x0F) * gsc + gmin;
        else
            k_val = (float)(p & 0x0F) * gsc + gmin;

        float partial = k_val * q_val;

        // --- Reduction: sum partial scores across all dims ---
        reduce_buf[dim] = partial;
        __syncthreads();

        // Tree reduction (只需 log2(D)=7 步)
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (dim < stride) reduce_buf[dim] += reduce_buf[dim + stride];
            __syncthreads();
        }

        // --- Online softmax (thread 0 更新，广播) ---
        if (dim == 0) {
            float score = reduce_buf[0] * k_radius[k_offset] * scale;
            float old_max = sh_state[0];
            float new_max = fmaxf(old_max, score);
            float corr = expf(old_max - new_max);
            sh_state[1] = sh_state[1] * corr + expf(score - new_max);
            sh_state[0] = new_max;
            sh_state[2] = score;  // broadcast score
            reduce_buf[0] = corr; // broadcast correction
        }
        __syncthreads();

        float corr = reduce_buf[0];
        float weight = expf(sh_state[2] - sh_state[0]);

        // --- 解压 V[s][dim] + 加权累加 ---
        int v_offset = (hkv * S + s);
        float vgmin = v_gmins[v_offset * num_groups + my_group];
        float vgsc = v_gscales[v_offset * num_groups + my_group];

        unsigned char vp = pv[v_offset * packed_dim + my_byte];
        float v_val;
        if (is_high_nibble)
            v_val = (float)((vp >> 4) & 0x0F) * vgsc + vgmin;
        else
            v_val = (float)(vp & 0x0F) * vgsc + vgmin;

        acc_val = acc_val * corr + weight * v_val * v_radius[v_offset];
        __syncthreads();
    }

    // 归一化并写回
    output[hq * D + dim] = acc_val / sh_state[1];
}
