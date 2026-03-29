/*
 * 压缩版 Flash Attention V5 — 多 token 批处理
 *
 * 关键优化: 每个 thread 在寄存器中处理 TOKENS_PER_THREAD 个 token，
 * 只在每批结束时做一次 warp reduction + softmax 更新。
 * syncthreads 次数从 S 降到 S / TOKENS_PER_THREAD。
 *
 * 但 online softmax 需要每个 token 的 full score...
 * 所以改为: 每个 thread 处理所有 token 的一个维度的 partial score，
 * 累加到寄存器，然后分批做 reduction。
 *
 * 实际策略: 2-pass
 *   Pass 1: 算所有 scores（每个 thread 算 partial，warp reduce）
 *           scores 存到 shared memory（需要 S 个 float）
 *   Pass 2: softmax + V 加权（从 shared memory 读 weights）
 *
 * 对于 S <= 16384，shared memory 需要 64KB（刚好是 sm_120 的上限）
 * 对于更大的 S，分块处理。
 */

#include <cuda_runtime.h>
#include <math.h>

#define MAX_S_PER_BLOCK 8192  // shared memory 限制

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__
void flash_compressed_v5(
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
    float* __restrict__ score_buf,  // [total_heads, S] 临时 buffer
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

    __shared__ float sh_warp_sums[4];

    // ============================================================
    // Pass 1: 计算所有 scores
    // ============================================================
    for (int s = 0; s < S; s++) {
        int k_offset = hkv * S + s;
        float gmin = k_gmins[k_offset * num_groups + my_group];
        float gsc = k_gscales[k_offset * num_groups + my_group];

        unsigned char p = pk[k_offset * packed_dim + my_byte];
        float k_val = is_high ?
            (float)((p >> 4) & 0x0F) * gsc + gmin :
            (float)(p & 0x0F) * gsc + gmin;

        float partial = k_val * q_val;
        float warp_sum = warp_reduce_sum(partial);

        if (lane_id == 0) sh_warp_sums[warp_id] = warp_sum;
        __syncthreads();

        if (tid == 0) {
            float total = sh_warp_sums[0] + sh_warp_sums[1]
                        + sh_warp_sums[2] + sh_warp_sums[3];
            score_buf[hq * S + s] = total * k_radius[k_offset] * scale;
        }
        __syncthreads();
    }

    // ============================================================
    // Pass 1.5: Softmax（thread 0 计算 weights）
    // ============================================================
    __shared__ float sh_max_val;
    __shared__ float sh_sum_val;

    if (tid == 0) {
        float max_s = -1e30f;
        for (int s = 0; s < S; s++) {
            float sc = score_buf[hq * S + s];
            if (sc > max_s) max_s = sc;
        }
        sh_max_val = max_s;

        float sum_e = 0.0f;
        for (int s = 0; s < S; s++) {
            float w = expf(score_buf[hq * S + s] - max_s);
            score_buf[hq * S + s] = w;  // 复用 buffer 存 weights
            sum_e += w;
        }
        sh_sum_val = sum_e;
    }
    __syncthreads();

    float sum_e = sh_sum_val;

    // ============================================================
    // Pass 2: V 解压 + 加权求和
    // ============================================================
    float acc = 0.0f;

    for (int s = 0; s < S; s++) {
        float weight = score_buf[hq * S + s] / sum_e;

        int v_offset = hkv * S + s;
        float vgmin = v_gmins[v_offset * num_groups + my_group];
        float vgsc = v_gscales[v_offset * num_groups + my_group];

        unsigned char vp = pv[v_offset * packed_dim + my_byte];
        float v_val = is_high ?
            (float)((vp >> 4) & 0x0F) * vgsc + vgmin :
            (float)(vp & 0x0F) * vgsc + vgmin;

        acc += weight * v_val * v_radius[v_offset];
    }

    if (tid < D) {
        output[hq * D + tid] = acc;
    }
}
