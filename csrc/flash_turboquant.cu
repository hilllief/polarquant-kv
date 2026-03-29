/*
 * TurboQuant 融合注意力 Kernel
 *
 * 创新点：Lloyd-Max codebook 替代 per-group min/max
 * - Centroid table (16 floats for 4-bit) 加载到 shared memory
 * - 每个 token 只读 packed_indices + radius（零 group params）
 * - 带宽需求降低 20%+
 *
 * 数据格式：
 *   K/V 存储: packed_indices[S, D/2] (uint8) + radius[S] (float)
 *   Codebook: centroids[2^n_bits] (float), 全局共享
 */

#include <cuda_runtime.h>
#include <math.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void turboquant_attention_kernel(
    const float* __restrict__ q_rot,         // [total_heads, D]
    const unsigned char* __restrict__ pk,    // [total_kv, S, D/2]
    const float* __restrict__ k_radius,      // [total_kv, S]
    const unsigned char* __restrict__ pv,    // [total_kv, S, D/2]
    const float* __restrict__ v_radius,      // [total_kv, S]
    const float* __restrict__ k_codebook,    // [n_levels] centroids for K
    const float* __restrict__ v_codebook,    // [n_levels] centroids for V
    float* __restrict__ output,              // [total_heads, D]
    float* __restrict__ score_buf,           // [total_heads, S]
    int total_heads, int total_kv, int S, int D,
    int n_levels, float scale
) {
    int hq = blockIdx.x;
    if (hq >= total_heads) return;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int hkv = hq * total_kv / total_heads;
    int packed_dim = D >> 1;
    int my_byte = tid >> 1;
    int is_high = tid & 1;

    // 加载 codebook 到 shared memory（所有 token 共享）
    __shared__ float k_cb[16];  // 最多 16 levels (4-bit)
    __shared__ float v_cb[16];
    __shared__ float sh[4];     // warp sums
    __shared__ float sh_inv;

    if (tid < n_levels) {
        k_cb[tid] = k_codebook[tid];
        v_cb[tid] = v_codebook[tid];
    }
    __syncthreads();

    float q_val = q_rot[hq * D + tid];

    // Pass 1: scores（从 codebook 查表解压）
    for (int s = 0; s < S; s++) {
        int off = hkv * S + s;
        unsigned char p = pk[off * packed_dim + my_byte];
        int idx = (p >> (is_high << 2)) & (n_levels - 1);
        float k_val = k_cb[idx];  // 查表！不需要 gmin/gscale

        float ws = warp_reduce_sum(k_val * q_val);
        if (lane_id == 0) sh[warp_id] = ws;
        __syncthreads();
        if (tid == 0)
            score_buf[hq * S + s] = (sh[0]+sh[1]+sh[2]+sh[3]) * k_radius[off] * scale;
        __syncthreads();
    }

    // Softmax
    if (tid == 0) {
        float mx = -1e30f;
        for (int s = 0; s < S; s++) mx = fmaxf(mx, score_buf[hq*S+s]);
        float se = 0.0f;
        for (int s = 0; s < S; s++) {
            float w = __expf(score_buf[hq*S+s] - mx);
            score_buf[hq*S+s] = w;
            se += w;
        }
        sh_inv = 1.0f / se;
    }
    __syncthreads();

    // Pass 2: V 加权求和（同样用 codebook 查表）
    float acc = 0.0f;
    float inv = sh_inv;
    for (int s = 0; s < S; s++) {
        float w = score_buf[hq*S+s] * inv;
        int off = hkv*S+s;
        unsigned char vp = pv[off*packed_dim+my_byte];
        int vidx = (vp >> (is_high<<2)) & (n_levels - 1);
        float v_val = v_cb[vidx];  // 查表！
        acc += w * v_val * v_radius[off];
    }
    output[hq*D+tid] = acc;
}


// ============================================================
// 压缩 kernel：旋转 + 归一化 + codebook 量化
// ============================================================
__global__ void turboquant_compress_kernel(
    const float* __restrict__ input,      // [N, D]
    const float* __restrict__ R,          // [D, D]
    const float* __restrict__ codebook,   // [n_levels]
    float* __restrict__ radius_out,       // [N]
    unsigned char* __restrict__ packed_out,// [N, D/2]
    int N, int D, int n_levels
) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= N) return;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* rotated = smem;
    float* direction = smem + D;

    // 加载 codebook 到 shared memory
    __shared__ float cb[16];
    if (tid < n_levels) cb[tid] = codebook[tid];

    // 旋转: q_rot = input @ R^T
    if (tid < D) {
        float sum = 0.0f;
        for (int j = 0; j < D; j++)
            sum += input[vec_idx * D + j] * R[tid * D + j];
        rotated[tid] = sum;
    }
    __syncthreads();

    // 范数
    __shared__ float reduce_buf[256];
    reduce_buf[tid] = (tid < D) ? rotated[tid] * rotated[tid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float radius = sqrtf(reduce_buf[0]);
    if (tid == 0) radius_out[vec_idx] = radius;

    // 归一化
    if (tid < D)
        direction[tid] = (radius > 1e-30f) ? (rotated[tid] / radius) : 0.0f;
    __syncthreads();

    // Codebook 量化 + 4-bit pack
    if (tid < D / 2) {
        int d0 = tid * 2;
        int d1 = tid * 2 + 1;
        float v0 = direction[d0];
        float v1 = direction[d1];

        // 找最近的 centroid
        int best0 = 0, best1 = 0;
        float min_dist0 = 1e30f, min_dist1 = 1e30f;
        for (int i = 0; i < n_levels; i++) {
            float d = fabsf(v0 - cb[i]);
            if (d < min_dist0) { min_dist0 = d; best0 = i; }
            d = fabsf(v1 - cb[i]);
            if (d < min_dist1) { min_dist1 = d; best1 = i; }
        }

        packed_out[vec_idx * (D/2) + tid] = (unsigned char)(best0 | (best1 << 4));
    }
}

// ============================================================
// 全 C++ 入口
// ============================================================
void turboquant_full_attention(
    int64_t q_ptr, int64_t R_ptr,
    int64_t pk_ptr, int64_t kr_ptr,
    int64_t pv_ptr, int64_t vr_ptr,
    int64_t k_cb_ptr, int64_t v_cb_ptr,
    int64_t out_ptr, int64_t q_rot_ptr, int64_t score_ptr, int64_t attn_out_ptr,
    int total_heads, int total_kv, int S, int D, int n_levels, float scale
) {
    // Step 1: Q 旋转
    // (复用 rotate kernel)
    extern __global__ void rotate_query_kernel(const float*, const float*, float*, int);
    // 简化：直接在这里内联
    {
        auto kernel = [](const float* Q, const float* R, float* Q_rot, int D) {
        };
        // 用 lambda 不行，直接写 kernel launch
    }

    // 简化方案：3 个 kernel 串联
    // Kernel 1: rotate (在 Python 端做 q @ R^T)
    // Kernel 2: attention
    int smem = (16 + 16 + 4 + 1) * sizeof(float);
    turboquant_attention_kernel<<<total_heads, D, smem>>>(
        (const float*)q_rot_ptr,
        (const unsigned char*)pk_ptr, (const float*)kr_ptr,
        (const unsigned char*)pv_ptr, (const float*)vr_ptr,
        (const float*)k_cb_ptr, (const float*)v_cb_ptr,
        (float*)attn_out_ptr, (float*)score_ptr,
        total_heads, total_kv, S, D, n_levels, scale
    );
    // Kernel 3: inverse rotate (在 Python 端做 output @ R)
}

void turboquant_compress(
    int64_t input_ptr, int64_t R_ptr, int64_t cb_ptr,
    int64_t radius_ptr, int64_t packed_ptr,
    int N, int D, int n_levels
) {
    int smem = (2 * D + 256) * sizeof(float);
    int block = 128;
    turboquant_compress_kernel<<<N, block, smem>>>(
        (const float*)input_ptr, (const float*)R_ptr, (const float*)cb_ptr,
        (float*)radius_ptr, (unsigned char*)packed_ptr,
        N, D, n_levels
    );
}

PYBIND11_MODULE(polarquant_turboquant, m) {
    m.def("turboquant_full_attention", &turboquant_full_attention);
    m.def("turboquant_compress", &turboquant_compress);
}
