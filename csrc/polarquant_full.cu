/*
 * PolarQuant-KV 全 C++ 实现
 *
 * 一个函数搞定：接收 raw GPU 指针 → 旋转 Q → 融合注意力 → 逆旋转 → 写回
 * Python 端只需要调用一次，零 tensor 准备开销。
 *
 * 包含两个 kernel：
 * 1. rotate_query_kernel: Q @ R^T
 * 2. flash_attn_kernel: 融合注意力（score + softmax + V加权）
 * 3. inverse_rotate_kernel: output @ R
 *
 * 全部在 C++ 端串联调用，一次 Python→C++ 调用完成所有工作。
 */

#include <cuda_runtime.h>
#include <math.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// ============================================================
// Kernel 1: Q 旋转 — q_rot[h,d] = sum_j Q[h,j] * R[j,d]
// grid=(total_heads,), block=(D,)
// ============================================================
__global__ void rotate_query_kernel(
    const float* __restrict__ Q,    // [total_heads, D], FP32
    const float* __restrict__ R,    // [D, D], FP32 (行主序, R^T 的效果通过索引实现)
    float* __restrict__ Q_rot,      // [total_heads, D], FP32
    int D
) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (d >= D) return;

    float sum = 0.0f;
    for (int j = 0; j < D; j++) {
        sum += Q[h * D + j] * R[d * D + j];  // Q @ R^T: sum_j Q[j] * R^T[j,d] = sum_j Q[j] * R[d,j]
    }
    Q_rot[h * D + d] = sum;
}

// ============================================================
// Kernel 2: 融合注意力（V6 优化版）
// ============================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void flash_attn_kernel(
    const float* __restrict__ q_rot,
    const unsigned char* __restrict__ pk,
    const float* __restrict__ km, const float* __restrict__ ks, const float* __restrict__ kr,
    const unsigned char* __restrict__ pv,
    const float* __restrict__ vm, const float* __restrict__ vs, const float* __restrict__ vr,
    float* __restrict__ output,
    float* __restrict__ score_buf,
    int total_heads, int total_kv, int S, int D,
    int group_size, int num_groups, float scale
) {
    int hq = blockIdx.x;
    if (hq >= total_heads) return;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int hkv = hq * total_kv / total_heads;
    int packed_dim = D >> 1;
    int my_group = tid / group_size;
    int my_byte = tid >> 1;
    int is_high = tid & 1;
    float q_val = q_rot[hq * D + tid];
    __shared__ float sh[4];

    // Pass 1: scores
    for (int s = 0; s < S; s++) {
        int off = hkv * S + s;
        float gmin = km[off * num_groups + my_group];
        float gsc = ks[off * num_groups + my_group];
        unsigned char p = pk[off * packed_dim + my_byte];
        float k_val = (float)((p >> (is_high << 2)) & 0x0F) * gsc + gmin;
        float ws = warp_reduce_sum(k_val * q_val);
        if (lane_id == 0) sh[warp_id] = ws;
        __syncthreads();
        if (tid == 0)
            score_buf[hq * S + s] = (sh[0]+sh[1]+sh[2]+sh[3]) * kr[off] * scale;
        __syncthreads();
    }

    // Softmax
    __shared__ float sh_inv;
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

    // Pass 2: V weighted sum
    float acc = 0.0f;
    float inv = sh_inv;
    for (int s = 0; s < S; s++) {
        float w = score_buf[hq*S+s] * inv;
        int off = hkv*S+s;
        float vgmin = vm[off*num_groups+my_group];
        float vgsc = vs[off*num_groups+my_group];
        unsigned char vp = pv[off*packed_dim+my_byte];
        float v_val = (float)((vp >> (is_high<<2)) & 0x0F) * vgsc + vgmin;
        acc += w * v_val * vr[off];
    }
    output[hq*D+tid] = acc;
}

// ============================================================
// Kernel 3: 逆旋转 — out[h,d] = sum_j rotated[h,j] * R[d,j]
// (R 是正交矩阵，R^{-1} = R^T，所以逆旋转 = output @ R)
// ============================================================
__global__ void inverse_rotate_kernel(
    const float* __restrict__ rotated,  // [total_heads, D]
    const float* __restrict__ R,        // [D, D]
    float* __restrict__ output,         // [total_heads, D]
    int D
) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (d >= D) return;

    float sum = 0.0f;
    for (int j = 0; j < D; j++) {
        sum += rotated[h * D + j] * R[j * D + d];  // rotated @ R
    }
    output[h * D + d] = sum;
}

// ============================================================
// 全 C++ 入口：一次调用完成所有工作
// ============================================================
void full_compressed_attention(
    // Q: [total_heads, D], FP32, GPU
    int64_t q_ptr, int64_t R_ptr,
    // Compressed K
    int64_t pk_ptr, int64_t km_ptr, int64_t ks_ptr, int64_t kr_ptr,
    // Compressed V
    int64_t pv_ptr, int64_t vm_ptr, int64_t vs_ptr, int64_t vr_ptr,
    // Output: [total_heads, D], FP32, GPU
    int64_t out_ptr,
    // Temp buffers (预分配)
    int64_t q_rot_ptr, int64_t score_ptr, int64_t attn_out_ptr,
    // Dimensions
    int total_heads, int total_kv, int S, int D,
    int group_size, int num_groups, float scale
) {
    // Step 1: Q 旋转
    rotate_query_kernel<<<total_heads, D>>>(
        (const float*)q_ptr, (const float*)R_ptr,
        (float*)q_rot_ptr, D
    );

    // Step 2: 融合注意力
    int smem = (4 + 1) * sizeof(float);
    flash_attn_kernel<<<total_heads, D, smem>>>(
        (const float*)q_rot_ptr,
        (const unsigned char*)pk_ptr,
        (const float*)km_ptr, (const float*)ks_ptr, (const float*)kr_ptr,
        (const unsigned char*)pv_ptr,
        (const float*)vm_ptr, (const float*)vs_ptr, (const float*)vr_ptr,
        (float*)attn_out_ptr, (float*)score_ptr,
        total_heads, total_kv, S, D, group_size, num_groups, scale
    );

    // Step 3: 逆旋转
    inverse_rotate_kernel<<<total_heads, D>>>(
        (const float*)attn_out_ptr, (const float*)R_ptr,
        (float*)out_ptr, D
    );
}

PYBIND11_MODULE(polarquant_full, m) {
    m.def("full_compressed_attention", &full_compressed_attention,
          "Full C++ compressed attention: rotate Q → fused attn → inverse rotate");
}
