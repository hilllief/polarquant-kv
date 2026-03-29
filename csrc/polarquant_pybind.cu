/*
 * PolarQuant-KV 纯 CUDA + pybind11（不依赖 torch headers）。
 * Python 端传入 GPU 指针（int64_t），C++ 端直接用。
 */

#include <cuda_runtime.h>
#include <math.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// 复用 V5 的 kernel（已编译为 PTX，这里用 C++ 版本）
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void flash_attn_kernel(
    const float* q, const unsigned char* pk,
    const float* km, const float* ks, const float* kr,
    const unsigned char* pv, const float* vm, const float* vs, const float* vr,
    float* output, float* score_buf,
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
    float q_val = q[hq * D + tid];
    __shared__ float sh[4];

    // Pass 1: scores
    for (int s = 0; s < S; s++) {
        int off = (hkv * S + s);
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
            score_buf[hq*S+s] = w; se += w;
        }
        sh_inv = 1.0f / se;
    }
    __syncthreads();

    // Pass 2: V weighted sum
    float acc = 0.0f;
    float inv = sh_inv;
    for (int s = 0; s < S; s++) {
        float w = score_buf[hq*S+s] * inv;
        int off = (hkv*S+s);
        float vgmin = vm[off*num_groups+my_group];
        float vgsc = vs[off*num_groups+my_group];
        unsigned char vp = pv[off*packed_dim+my_byte];
        float v_val = (float)((vp >> (is_high<<2)) & 0x0F) * vgsc + vgmin;
        acc += w * v_val * vr[off];
    }
    output[hq*D+tid] = acc;
}

// pybind11 接口：接收 int64_t 指针
void launch_flash_attn(
    int64_t q_ptr, int64_t pk_ptr, int64_t km_ptr, int64_t ks_ptr, int64_t kr_ptr,
    int64_t pv_ptr, int64_t vm_ptr, int64_t vs_ptr, int64_t vr_ptr,
    int64_t out_ptr, int64_t score_ptr,
    int total_heads, int total_kv, int S, int D,
    int group_size, int num_groups, float scale
) {
    int smem = (4 + 1) * sizeof(float);
    flash_attn_kernel<<<total_heads, D, smem>>>(
        (const float*)q_ptr, (const unsigned char*)pk_ptr,
        (const float*)km_ptr, (const float*)ks_ptr, (const float*)kr_ptr,
        (const unsigned char*)pv_ptr, (const float*)vm_ptr,
        (const float*)vs_ptr, (const float*)vr_ptr,
        (float*)out_ptr, (float*)score_ptr,
        total_heads, total_kv, S, D, group_size, num_groups, scale
    );
}

PYBIND11_MODULE(polarquant_native, m) {
    m.def("launch_flash_attn", &launch_flash_attn);
}
