/*
 * PolarQuant-KV 融合 CUDA Kernel
 *
 * 1. fused_compress_kernel: 旋转 + 范数 + 归一化 + 量化 + bit pack
 * 2. fused_score_kernel: 从压缩格式直接计算注意力分数（不解压 K）
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// ============================================================
// Kernel 1: 融合压缩 (per-vector, 一个 thread block 处理一个向量)
// ============================================================

__global__ void fused_compress_4bit_kernel(
    const half* __restrict__ input,       // [N, D] 输入 KV 向量
    const float* __restrict__ R,          // [D, D] 旋转矩阵
    half* __restrict__ radius_out,        // [N] 半径
    unsigned char* __restrict__ packed_out,// [N, D/2] 4-bit packed
    half* __restrict__ gmins_out,         // [N, G] group mins
    half* __restrict__ gscales_out,       // [N, G] group scales
    int N, int D, int group_size
) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= N) return;

    int tid = threadIdx.x;
    int num_groups = (D + group_size - 1) / group_size;

    // Shared memory: 旋转后的向量 + query 旋转后
    extern __shared__ float smem[];
    float* rotated = smem;           // [D]
    float* direction = smem + D;     // [D]

    // Step 1: 旋转 v_rotated[k] = sum_j input[vec_idx, j] * R[j, k]
    // 每个线程算一个维度
    if (tid < D) {
        float sum = 0.0f;
        for (int j = 0; j < D; j++) {
            sum += __half2float(input[vec_idx * D + j]) * R[j * D + tid];
        }
        rotated[tid] = sum;
    }
    __syncthreads();

    // Step 2: 计算范数 (parallel reduction)
    __shared__ float norm_sq_shared[256];
    float my_sq = 0.0f;
    if (tid < D) {
        my_sq = rotated[tid] * rotated[tid];
    }
    norm_sq_shared[tid] = my_sq;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            norm_sq_shared[tid] += norm_sq_shared[tid + s];
        }
        __syncthreads();
    }

    float radius = sqrtf(norm_sq_shared[0]);
    if (tid == 0) {
        radius_out[vec_idx] = __float2half(radius);
    }

    // Step 3: 归一化
    if (tid < D) {
        direction[tid] = (radius > 1e-30f) ? (rotated[tid] / radius) : 0.0f;
    }
    __syncthreads();

    // Step 4: 分组量化 + bit pack
    // 每个线程处理一个 group
    if (tid < num_groups) {
        int g_start = tid * group_size;
        int g_end = min(g_start + group_size, D);

        // 找 min/max
        float gmin = direction[g_start];
        float gmax = direction[g_start];
        for (int i = g_start + 1; i < g_end; i++) {
            float v = direction[i];
            gmin = fminf(gmin, v);
            gmax = fmaxf(gmax, v);
        }

        float range = gmax - gmin;
        float scale = (range > 0.0f) ? (range / 15.0f) : 1.0f;  // 4-bit: 15 levels

        gmins_out[vec_idx * num_groups + tid] = __float2half(gmin);
        gscales_out[vec_idx * num_groups + tid] = __float2half(scale);

        // 量化 + 4-bit pack (2 values per byte)
        for (int i = g_start; i < g_end; i += 2) {
            float v0 = direction[i];
            unsigned char q0 = (unsigned char)fminf(15.0f, fmaxf(0.0f, roundf((v0 - gmin) / scale)));

            unsigned char q1 = 0;
            if (i + 1 < g_end) {
                float v1 = direction[i + 1];
                q1 = (unsigned char)fminf(15.0f, fmaxf(0.0f, roundf((v1 - gmin) / scale)));
            }

            // Pack: low nibble = q0, high nibble = q1
            unsigned char packed = q0 | (q1 << 4);
            packed_out[vec_idx * (D / 2) + i / 2] = packed;
        }
    }
}


// ============================================================
// Kernel 2: 融合注意力分数 (从压缩 K 直接算 score，不解压)
// ============================================================

__global__ void fused_score_4bit_kernel(
    const float* __restrict__ q_rotated,   // [D] 旋转空间的 query
    const unsigned char* __restrict__ packed_keys, // [S, D/2] 压缩的 keys
    const half* __restrict__ gmins,        // [S, G] group mins
    const half* __restrict__ gscales,      // [S, G] group scales
    const half* __restrict__ radius,       // [S] 半径
    float* __restrict__ scores,            // [S] 输出分数
    int S, int D, int group_size, float scale
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;

    int num_groups = (D + group_size - 1) / group_size;

    float dot = 0.0f;
    for (int g = 0; g < num_groups; g++) {
        float gmin = __half2float(gmins[s * num_groups + g]);
        float gscale = __half2float(gscales[s * num_groups + g]);
        int g_start = g * group_size;
        int g_end = min(g_start + group_size, D);

        for (int i = g_start; i < g_end; i += 2) {
            unsigned char packed = packed_keys[s * (D / 2) + i / 2];
            unsigned char q0 = packed & 0x0F;
            unsigned char q1 = (packed >> 4) & 0x0F;

            float v0 = (float)q0 * gscale + gmin;
            dot += v0 * q_rotated[i];

            if (i + 1 < g_end) {
                float v1 = (float)q1 * gscale + gmin;
                dot += v1 * q_rotated[i + 1];
            }
        }
    }

    float r = __half2float(radius[s]);
    scores[s] = dot * r * scale;
}


// ============================================================
// Python 绑定
// ============================================================

torch::Tensor fused_compress_4bit(
    torch::Tensor input,          // [N, D], FP16, CUDA
    torch::Tensor rotation_matrix // [D, D], FP32, CUDA
) {
    int N = input.size(0);
    int D = input.size(1);
    int group_size = 32;
    int num_groups = (D + group_size - 1) / group_size;

    auto opts_half = torch::TensorOptions().dtype(torch::kFloat16).device(input.device());
    auto opts_uint8 = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());

    auto radius = torch::empty({N}, opts_half);
    auto packed = torch::empty({N, D / 2}, opts_uint8);
    auto gmins = torch::empty({N, num_groups}, opts_half);
    auto gscales = torch::empty({N, num_groups}, opts_half);

    int threads = max(D, num_groups);
    threads = min(threads, 256);
    // Pad to power of 2 for reduction
    int block_size = 1;
    while (block_size < threads) block_size <<= 1;
    block_size = min(block_size, 256);

    int smem_size = 2 * D * sizeof(float);

    fused_compress_4bit_kernel<<<N, block_size, smem_size>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        rotation_matrix.data_ptr<float>(),
        reinterpret_cast<half*>(radius.data_ptr<at::Half>()),
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(gmins.data_ptr<at::Half>()),
        reinterpret_cast<half*>(gscales.data_ptr<at::Half>()),
        N, D, group_size
    );

    return torch::stack({radius.unsqueeze(1).to(torch::kFloat32),
                         packed.to(torch::kFloat32),
                         gmins.to(torch::kFloat32),
                         gscales.to(torch::kFloat32)});
    // 返回一个 list 更好，但 pybind 需要 tuple
    // 简化：返回 dict-like 的多个 tensor
}


torch::Tensor fused_score_4bit(
    torch::Tensor q_rotated,      // [D], FP32, CUDA
    torch::Tensor packed_keys,    // [S, D/2], uint8, CUDA
    torch::Tensor gmins,          // [S, G], FP16, CUDA
    torch::Tensor gscales,        // [S, G], FP16, CUDA
    torch::Tensor radius,         // [S], FP16, CUDA
    int D, int group_size, float scale
) {
    int S = packed_keys.size(0);
    auto scores = torch::empty({S}, torch::TensorOptions().dtype(torch::kFloat32).device(q_rotated.device()));

    int threads = 256;
    int blocks = (S + threads - 1) / threads;

    fused_score_4bit_kernel<<<blocks, threads>>>(
        q_rotated.data_ptr<float>(),
        packed_keys.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(gmins.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(gscales.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(radius.data_ptr<at::Half>()),
        scores.data_ptr<float>(),
        S, D, group_size, scale
    );

    return scores;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_compress_4bit", &fused_compress_4bit, "Fused PolarQuant 4-bit compress");
    m.def("fused_score_4bit", &fused_score_4bit, "Fused 4-bit attention score");
}
