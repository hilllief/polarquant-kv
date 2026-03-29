// 绕过 CCCL preprocessor 检查 + std namespace 歧义
#include "../csrc/torch_compat.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 最简单的测试 kernel
__global__ void test_add_kernel(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

torch::Tensor test_add(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    test_add_kernel<<<blocks, threads>>>(
        out.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_add", &test_add, "Test CUDA add");
}
