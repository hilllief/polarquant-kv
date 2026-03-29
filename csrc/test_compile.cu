#include <stdio.h>

__global__ void test_kernel(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)i * 2.0f;
}

int main() {
    printf("CUDA compile test OK\n");
    return 0;
}
