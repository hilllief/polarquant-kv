#include <mma.h>
using namespace nvcuda;

__global__ void test_wmma_kernel() {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(a_frag, __float2half(1.0f));
    wmma::fill_fragment(b_frag, __float2half(1.0f));
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
}

int main() { return 0; }
