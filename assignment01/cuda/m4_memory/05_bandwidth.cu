// Problem 4.7: Memory-access patterns and bandwidth.
//
// Use the same kernel while changing only the read stride.

#include "common.h"

// When stride = 1, memory accesses are contiguous.
//
// As stride increases, adjacent threads in a warp read addresses separated
// by stride float elements.
//
// Because n is a power of two, & (n - 1) is equivalent to modulo n.
__global__ void strided_copy(
    const float *in,
    float *out,
    int n,
    int stride
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int j = (long)i * stride & (n - 1);
        out[i] = in[j];
    }
}

int main() {
    // 16 million elements; n is a power of two.
    const int n = 1 << 24;
    size_t bytes = (size_t)n * sizeof(float);

    float *d_in;
    float *d_out;

    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_in, 1, bytes));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Warm-up run.
    strided_copy<<<blocks, threads>>>(d_in, d_out, n, 1);
    CUDA_CHECK_KERNEL();

    const int reps = 20;
    int strides[] = {1, 2, 4, 8, 16, 32};

    printf("%8s %12s %12s\n", "stride", "ms", "GB/s");

    for (int s : strides) {
        GpuTimer timer;
        timer.start();

        for (int r = 0; r < reps; r++) {
            strided_copy<<<blocks, threads>>>(
                d_in,
                d_out,
                n,
                s
            );
        }

        float ms = timer.stop_ms() / reps;
        CUDA_CHECK_KERNEL();

        // Each element involves a 4-byte read and a 4-byte write.
        double gbps =
            2.0 * bytes / (ms * 1e-3) / 1e9;

        printf(
            "%8d %12.4f %12.1f\n",
            s,
            ms,
            gbps
        );
    }

    return 0;
}