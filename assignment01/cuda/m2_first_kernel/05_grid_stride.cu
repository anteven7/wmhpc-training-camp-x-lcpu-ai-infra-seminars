// Problem 2.7: Grid-stride loop (code modification exercise).
//
// Current situation: The launch uses only 64 blocks, so the total number of
// threads is much smaller than n. As a result, the program produces FAIL.
//
// Task: You are not allowed to change the launch configuration. Modify the
// kernel to use a grid-stride loop. Each thread should process multiple
// elements by advancing by the total size of the grid, allowing the program
// to produce PASS for any value of n.
//
// Reference: NVIDIA blog,
// "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops"
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/

#include "common.h"

__global__ void vectorAdd(
    const float *a,
    const float *b,
    float *c,
    int n
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i<n; i+=stride) {
        if (idx < n)
            c[i] = a[i] + b[i];
    }
}

int main() {
    // 16 million elements—far more than the 64 × 256 = 16,384 threads.
    const int n = 1 << 24;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);

    for (int i = 0; i < n; i++)
        h_ref[i] = h_a[i] + h_b[i];

    float *d_a, *d_b, *d_c;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_a,
        h_a,
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_b,
        h_b,
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

    // The launch configuration must not be changed.
    vectorAdd<<<64, 256>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(
        h_c,
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    REPORT(check_close(h_c, h_ref, n));

    return 0;
}