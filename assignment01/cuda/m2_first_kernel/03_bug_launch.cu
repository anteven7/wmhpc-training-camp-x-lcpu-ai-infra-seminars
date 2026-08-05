// Problem 2.5: Find the bug.
//
// This program does not report an error; it simply produces FAIL.
// It looks as though the kernel did not run at all...
//
// Task: First identify the specific error (see the hint at the end of the
// file), then explain its cause and fix it.

#include "common.h"

__global__ void vectorAdd(
    const float *a,
    const float *b,
    float *c,
    int n
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main() {
    const int n = 1000003;
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

    int threads = 1024;
    int blocks = (n + threads - 1) / threads;
    printf("threads = %d\n", threads);
    printf("blocks  = %d\n", blocks);
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
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

// Hint: Add the following line after the kernel-launch statement and run the
// program again:
//
//     CUDA_CHECK_KERNEL();
//
// The reported error will tell you which direction to investigate. After
// finding the problem, remember to answer this question: Why does the program
// remain completely silent when this line is absent?
//
// Which limit printed in Problem 0.2 is relevant here?