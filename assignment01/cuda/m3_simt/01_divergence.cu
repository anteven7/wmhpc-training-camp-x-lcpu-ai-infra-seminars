// Problem 3.2: Divergence
//
// Every thread in both kernels performs the same amount of computation.
// The only difference is how the branches divide the threads.
//
// First write down your prediction in the handout, then run the program
// and compare the results.

#include "common.h"

// Branch according to even/odd thread indices:
// within the same warp, half of the threads take each branch.
__global__ void diverge_in_warp(float *out, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = tid * 0.5f;

    if (tid % 2 == 0) {
        for (int i = 0; i < iters; i++)
            x = x * 1.000001f + 0.5f;
    } else {
        for (int i = 0; i < iters; i++)
            x = x * 0.999999f - 0.5f;
    }

    out[tid] = x;
}

// Branch according to the warp:
// all threads within one warp take the same branch.
__global__ void diverge_by_warp(float *out, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = tid * 0.5f;

    if ((tid / 32) % 2 == 0) {
        for (int i = 0; i < iters; i++)
            x = x * 1.000001f + 0.5f;
    } else {
        for (int i = 0; i < iters; i++)
            x = x * 0.999999f - 0.5f;
    }

    out[tid] = x;
}

int main() {
    const int blocks = 1024;
    const int threads = 256;
    const int iters = 20000;

    const int n = blocks * threads;

    float *d_out;
    CUDA_CHECK(cudaMalloc(
        &d_out,
        (size_t)n * sizeof(float)
    ));

    // Warm up each kernel once.
    diverge_in_warp<<<blocks, threads>>>(d_out, iters);
    diverge_by_warp<<<blocks, threads>>>(d_out, iters);
    CUDA_CHECK_KERNEL();

    GpuTimer timer;

    timer.start();
    diverge_in_warp<<<blocks, threads>>>(d_out, iters);
    float ms_in = timer.stop_ms();

    timer.start();
    diverge_by_warp<<<blocks, threads>>>(d_out, iters);
    float ms_by = timer.stop_ms();

    CUDA_CHECK_KERNEL();

    printf(
        "Branching within warp (tid %% 2)    : %8.3f ms\n",
        ms_in
    );

    printf(
        "Branching by warp (tid/32 %% 2)     : %8.3f ms\n",
        ms_by
    );

    printf("Ratio: %.2f\n", ms_in / ms_by);

    return 0;
}