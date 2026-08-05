// Problem 2.3: Convert explicit memory management to Unified Memory (MODIFY).
//
// Below is a complete, runnable version using explicit memory management.
// Tasks:
//
//   0. First, run this version without modifications and record its execution
//      time. Your changes will overwrite this version, and you will need its
//      timing as the baseline for the comparison in step 4.
//
//   1. Replace cudaMalloc + malloc with cudaMallocManaged.
//
//   2. Remove every cudaMemcpy. The kernel should directly read and write the
//      same set of pointers, and the CPU should also read them directly.
//
//   3. Think carefully about where cudaDeviceSynchronize is required.
//
//   4. Compare the execution times of both versions. The timing window must be
//      identical in both versions: allocation and data initialization must
//      remain outside the timing window. Timing starts when the data is already
//      prepared in memory and ends after the CPU has read the complete result.
//      The checksum accumulation loop below represents "the CPU reading the
//      complete result," so do not remove it.
//
// The modified version must still produce PASS.

#include <chrono>
#include "common.h"

__global__ void vectorAdd(
    const float *a,
    const float *b,
    float *c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main() {
    const int n = 1 << 24;  // 16 million elements
    size_t bytes = (size_t)n * sizeof(float);

    // Initialize the CUDA context first. The first CUDA API call can take
    // several hundred milliseconds because of initialization. Including it
    // in the timing window would completely hide the difference that we want
    // to observe.
    CUDA_CHECK(cudaFree(0));
/* 
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
 */

 
    float *h_a = nullptr;
    float *h_b = nullptr;
    float *h_c = nullptr;

    cudaMallocManaged(&h_a, bytes);
    cudaMallocManaged(&h_b, bytes);
    cudaMallocManaged(&h_c, bytes);


    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);

    // Calculate the expected checksum on the host in advance. This is also
    // excluded from the timing measurement.
    double want = 0;

    for (int i = 0; i < n; i++)
        want += (double)(h_a[i] + h_b[i]);

/*     float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
 */
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // ================= Timing window begins =================
    auto t0 = std::chrono::steady_clock::now();
/* 
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
    )); */

    vectorAdd<<<blocks, threads>>>(h_a, h_b, h_c, n);
    CUDA_CHECK_KERNEL(); //this does already cudaDeviceSynchronize
/* 
    CUDA_CHECK(cudaMemcpy(
        h_c,
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    )) ; */

    // The CPU reads the complete result. In the Unified Memory version, this
    // step is what causes the result pages to migrate back to the host.
    double got = 0;

    for (int i = 0; i < n; i++)
        got += (double)h_c[i];

    auto t1 = std::chrono::steady_clock::now();
    // ================= Timing window ends =================

    printf(
        "Transfers + kernel + reading result: %.1f ms\n",
        std::chrono::duration<double, std::milli>(t1 - t0).count()
    );

    REPORT(fabs(got - want) <= 1e-3 * (1.0 + fabs(want)));

    return 0;
}