// Problem 4.5: Histogram (fill in the blank).
//
// Count how many of 16 million byte values fall into each of 256 bins.
//
// Note: Multiple threads may try to modify the same bin simultaneously.

#include "common.h"

__global__ void histogram(
    const unsigned char *data,
    unsigned int *hist,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (;i < n; i += stride) {
        unsigned char v = data[i];

        // ===== Blank 1: Add 1 to hist[v].
        // Which atomic operation should be used? =====
        /* fill this in */
        atomicAdd(&hist[v], 1u);
    }
}

int main() {
    const int n = 1 << 24;
    const int BINS = 256;

    unsigned char *h_data = (unsigned char *)malloc(n);
    unsigned int h_hist[BINS];
    unsigned int h_ref[BINS] = {0};

    srand(9);

    for (int i = 0; i < n; i++)
        h_data[i] = (unsigned char)(rand() % BINS);

    for (int i = 0; i < n; i++)
        h_ref[h_data[i]]++;

    unsigned char *d_data;
    unsigned int *d_hist;

    CUDA_CHECK(cudaMalloc(&d_data, n));

    CUDA_CHECK(cudaMalloc(
        &d_hist,
        BINS * sizeof(unsigned int)
    ));

    CUDA_CHECK(cudaMemcpy(
        d_data,
        h_data,
        n,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemset(
        d_hist,
        0,
        BINS * sizeof(unsigned int)
    ));

    int threads = 256;
    int blocks = 1024;

    histogram<<<blocks, threads>>>(d_data, d_hist, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(
        h_hist,
        d_hist,
        BINS * sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    ));

    int ok = 1;

    for (int b = 0; b < BINS; b++) {
        if (h_hist[b] != h_ref[b]) {
            fprintf(
                stderr,
                "bin %d: got %u, expected %u\n",
                b,
                h_hist[b],
                h_ref[b]
            );

            ok = 0;
            break;
        }
    }

    // Measure execution time for comparison with the optimized version
    // implemented in Problem 4.6.
    const int reps = 50;
    GpuTimer timer;

    timer.start();

    for (int r = 0; r < reps; r++)
        histogram<<<blocks, threads>>>(d_data, d_hist, n);

    float ms = timer.stop_ms() / reps;
    CUDA_CHECK_KERNEL();

    printf(
        "Average time %.4f ms  (%.2f GB/s)\n",
        ms,
        n / ms / 1e6
    );

    REPORT(ok);

    return 0;
}