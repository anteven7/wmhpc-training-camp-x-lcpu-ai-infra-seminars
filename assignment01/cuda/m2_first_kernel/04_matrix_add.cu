// Problem 2.6: Two-dimensional matrix addition (fill in the blanks).
//
// Use a two-dimensional block and grid to process an M × N matrix.
// All four blanks are related to two-dimensional indexing.
//
// This file will not compile until all blanks have been completed.

#include "common.h"

__global__ void matrixAdd(
    const float *a,
    const float *b,
    float *c,
    int M,
    int N
) {
    // ====== Blank 1: Row handled by this thread
    //        (use the built-in variables in the y direction) ======
    int row = threadIdx.y + blockIdx.y * blockDim.y;/* Fill here */

    // ====== Blank 2: Column handled by this thread
    //        (use the built-in variables in the x direction) ======
    int col = threadIdx.x + blockIdx.x * blockDim.x;/* Fill here */

    // ====== Blank 3: Two-dimensional bounds check ======
    if (row < M && col < N) {
        // Flatten the row-major two-dimensional index into a
        // one-dimensional index.
        int idx = row * N + col;

        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int M = 1000;
    const int N = 700;  // Neither dimension is a multiple of 16

    const long total = (long)M * N;
    size_t bytes = total * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    fill_random(h_a, total, 1);
    fill_random(h_b, total, 2);

    for (long i = 0; i < total; i++)
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

    // 16 columns in the x direction and 16 rows in the y direction.
    dim3 threads(16, 16);

    // ====== Blank 4: Two-dimensional grid—round up in both dimensions ======
    int block_y = (M + 16 - 1)/16;
    int block_x = (N + 16 - 1)/16;

    dim3 blocks(block_x, block_y);

    matrixAdd<<<blocks, threads>>>(d_a, d_b, d_c, M, N);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(
        h_c,
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    REPORT(check_close(h_c, h_ref, total));

    return 0;
}