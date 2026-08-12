// Problem 4.2: Three-point average stencil (fill in the blanks).
//
// out[i] = (in[i-1] + in[i] + in[i+1]) / 3.
// Treat out-of-bounds positions as 0.
//
// Write two kernels: one using static shared memory and one using dynamic
// shared memory.
//
// This file will not compile until all blanks have been completed.
//
// Note: The shared tile array is used to reduce the number of global-memory
// accesses. Each element is read from device memory only once, while its
// three uses within the block are served from on-chip shared memory, whose
// latency is much lower than global memory.

#include "common.h"

#define BLOCK 256
#define RADIUS 1

__global__ void stencil_static(const float *in, float *out, int n) {
    // ===== Blank 1: Declare a static shared-memory array large enough to
    // hold BLOCK elements plus the halo on both sides. =====
    __shared__ float tile[BLOCK + RADIUS*2];

    int g = blockIdx.x * blockDim.x + threadIdx.x; // Global index
    int l = threadIdx.x + RADIUS;                  // Position inside tile

    tile[l] = (g < n) ? in[g] : 0.f;

    // Threads at the two ends of the block load one additional halo element.
    if (threadIdx.x < RADIUS) {
        int left = g - RADIUS;
        int right = g + BLOCK;

        tile[l - RADIUS] = (left >= 0) ? in[left] : 0.f;
        tile[l + BLOCK] = (right < n) ? in[right] : 0.f;
    }

    // ===== Blank 2: Add one line here. =====
    __syncthreads();

    if (g < n) {
        // ===== Blank 3: Calculate the three-point average using tile.
        // Do not use in. =====
        out[g] = (tile[l-1] + tile[l] + tile[l+1]) / 3.f;
    }
}

__global__ void stencil_dynamic(const float *in, float *out, int n) {
    // ===== Blank 4: Declare the dynamic shared-memory array.
    // Its size is provided when the kernel is launched. =====
    /* fill this in: declare the dynamic shared array tile */
    extern __shared__ float array[];

    float* tile = array;

    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int l = threadIdx.x + RADIUS;

    tile[l] = (g < n) ? in[g] : 0.f;

    if (threadIdx.x < RADIUS) {
        int left = g - RADIUS;
        int right = g + BLOCK;

        tile[l - RADIUS] = (left >= 0) ? in[left] : 0.f;
        tile[l + BLOCK] = (right < n) ? in[right] : 0.f;
    }

    __syncthreads();

    if (g < n) {
        out[g] = (tile[l - 1] + tile[l] + tile[l + 1]) / 3.f;
    }
}

int main() {
    const int n = 1000003;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    fill_random(h_in, n, 3);

    for (int i = 0; i < n; i++) {
        float l = (i > 0) ? h_in[i - 1] : 0.f;
        float r = (i < n - 1) ? h_in[i + 1] : 0.f;

        h_ref[i] = (l + h_in[i] + r) / 3.f;
    }

    float *d_in, *d_out;

    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int blocks = (n + BLOCK - 1) / BLOCK;

    stencil_static<<<blocks, BLOCK>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(
        h_out,
        d_out,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    if (!check_close(h_out, h_ref, n))
        REPORT(0);

    printf("static  PASS\n");

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    // ===== Blank 5: Launch the dynamic shared-memory version.
    // How many bytes should the third launch parameter contain? =====
    size_t b = (BLOCK + 2 * RADIUS) * sizeof(float);
    stencil_dynamic<<<blocks, BLOCK, b/* fill this in */>>>(
        d_in,
        d_out,
        n
    );

    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(
        h_out,
        d_out,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    if (!check_close(h_out, h_ref, n))
        REPORT(0);

    printf("dynamic PASS\n");

    REPORT(1);
    return 0;
}