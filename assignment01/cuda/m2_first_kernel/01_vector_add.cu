// Problem 2.1: Vector addition (fill in the blanks)
// Each of the six blanks tests one concept. Fill them in, compile, and run;
// receiving "PASS" means the exercise is complete.
// This file will not compile until all blanks have been filled in.
#include "common.h"

// ====== Blank 1: What function qualifier does a kernel require? ======
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // ====== Blank 2: The global index handled by this thread ======
    int idx =  threadIdx.x + blockDim.x * blockIdx.x;

    // ====== Blank 3: Bounds check—the total number of threads may exceed the number of elements ======
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 1000003;  // Intentionally chosen not to be a multiple of 256
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

    // ====== Blank 4: Copy h_a and h_b to the device
    //        (pay attention to the final direction argument) ======
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice /* Fill here */));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice/* Fill here */));

    int threadsPerBlock = 256;

    // ====== Blank 5: Number of blocks—round up to ensure all n elements are covered ======
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // ====== Blank 6: Launch the kernel (where does the execution configuration go?) ======
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_c, h_ref, n));

    return 0;
}