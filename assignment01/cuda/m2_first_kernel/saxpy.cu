#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>


// cuda macro 
#define CHECK(call) do {                                                \
        cudaError_t err = call;                                         \
        if(err != cudaSuccess){                                         \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",             \
                    cudaGetErrorName(err), __FILE__, __LINE__,          \
                    cudaGetErrorString(err));                           \
            exit(1);                                                    \
        }                                                               \
    } while(0)

__global__ void saxpy(
    const float* x,
    float* y,
    int n
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
        y[idx] = 2.0f * x[idx] + y[idx];
}

int main(int argc, char **argv){

    int n = std::atoi(argv[1]);

    if (n == 0) {
    printf("SUM=0\n");
    return 0;
    }

    size_t bytes = (size_t)n * sizeof(float);

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);

    // we init the arrays at host 
    for (int i = 0; i < n; i++){
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] = (i % 1024) - 512;
    }

    // device pointers, we allocate and copy 
    float *d_x, *d_y;

    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    // define threads and blocks, launch kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
   
    saxpy<<<blocks, threads>>>(d_x, d_y, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // copy back to host
    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

    // checking with cpu sum
    double sum = 0;
    for (int i = 0; i<n; i++){
        sum = sum + h_y[i];
    }

    printf("SUM=%.0f\n", sum);

    return 0;
}
