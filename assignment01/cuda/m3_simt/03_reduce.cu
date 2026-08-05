// Problem 3.5: Sum reduction within a block.
//
// Task: Implement the following two kernels from scratch. The correctness
// checks and timing code in main are already provided; do not modify them.
//
// Contract—the parts shared by both kernels:
//
//   - The launch configuration is <<<nblocks, BLOCK>>>, where BLOCK = 256.
//
//   - Block b is responsible for the 256 elements from:
//
//         in[b * BLOCK]
//
//     through:
//
//         in[b * BLOCK + 255]
//
//     It must calculate their sum and write it into out[b]. After the
//     reduction finishes, one thread—normally tid == 0—writes the result.
//
//   - Perform the summation in shared memory. Each thread first moves its
//     corresponding input element into:
//
//         __shared__ float buf[BLOCK]
//
//     All subsequent additions must operate on buf.
//
//   - Call __syncthreads() before and after the paired additions in every
//     reduction round.
//
// The only difference between the two kernels is which threads perform the
// additions in each round and which positions they add:
//
//   - reduce_interleaved—interleaved pairing:
//
//     The stride s takes the values 1, 2, 4, ..., 128.
//
//     In every round, threads satisfying:
//
//         tid % (2 * s) == 0
//
//     perform:
//
//         buf[tid] += buf[tid + s]
//
//     Example with eight elements:
//
//       s = 1:
//         buf[0] += buf[1]
//         buf[2] += buf[3]
//         buf[4] += buf[5]
//         buf[6] += buf[7]
//
//         Threads 0, 2, 4, and 6 work. Active threads are interleaved
//         with inactive threads within the warp.
//
//       s = 2:
//         buf[0] += buf[2]
//         buf[4] += buf[6]
//
//         Threads 0 and 4 work.
//
//       s = 4:
//         buf[0] += buf[4]
//
//         Only thread 0 works.
//
//   - reduce_contiguous—contiguous pairing:
//
//     The stride s takes the values 128, 64, ..., 1.
//
//     In every round, threads satisfying:
//
//         tid < s
//
//     perform:
//
//         buf[tid] += buf[tid + s]
//
//     Example with eight elements:
//
//       s = 4:
//         buf[0] += buf[4]
//         buf[1] += buf[5]
//         buf[2] += buf[6]
//         buf[3] += buf[7]
//
//         Threads 0–3 work. The active threads are grouped together at the
//         low-index end.
//
//       s = 2:
//         buf[0] += buf[2]
//         buf[1] += buf[3]
//
//         Threads 0 and 1 work.
//
//       s = 1:
//         buf[0] += buf[1]
//
//         Only thread 0 works.
//
//   Both versions perform exactly the same number of additions. The only
//   difference is how the active threads are distributed within each warp.
//
// Important: Write the loop bounds in both kernels using blockDim.x—the
// runtime value—not the BLOCK macro. Using a compile-time constant would
// allow the compiler to completely unroll the loops and optimize % into
// bitwise operations, affecting the performance comparison.
//
// After implementing the kernels, run the program. Once both versions produce
// PASS, record their execution times and ratio and explain the difference.
//
// Optional third version—shuffle reduction:
//
// The tests run only the two kernels above. You may add another run_one call
// in main for your shuffle version without affecting the tests for the first
// two implementations.

#include "common.h"

#define BLOCK 256

__global__ void reduce_interleaved(const float *in, float *out) {
    // TODO: Begin implementing the interleaved-pairing version here.

    __shared__ float buf[BLOCK];

    int t = threadIdx.x;
    int base = blockIdx.x * BLOCK;
    int tid = t + base;

    buf[t] = in[tid];
    __syncthreads();

    for(int s = 1; s<=blockDim.x/2; s*=2){
        if (t % (2 * s) == 0){
            buf[t] += buf[t+s];
        }
        __syncthreads();
    }

    if (t==0){
        out[blockIdx.x] = buf[0];
    }



}

__global__ void reduce_contiguous(const float *in, float *out) {
    // TODO: Begin implementing the contiguous-pairing version here.
    __shared__ float buf[BLOCK];

    int t = threadIdx.x;
    int base = blockIdx.x * BLOCK;
    int tid = t + base;

    buf[t] = in[tid];
    __syncthreads();

    for(int s = blockDim.x/2; s>=1; s=s/2){
        if(t<s){
            buf[t]+=buf[t+s];
        }
        __syncthreads();
    }
    
    if (t==0){
        out[blockIdx.x] = buf[0];
    }
}

// ------------- Correctness checks and timing below—do not modify -------------

typedef void (*reduce_fn)(const float *, float *);

static float run_one(
    reduce_fn fn,
    const char *name,
    const float *d_in,
    float *d_out,
    float *h_out,
    const float *h_partial,
    int nblocks
) {
    CUDA_CHECK(cudaMemset(
        d_out,
        0,
        nblocks * sizeof(float)
    ));

    fn<<<nblocks, BLOCK>>>(d_in, d_out);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(
        h_out,
        d_out,
        nblocks * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    if (!check_close(h_out, h_partial, nblocks, 1e-3f)) {
        printf("%s: FAIL\n", name);
        emit_result("3.5", "fail", "{}");
        exit(1);
    }

    const int reps = 200;
    GpuTimer timer;

    timer.start();

    for (int r = 0; r < reps; r++)
        fn<<<nblocks, BLOCK>>>(d_in, d_out);

    float ms = timer.stop_ms() / reps;
    CUDA_CHECK_KERNEL();

    printf("%s: PASS  average %.4f ms\n", name, ms);

    return ms;
}

int main() {
    const int nblocks = 4096;
    const int n = nblocks * BLOCK;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(nblocks * sizeof(float));
    float *h_partial = (float *)malloc(nblocks * sizeof(float));

    fill_random(h_in, n, 11);

    for (int b = 0; b < nblocks; b++) {
        double s = 0;

        for (int t = 0; t < BLOCK; t++)
            s += h_in[b * BLOCK + t];

        h_partial[b] = (float)s;
    }

    float *d_in, *d_out;

    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(
        &d_out,
        nblocks * sizeof(float)
    ));

    CUDA_CHECK(cudaMemcpy(
        d_in,
        h_in,
        bytes,
        cudaMemcpyHostToDevice
    ));

    float ms_i = run_one(
        reduce_interleaved,
        "interleaved",
        d_in,
        d_out,
        h_out,
        h_partial,
        nblocks
    );

    float ms_c = run_one(
        reduce_contiguous,
        "contiguous ",
        d_in,
        d_out,
        h_out,
        h_partial,
        nblocks
    );

    // Threshold: 1.5×.
    //
    // Measurements:
    //   A100: 2.22×
    //   V100: 2.33×
    //
    // If both kernels are implemented identically, the ratio is approximately 1×.
    float ratio = report_speedup(
        "interleaved / contiguous",
        ms_i,
        ms_c,
        1.5f,
        "The execution times are almost identical; check whether both "
        "kernels were implemented in the same way."
    );

    char metrics[192];

    snprintf(
        metrics,
        sizeof(metrics),
        "{\"interleaved_ms\":%.4f,\"contiguous_ms\":%.4f,\"ratio\":%.3f}",
        ms_i,
        ms_c,
        ratio
    );

    emit_result("3.5", "pass", metrics);

    return 0;
}