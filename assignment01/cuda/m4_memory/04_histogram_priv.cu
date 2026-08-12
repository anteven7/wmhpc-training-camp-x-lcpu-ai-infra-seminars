// Problem 4.6: Histogram privatization (MODIFY).
//
// The histogram_naive kernel below is the completed version from Problem 4.5.
// All threads compete to update the same set of 256 global counters.
//
// Leave this kernel unchanged so that it can be used as the comparison
// baseline.
//
// Task: Implement histogram_priv using shared-memory privatization:
//
//   1. Each block declares its own counters in shared memory and initializes
//      them to zero.
//
//   2. Threads in the block use atomicAdd to update their block's private
//      counters.
//
//   3. After synchronization, merge all 256 bins from the shared-memory
//      histogram into the global histogram using atomicAdd.
//
// Both versions must produce PASS. The evaluation results will include the
// execution times and their ratio. Explain where the speedup comes from.

#include "common.h"

#define BINS 256

__global__ void histogram_naive(
    const unsigned char *data,
    unsigned int *hist,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < n; i += stride) {
        atomicAdd(&hist[data[i]], 1u);
    }
}

__global__ void histogram_priv(
    const unsigned char *data,
    unsigned int *hist,
    int n
) {

    // TODO: Implement the shared-memory privatized version here.
    __shared__ int counter[BINS];

    for (int b = threadIdx.x; b < BINS; b += blockDim.x) {
        counter[b] = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < n; i += stride) {
        atomicAdd(&counter[data[i]], 1u);
    }

    __syncthreads();

    for (int b = threadIdx.x; b < BINS; b += blockDim.x) {
    atomicAdd(&hist[b], counter[b]);
    }


}

// ---------------- Testing and timing below—do not modify ----------------

typedef void (*hist_fn)(
    const unsigned char *,
    unsigned int *,
    int
);

static float run_one(
    hist_fn fn,
    const char *name,
    const unsigned char *d_data,
    unsigned int *d_hist,
    const unsigned int *h_ref,
    int n,
    int blocks,
    int threads
) {
    unsigned int h_hist[BINS];

    CUDA_CHECK(cudaMemset(
        d_hist,
        0,
        BINS * sizeof(unsigned int)
    ));

    fn<<<blocks, threads>>>(d_data, d_hist, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(
        h_hist,
        d_hist,
        BINS * sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    ));

    for (int b = 0; b < BINS; b++) {
        if (h_hist[b] != h_ref[b]) {
            fprintf(
                stderr,
                "bin %d: got %u, expected %u\n",
                b,
                h_hist[b],
                h_ref[b]
            );

            printf("%s: FAIL\n", name);
            emit_result("4.6", "fail", "{}");
            exit(1);
        }
    }

    const int reps = 50;
    GpuTimer timer;

    timer.start();

    for (int r = 0; r < reps; r++)
        fn<<<blocks, threads>>>(d_data, d_hist, n);

    float ms = timer.stop_ms() / reps;
    CUDA_CHECK_KERNEL();

    printf(
        "%s: PASS  average %.4f ms  (%.2f GB/s)\n",
        name,
        ms,
        n / ms / 1e6
    );

    return ms;
}

int main() {
    const int n = 1 << 24;

    unsigned char *h_data = (unsigned char *)malloc(n);
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

    int threads = 256;
    int blocks = 1024;

    float ms_naive = run_one(
        histogram_naive,
        "naive",
        d_data,
        d_hist,
        h_ref,
        n,
        blocks,
        threads
    );

    float ms_priv = run_one(
        histogram_priv,
        "priv ",
        d_data,
        d_hist,
        h_ref,
        n,
        blocks,
        threads
    );

    // Threshold: 10x.
    //
    // Measured results:
    //   A100: 149x
    //   V100: 86x
    //
    // A ratio around 1x indicates that privatization was not actually
    // implemented.
    float ratio = report_speedup(
        "naive / priv",
        ms_naive,
        ms_priv,
        10.0f,
        "The speedup is below 10x; check whether privatization is actually "
        "working."
    );

    // Report how much shared memory the privatized version actually uses.
    // This value is reported only and is not used as a pass/fail condition.
    cudaFuncAttributes attr;

    CUDA_CHECK(cudaFuncGetAttributes(
        &attr,
        histogram_priv
    ));

    if (attr.sharedSizeBytes == 0) {
        printf(
            "WARN: the privatized version does not use shared memory "
            "(does not affect PASS)\n"
        );
    }

    char metrics[256];

    snprintf(
        metrics,
        sizeof(metrics),
        "{\"naive_ms\":%.4f,\"priv_ms\":%.4f,\"speedup\":%.3f,"
        "\"shared_bytes\":%zu}",
        ms_naive,
        ms_priv,
        ratio,
        attr.sharedSizeBytes
    );

    emit_result("4.6", "pass", metrics);

    return 0;
}