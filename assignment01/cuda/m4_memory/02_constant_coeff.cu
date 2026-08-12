// Problem 4.3: Move the coefficient table into constant memory (MODIFY).
//
// The poly_eval_global kernel below stores the eight polynomial coefficients
// in global memory, and every thread reads them eight times.
//
// Leave this kernel unchanged so that it can be used as the comparison
// baseline.
//
// Tasks:
//
//   1. Declare:
//
//          __constant__ float COEF[8];
//
//   2. At the location marked TODO in main, use cudaMemcpyToSymbol to copy
//      the coefficients into COEF.
//
//   3. Implement poly_eval_const so that it reads the coefficients from COEF.
//      Keep its parameter list unchanged because the testing code must run
//      both versions using the same function-pointer type. The coef pointer
//      does not need to be used inside poly_eval_const.
//
// Both versions must produce PASS. The evaluation results will include the
// execution time of each version and their ratio.

#include "common.h"

__constant__ float COEF[8];


__global__ void poly_eval_global(
    const float *x,
    float *y,
    const float *coef,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float xi = x[i];
        float acc = 0.f;

        // Horner's method: evaluate from the highest-degree coefficient
        // down to the constant coefficient.
        for (int k = 7; k >= 0; k--)
            acc = acc * xi + coef[k];

        y[i] = acc;
    }
}

__global__ void poly_eval_const(
    const float *x,
    float *y,
    const float *coef,
    int n
) {
    // TODO: Implement the version that reads from the __constant__ COEF
    // array, starting here.

    int i = blockIdx.x * blockDim.x + threadIdx.x;

     if (i < n) {
        float xi = x[i];
        float acc = 0.f;

        // Horner's method: evaluate from the highest-degree coefficient
        // down to the constant coefficient.
        for (int k = 7; k >= 0; k--)
            acc = acc * xi + COEF[k];

        y[i] = acc;
    }

}

// ---------------- Testing and timing below—do not modify ----------------

typedef void (*poly_fn)(
    const float *,
    float *,
    const float *,
    int
);

static float run_one(
    poly_fn fn,
    const char *name,
    const float *d_x,
    float *d_y,
    const float *d_coef,
    float *h_y,
    const float *h_ref,
    int n,
    int blocks,
    int threads
) {
    CUDA_CHECK(cudaMemset(
        d_y,
        0,
        (size_t)n * sizeof(float)
    ));

    fn<<<blocks, threads>>>(d_x, d_y, d_coef, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(
        h_y,
        d_y,
        (size_t)n * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    if (!check_close(h_y, h_ref, n, 1e-3f)) {
        printf("%s: FAIL\n", name);
        emit_result("4.3", "fail", "{}");
        exit(1);
    }

    const int reps = 100;
    GpuTimer timer;

    timer.start();

    for (int r = 0; r < reps; r++)
        fn<<<blocks, threads>>>(d_x, d_y, d_coef, n);

    float ms = timer.stop_ms() / reps;
    CUDA_CHECK_KERNEL();

    printf("%s: PASS  average %.4f ms\n", name, ms);

    return ms;
}

int main() {
    const int n = 1 << 24;
    size_t bytes = (size_t)n * sizeof(float);

    float h_coef[8] = {
        1.f,
        -0.5f,
        0.25f,
        -0.125f,
        0.0625f,
        -0.03125f,
        0.015625f,
        -0.0078125f
    };

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    fill_random(h_x, n, 5);

    // Scale the values to approximately [0, 1) to prevent overflow.
    for (int i = 0; i < n; i++)
        h_x[i] = h_x[i] * 0.1f;

    for (int i = 0; i < n; i++) {
        float acc = 0.f;

        for (int k = 7; k >= 0; k--)
            acc = acc * h_x[i] + h_coef[k];

        h_ref[i] = acc;
    }

    float *d_x;
    float *d_y;
    float *d_coef;

    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_coef, sizeof(h_coef)));

    CUDA_CHECK(cudaMemcpy(
        d_x,
        h_x,
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_coef,
        h_coef,
        sizeof(h_coef),
        cudaMemcpyHostToDevice
    ));

    // TODO: Copy h_coef into the __constant__ array that you declared.
    // Use cudaMemcpyToSymbol.
    cudaMemcpyToSymbol(COEF, h_coef, sizeof(h_coef));
   
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float ms_g = run_one(
        poly_eval_global,
        "global  ",
        d_x,
        d_y,
        d_coef,
        h_y,
        h_ref,
        n,
        blocks,
        threads
    );

    float ms_c = run_one(
        poly_eval_const,
        "constant",
        d_x,
        d_y,
        d_coef,
        h_y,
        h_ref,
        n,
        blocks,
        threads
    );

    // This problem is not expected to produce a speedup. A ratio close to
    // 1.00x is normal, so no performance-degradation warning is configured.
    float ratio = report_speedup(
        "global / constant",
        ms_g,
        ms_c,
        0.f,
        NULL
    );

    char metrics[192];

    snprintf(
        metrics,
        sizeof(metrics),
        "{\"global_ms\":%.4f,\"const_ms\":%.4f,\"speedup\":%.3f}",
        ms_g,
        ms_c,
        ratio
    );

    emit_result("4.3", "pass", metrics);

    return 0;
}