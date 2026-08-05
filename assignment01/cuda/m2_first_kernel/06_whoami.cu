// Problem 2.8: Observe the execution order.
//
// Run the program two or three times and compare the order in which the blocks
// appear. Then answer the corresponding question in the handout.

#include "common.h"

__global__ void whoami() {
    // Have thread 0 from every block report its block number.
    if (threadIdx.x == 0) {
        printf("block %d reporting in\n", blockIdx.x);
    }
}

int main() {
    whoami<<<16, 32>>>();
    CUDA_CHECK_KERNEL();

    return 0;
}