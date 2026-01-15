#include "include/cache_benchmark.h"
#include <cuda_runtime.h>

int main() {
    cudaSetDevice(0);
    cuda_perf::L1CacheSweepTest l1_test;
    l1_test.execute();
    return 0;
}
