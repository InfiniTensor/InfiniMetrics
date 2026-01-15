#pragma once

#include "cuda_utils.h"
#include "performance_test.h"
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cstring>

namespace cuda_perf {

// ============================================================
// Device-to-Device Bandwidth Test
// ============================================================

class DeviceToDeviceBandwidthTest {
public:
    void execute(const TestConfig& config = TestConfig()) {
        std::cout << "\n===================================================\n";
        std::cout << "Device to Device Bandwidth Test\n";
        std::cout << "===================================================\n\n";

        int device_id = 0;
        cudaSetDevice(device_id);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        std::cout << " Device to Device Bandwidth, 1 Device(s)\n";
        std::cout << "   Transfer Size (Bytes)\tBandwidth(GB/s)\n";
        std::cout << std::string(45, '-') << "\n";

        const unsigned int MEMCOPY_ITERATIONS = 100;
        const unsigned int DEFAULT_SIZE = 32 * 1024 * 1024;  // 32 MB

        unsigned char *h_idata = (unsigned char *)malloc(DEFAULT_SIZE);
        unsigned char *d_idata, *d_odata;

        // Initialize the host memory
        for (unsigned int i = 0; i < DEFAULT_SIZE / sizeof(unsigned char); i++) {
            h_idata[i] = (unsigned char)(i & 0xff);
        }

        // Allocate device memory
        cudaMalloc(&d_idata, DEFAULT_SIZE);
        cudaMalloc(&d_odata, DEFAULT_SIZE);

        // Initialize memory
        cudaMemcpy(d_idata, h_idata, DEFAULT_SIZE, cudaMemcpyHostToDevice);

        // Create events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Run the memcopy
        cudaEventRecord(start, 0);

        for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
            cudaMemcpy(d_odata, d_idata, DEFAULT_SIZE, cudaMemcpyDeviceToDevice);
        }

        cudaEventRecord(stop, 0);

        // Since device to device memory copies are non-blocking,
        // cudaDeviceSynchronize() is required in order to get proper timing.
        cudaDeviceSynchronize();

        float elapsedTimeInMs = 0.0f;
        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);

        // Calculate bandwidth in GB/s (note: 2.0x because we're reading and writing)
        double time_s = elapsedTimeInMs / 1e3;
        float bandwidthInGBs = (2.0f * DEFAULT_SIZE * (float)MEMCOPY_ITERATIONS) / (double)1e9;
        bandwidthInGBs = bandwidthInGBs / time_s;

        // Print result
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "   " << DEFAULT_SIZE << "\t\t" << bandwidthInGBs << "\n\n";

        // Clean up memory
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaFree(d_idata);
        cudaFree(d_odata);
        free(h_idata);
    }
};

// ============================================================
// Device to Host Bandwidth Test (Pinned Memory)
// ============================================================

class DeviceToHostBandwidthTest {
public:
    void execute(const TestConfig& config = TestConfig()) {
        std::cout << "\n===================================================\n";
        std::cout << "Device to Host Bandwidth Test (Pinned Memory)\n";
        std::cout << "===================================================\n\n";

        const unsigned int MEMCOPY_ITERATIONS = 100;
        const unsigned int DEFAULT_SIZE = 32 * 1024 * 1024;  // 32 MB

        std::cout << " Device to Host Bandwidth, 1 Device(s)\n";
        std::cout << " PINNED Memory Transfers\n";
        std::cout << "   Transfer Size (Bytes)\tBandwidth(GB/s)\n";
        std::cout << std::string(45, '-') << "\n";

        // Allocate pinned memory
        unsigned char *h_idata, *h_odata;
        cudaHostAlloc(&h_idata, DEFAULT_SIZE, 0);
        cudaHostAlloc(&h_odata, DEFAULT_SIZE, 0);

        // Initialize the memory
        for (unsigned int i = 0; i < DEFAULT_SIZE / sizeof(unsigned char); i++) {
            h_idata[i] = (unsigned char)(i & 0xff);
        }

        // Allocate device memory
        unsigned char *d_idata;
        cudaMalloc(&d_idata, DEFAULT_SIZE);

        // Initialize the device memory
        cudaMemcpy(d_idata, h_idata, DEFAULT_SIZE, cudaMemcpyHostToDevice);

        // Create events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Copy data from GPU to Host
        cudaEventRecord(start, 0);

        for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
            cudaMemcpyAsync(h_odata, d_idata, DEFAULT_SIZE, cudaMemcpyDeviceToHost, 0);
        }

        cudaEventRecord(stop, 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);

        // Calculate bandwidth in GB/s
        double time_s = elapsedTimeInMs / 1e3;
        float bandwidthInGBs = (DEFAULT_SIZE * (float)MEMCOPY_ITERATIONS) / (double)1e9;
        bandwidthInGBs = bandwidthInGBs / time_s;

        // Print result
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "   " << DEFAULT_SIZE << "\t\t" << bandwidthInGBs << "\n\n";

        // Clean up memory
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaFree(d_idata);
        cudaFreeHost(h_idata);
        cudaFreeHost(h_odata);
    }

private:
    float elapsedTimeInMs;
};

// ============================================================
// Host to Device Bandwidth Test (Pinned Memory)
// ============================================================

class HostToDeviceBandwidthTest {
public:
    void execute(const TestConfig& config = TestConfig()) {
        std::cout << "\n===================================================\n";
        std::cout << "Host to Device Bandwidth Test (Pinned Memory)\n";
        std::cout << "===================================================\n\n";

        const unsigned int MEMCOPY_ITERATIONS = 100;
        const unsigned int DEFAULT_SIZE = 32 * 1024 * 1024;  // 32 MB

        std::cout << " Host to Device Bandwidth, 1 Device(s)\n";
        std::cout << " PINNED Memory Transfers\n";
        std::cout << "   Transfer Size (Bytes)\tBandwidth(GB/s)\n";
        std::cout << std::string(45, '-') << "\n";

        // Allocate pinned memory
        unsigned char *h_odata;
        cudaHostAlloc(&h_odata, DEFAULT_SIZE, 0);

        // Initialize the memory
        for (unsigned int i = 0; i < DEFAULT_SIZE / sizeof(unsigned char); i++) {
            h_odata[i] = (unsigned char)(i & 0xff);
        }

        // Allocate device memory
        unsigned char *d_idata;
        cudaMalloc(&d_idata, DEFAULT_SIZE);

        // Create events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Copy host memory to device memory
        cudaEventRecord(start, 0);

        for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
            cudaMemcpyAsync(d_idata, h_odata, DEFAULT_SIZE, cudaMemcpyHostToDevice, 0);
        }

        cudaEventRecord(stop, 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);

        // Calculate bandwidth in GB/s
        double time_s = elapsedTimeInMs / 1e3;
        float bandwidthInGBs = (DEFAULT_SIZE * (float)MEMCOPY_ITERATIONS) / (double)1e9;
        bandwidthInGBs = bandwidthInGBs / time_s;

        // Print result
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "   " << DEFAULT_SIZE << "\t\t" << bandwidthInGBs << "\n\n";

        // Clean up memory
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaFree(d_idata);
        cudaFreeHost(h_odata);
    }

private:
    float elapsedTimeInMs;
};

// ============================================================
// Unified Bandwidth Test Suite
// ============================================================

class BandwidthTestSuite {
public:
    BandwidthTestSuite(bool test_dtod = true, bool test_dtoh = true, bool test_htod = true)
        : test_dtod_(test_dtod), test_dtoh_(test_dtoh), test_htod_(test_htod) {}

    void execute(const TestConfig& config = TestConfig()) {
        if (test_dtod_) {
            DeviceToDeviceBandwidthTest dtod_test;
            dtod_test.execute(config);
        }

        if (test_dtoh_) {
            DeviceToHostBandwidthTest dtoh_test;
            dtoh_test.execute(config);
        }

        if (test_htod_) {
            HostToDeviceBandwidthTest htod_test;
            htod_test.execute(config);
        }
    }

private:
    bool test_dtod_;
    bool test_dtoh_;
    bool test_htod_;
};

} // namespace cuda_perf
