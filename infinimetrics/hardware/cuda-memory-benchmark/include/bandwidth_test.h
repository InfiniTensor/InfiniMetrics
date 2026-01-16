#pragma once

#include "cuda_utils.h"
#include "performance_test.h"
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cstring>
#include <functional>

namespace cuda_perf {

// ============================================================
// Generic Bandwidth Test - Template-based Implementation
// ============================================================

enum class BandwidthDirection {
    DEVICE_TO_DEVICE,
    DEVICE_TO_HOST,
    HOST_TO_DEVICE
};

template <BandwidthDirection Dir>
class BandwidthTest {
public:
    void execute(const TestConfig& config = TestConfig()) {
        print_header();
        run_test();
        print_result();
    }

private:
    static constexpr unsigned int MEMCOPY_ITERATIONS = 100;
    static constexpr unsigned int DEFAULT_SIZE = 32 * 1024 * 1024;  // 32 MB
    float elapsedTimeInMs = 0.0f;

    void print_header() const {
        std::cout << "\n===================================================\n";
        std::cout << get_test_name() << "\n";
        std::cout << "===================================================\n\n";

        std::cout << get_device_info();
        if constexpr (Dir != BandwidthDirection::DEVICE_TO_DEVICE) {
            std::cout << " PINNED Memory Transfers\n";
        }
        std::cout << "   Transfer Size (Bytes)\tBandwidth(GB/s)\n";
        std::cout << std::string(45, '-') << "\n";
    }

    const char* get_test_name() const {
        if constexpr (Dir == BandwidthDirection::DEVICE_TO_DEVICE) {
            return "Device to Device Bandwidth Test";
        } else if constexpr (Dir == BandwidthDirection::DEVICE_TO_HOST) {
            return "Device to Host Bandwidth Test (Pinned Memory)";
        } else {
            return "Host to Device Bandwidth Test (Pinned Memory)";
        }
    }

    const char* get_device_info() const {
        if constexpr (Dir == BandwidthDirection::DEVICE_TO_DEVICE) {
            return " Device to Device Bandwidth, 1 Device(s)\n";
        } else if constexpr (Dir == BandwidthDirection::DEVICE_TO_HOST) {
            return " Device to Host Bandwidth, 1 Device(s)\n";
        } else {
            return " Host to Device Bandwidth, 1 Device(s)\n";
        }
    }

    void run_test() {
        if constexpr (Dir == BandwidthDirection::DEVICE_TO_DEVICE) {
            run_dtod_test();
        } else if constexpr (Dir == BandwidthDirection::DEVICE_TO_HOST) {
            run_dtoh_test();
        } else {
            run_htod_test();
        }
    }

    void run_dtod_test() {
        unsigned char *h_idata = (unsigned char *)malloc(DEFAULT_SIZE);
        unsigned char *d_idata, *d_odata;

        // Initialize host memory
        for (unsigned int i = 0; i < DEFAULT_SIZE / sizeof(unsigned char); i++) {
            h_idata[i] = (unsigned char)(i & 0xff);
        }

        // Allocate device memory
        cudaMalloc(&d_idata, DEFAULT_SIZE);
        cudaMalloc(&d_odata, DEFAULT_SIZE);
        cudaMemcpy(d_idata, h_idata, DEFAULT_SIZE, cudaMemcpyHostToDevice);

        // Time the copy
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
            cudaMemcpy(d_odata, d_idata, DEFAULT_SIZE, cudaMemcpyDeviceToDevice);
        }
        cudaEventRecord(stop, 0);
        cudaDeviceSynchronize();

        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);

        // Cleanup
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaFree(d_idata);
        cudaFree(d_odata);
        free(h_idata);
    }

    void run_dtoh_test() {
        unsigned char *h_idata, *h_odata;
        unsigned char *d_idata;

        cudaHostAlloc(&h_idata, DEFAULT_SIZE, 0);
        cudaHostAlloc(&h_odata, DEFAULT_SIZE, 0);
        cudaMalloc(&d_idata, DEFAULT_SIZE);

        // Initialize host memory
        for (unsigned int i = 0; i < DEFAULT_SIZE / sizeof(unsigned char); i++) {
            h_idata[i] = (unsigned char)(i & 0xff);
        }
        cudaMemcpy(d_idata, h_idata, DEFAULT_SIZE, cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
            cudaMemcpyAsync(h_odata, d_idata, DEFAULT_SIZE, cudaMemcpyDeviceToHost, 0);
        }
        cudaEventRecord(stop, 0);
        cudaDeviceSynchronize();

        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);

        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaFree(d_idata);
        cudaFreeHost(h_idata);
        cudaFreeHost(h_odata);
    }

    void run_htod_test() {
        unsigned char *h_odata;
        unsigned char *d_idata;

        cudaHostAlloc(&h_odata, DEFAULT_SIZE, 0);

        for (unsigned int i = 0; i < DEFAULT_SIZE / sizeof(unsigned char); i++) {
            h_odata[i] = (unsigned char)(i & 0xff);
        }

        cudaMalloc(&d_idata, DEFAULT_SIZE);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
            cudaMemcpyAsync(d_idata, h_odata, DEFAULT_SIZE, cudaMemcpyHostToDevice, 0);
        }
        cudaEventRecord(stop, 0);
        cudaDeviceSynchronize();

        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);

        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaFree(d_idata);
        cudaFreeHost(h_odata);
    }

    void print_result() const {
        double time_s = elapsedTimeInMs / 1e3;

        float bandwidthInGBs;
        if constexpr (Dir == BandwidthDirection::DEVICE_TO_DEVICE) {
            bandwidthInGBs = (2.0f * DEFAULT_SIZE * MEMCOPY_ITERATIONS) / 1e9;
        } else {
            bandwidthInGBs = (DEFAULT_SIZE * (float)MEMCOPY_ITERATIONS) / 1e9;
        }
        bandwidthInGBs /= time_s;

        std::cout << std::fixed << std::setprecision(1);
        std::cout << "   " << DEFAULT_SIZE << "\t\t" << bandwidthInGBs << "\n\n";
    }
};

// Type aliases for backward compatibility
using DeviceToDeviceBandwidthTest = BandwidthTest<BandwidthDirection::DEVICE_TO_DEVICE>;
using DeviceToHostBandwidthTest = BandwidthTest<BandwidthDirection::DEVICE_TO_HOST>;
using HostToDeviceBandwidthTest = BandwidthTest<BandwidthDirection::HOST_TO_DEVICE>;

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
