#pragma once

#include "performance_test.h"
#include "cuda_utils.h"
#include <vector>
#include <algorithm>

namespace cuda_perf {

enum class CopyDirection {
    HOST_TO_DEVICE,
    DEVICE_TO_HOST,
    DEVICE_TO_DEVICE,
    BIDIRECTIONAL
};

class MemoryCopyTest : public PerformanceTest {
public:
    MemoryCopyTest(size_t buffer_size_bytes, CopyDirection direction,
                   bool use_pinned_memory = true)
        : buffer_size_(buffer_size_bytes)
        , direction_(direction)
        , use_pinned_(use_pinned_memory)
        , device_id_(0) {
    }

    std::string name() const override {
        std::string dir_str;
        switch (direction_) {
            case CopyDirection::HOST_TO_DEVICE:
                dir_str = "HostToDevice";
                break;
            case CopyDirection::DEVICE_TO_HOST:
                dir_str = "DeviceToHost";
                break;
            case CopyDirection::DEVICE_TO_DEVICE:
                dir_str = "DeviceToDevice";
                break;
            case CopyDirection::BIDIRECTIONAL:
                dir_str = "Bidirectional";
                break;
        }
        return "MemoryCopy_" + dir_str +
               (use_pinned_ ? "_Pinned" : "_Pageable");
    }

    std::string description() const override {
        std::ostringstream oss;
        oss << "Testing " << name() << " copy bandwidth\n";
        oss << "Buffer size: " << (buffer_size_ / 1024.0 / 1024.0) << " MB";
        return oss.str();
    }

    bool initialize() override {
        // Get device count and set device
        int device_count = get_device_count();
        if (device_count == 0) {
            std::cerr << "No CUDA devices found" << std::endl;
            return false;
        }
        CUDA_CHECK(cudaSetDevice(device_id_));

        // Allocate device memory
        d_src_.resize(buffer_size_ / sizeof(float));
        d_dst_.resize(buffer_size_ / sizeof(float));

        if (!d_src_.is_valid() || !d_dst_.is_valid()) {
            std::cerr << "Failed to allocate device memory" << std::endl;
            return false;
        }

        // Allocate host memory
        if (use_pinned_) {
            h_src_.resize(buffer_size_ / sizeof(float));
            h_dst_.resize(buffer_size_ / sizeof(float));
        } else {
            h_pageable_.resize(buffer_size_ / sizeof(float));
            h_dst_pageable_.resize(buffer_size_ / sizeof(float));
        }

        // Initialize host data with pattern
        std::vector<float> pattern(buffer_size_ / sizeof(float));
        for (size_t i = 0; i < pattern.size(); ++i) {
            pattern[i] = static_cast<float>(i % 1024);
        }

        float* h_src_ptr = use_pinned_ ? h_src_.data() : h_pageable_.data();
        std::copy(pattern.begin(), pattern.end(), h_src_ptr);

        // Copy initial data to device
        CUDA_CHECK(cudaMemcpy(d_src_.data(), h_src_ptr, buffer_size_,
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaDeviceSynchronize());
        return true;
    }

    void cleanup() override {
        d_src_.resize(0);
        d_dst_.resize(0);
        h_src_.resize(0);
        h_dst_.resize(0);
        h_pageable_.clear();
        h_dst_pageable_.clear();
    }

    void run_warmup() override {
        Timer timer;
        if (direction_ == CopyDirection::HOST_TO_DEVICE ||
            direction_ == CopyDirection::BIDIRECTIONAL) {
            float* h_src = use_pinned_ ? h_src_.data() : h_pageable_.data();
            cudaMemcpyAsync(d_dst_.data(), h_src, buffer_size_,
                           cudaMemcpyHostToDevice, stream_.get());
        }
        if (direction_ == CopyDirection::DEVICE_TO_HOST ||
            direction_ == CopyDirection::BIDIRECTIONAL) {
            float* h_dst = use_pinned_ ? h_dst_.data() : h_dst_pageable_.data();
            cudaMemcpyAsync(h_dst, d_src_.data(), buffer_size_,
                           cudaMemcpyDeviceToHost, stream_.get());
        }
        if (direction_ == CopyDirection::DEVICE_TO_DEVICE) {
            cudaMemcpyAsync(d_dst_.data(), d_src_.data(), buffer_size_,
                           cudaMemcpyDeviceToDevice, stream_.get());
        }
        stream_.synchronize();
    }

    double run_single_test() override {
        Timer timer;

        if (direction_ == CopyDirection::HOST_TO_DEVICE) {
            float* h_src = use_pinned_ ? h_src_.data() : h_pageable_.data();
            CUDA_CHECK(cudaMemcpyAsync(d_dst_.data(), h_src, buffer_size_,
                                       cudaMemcpyHostToDevice, stream_.get()));
            stream_.synchronize();
        }
        else if (direction_ == CopyDirection::DEVICE_TO_HOST) {
            float* h_dst = use_pinned_ ? h_dst_.data() : h_dst_pageable_.data();
            CUDA_CHECK(cudaMemcpyAsync(h_dst, d_src_.data(), buffer_size_,
                                       cudaMemcpyDeviceToHost, stream_.get()));
            stream_.synchronize();
        }
        else if (direction_ == CopyDirection::DEVICE_TO_DEVICE) {
            CUDA_CHECK(cudaMemcpyAsync(d_dst_.data(), d_src_.data(), buffer_size_,
                                       cudaMemcpyDeviceToDevice, stream_.get()));
            stream_.synchronize();
        }
        else if (direction_ == CopyDirection::BIDIRECTIONAL) {
            float* h_src = use_pinned_ ? h_src_.data() : h_pageable_.data();
            float* h_dst = use_pinned_ ? h_dst_.data() : h_dst_pageable_.data();
            CUDA_CHECK(cudaMemcpyAsync(d_dst_.data(), h_src, buffer_size_,
                                       cudaMemcpyHostToDevice, stream_.get()));
            CUDA_CHECK(cudaMemcpyAsync(h_dst, d_src_.data(), buffer_size_,
                                       cudaMemcpyDeviceToHost, stream_.get()));
            stream_.synchronize();
        }

        return timer.elapsed_milliseconds();
    }

protected:
    void print_results(const PerformanceMetrics& metrics) override {
        double avg_time_sec = metrics.trimmed_mean() / 1000.0;
        double data_gb = buffer_size_ / (1024.0 * 1024.0 * 1024.0);

        double bandwidth_gb_s = data_gb / avg_time_sec;
        if (direction_ == CopyDirection::BIDIRECTIONAL) {
            bandwidth_gb_s *= 2.0;  // Both directions
        }

        std::cout << "\nResults:\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Average time:      " << metrics.trimmed_mean() << " ms\n";
        std::cout << "  Bandwidth:         " << std::setprecision(2)
                  << bandwidth_gb_s << " GB/s\n";
        std::cout << "  Buffer size:       " << std::setprecision(2)
                  << data_gb << " GB\n";
        std::cout << "  Coeff. of variation: " << std::setprecision(2)
                  << (metrics.coefficient_of_variation() * 100.0) << "%\n";
    }

private:
    size_t buffer_size_;
    CopyDirection direction_;
    bool use_pinned_;
    int device_id_;

    CudaDeviceBuffer<float> d_src_;
    CudaDeviceBuffer<float> d_dst_;
    CudaPinnedBuffer<float> h_src_;
    CudaPinnedBuffer<float> h_dst_;
    std::vector<float> h_pageable_;
    std::vector<float> h_dst_pageable_;
    CudaStream stream_;
};

// Memory copy bandwidth sweep test (multiple sizes)
class MemoryCopySweepTest {
public:
    MemoryCopySweepTest(CopyDirection direction, bool use_pinned = true)
        : direction_(direction), use_pinned_(use_pinned) {}

    void execute(const TestConfig& config = TestConfig()) {
        std::cout << "\n===================================================\n";
        std::cout << "Memory Copy Bandwidth Sweep Test\n";
        std::cout << "Direction: ";
        switch (direction_) {
            case CopyDirection::HOST_TO_DEVICE:
                std::cout << "Host to Device"; break;
            case CopyDirection::DEVICE_TO_HOST:
                std::cout << "Device to Host"; break;
            case CopyDirection::DEVICE_TO_DEVICE:
                std::cout << "Device to Device"; break;
            case CopyDirection::BIDIRECTIONAL:
                std::cout << "Bidirectional"; break;
        }
        std::cout << "\nMemory Type: " << (use_pinned_ ? "Pinned" : "Pageable");
        std::cout << "\n===================================================\n\n";

        // Print table header
        std::cout << std::left << std::setw(15) << "Size (MB)"
                  << std::right << std::setw(12) << "Time (ms)"
                  << std::setw(15) << "Bandwidth (GB/s)"
                  << std::setw(12) << "CV (%)\n";
        std::cout << std::string(54, '-') << "\n";

        // Test different sizes
        std::vector<size_t> sizes_kb = {
            64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
            32768, 65536, 131072, 262144, 524288, 1048576
        };

        for (size_t size_kb : sizes_kb) {
            size_t size_bytes = size_kb * 1024;

            MemoryCopyTest test(size_bytes, direction_, use_pinned_);
            TestConfig test_config = config;
            test_config.measurement_iterations = 10;
            test_config.verbose = false;

            // Run test and collect metrics
            if (!test.initialize()) {
                std::cerr << "Failed to initialize test for size " << size_kb << " KB" << std::endl;
                continue;
            }

            for (size_t i = 0; i < test_config.warmup_iterations; ++i) {
                test.run_warmup();
            }

            PerformanceMetrics metrics;
            for (size_t i = 0; i < test_config.measurement_iterations; ++i) {
                double time_ms = test.run_single_test();
                metrics.add_sample(time_ms);
            }

            test.cleanup();

            // Calculate bandwidth
            double avg_time_sec = metrics.trimmed_mean() / 1000.0;
            double data_gb = size_bytes / (1024.0 * 1024.0 * 1024.0);
            double bandwidth_gb_s = data_gb / avg_time_sec;
            if (direction_ == CopyDirection::BIDIRECTIONAL) {
                bandwidth_gb_s *= 2.0;
            }

            // Print row
            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::left << std::setw(15) << (size_bytes / 1024.0 / 1024.0);
            std::cout << std::right << std::setw(12) << metrics.trimmed_mean();
            std::cout << std::setw(15) << bandwidth_gb_s;
            std::cout << std::setw(12) << (metrics.coefficient_of_variation() * 100.0);
            std::cout << "\n";
        }
        std::cout << "\n";
    }

private:
    CopyDirection direction_;
    bool use_pinned_;
};

} // namespace cuda_perf
