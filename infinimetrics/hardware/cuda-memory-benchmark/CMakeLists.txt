cmake_minimum_required(VERSION 3.18)
project(CudaPerfSuite VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures")

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(CUDA_SOURCES
    src/main.cu
)

set(CXX_SOURCES
    # Add any .cpp files here if needed
)

# Create executable
add_executable(cuda_perf_suite
    ${CUDA_SOURCES}
    ${CXX_SOURCES}
)

# Link libraries
target_link_libraries(cuda_perf_suite
    ${CUDA_LIBRARIES}
    ${CMAKE_THREADS_LIB_INIT}
)

# Installation
install(TARGETS cuda_perf_suite
    RUNTIME DESTINATION bin
)

# Print configuration
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project:         ${PROJECT_NAME}")
message(STATUS "  Version:         ${PROJECT_VERSION}")
message(STATUS "  Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "")
