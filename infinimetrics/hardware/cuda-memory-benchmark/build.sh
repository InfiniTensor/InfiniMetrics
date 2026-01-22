#!/bin/bash

# Build script for CUDA Performance Suite

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  CUDA Performance Suite - Build Script"
echo "=========================================="
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}ERROR: nvcc not found. Please install CUDA toolkit.${NC}"
    exit 1
fi

# Create build directory
echo -e "${YELLOW}Creating build directory...${NC}"
mkdir -p build
cd build

# Configure with CMake
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo -e "${YELLOW}Building...${NC}"
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Build completed successfully!${NC}"
    echo ""
    echo "Executable: build/cuda_perf_suite"
    echo ""
    echo "Usage:"
    echo "  ./build/cuda_perf_suite --help        # Show help"
    echo "  ./build/cuda_perf_suite --all         # Run all tests"
    echo "  ./build/cuda_perf_suite --memory      # Run memory bandwidth tests"
    echo "  ./build/cuda_perf_suite --stream      # Run STREAM benchmark only"
    echo "  ./build/cuda_perf_suite --cache       # Run cache tests only"
    echo ""
else
    echo ""
    echo -e "${RED}✗ Build failed!${NC}"
    echo "Please check the error messages above."
    exit 1
fi

