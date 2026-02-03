#!/bin/bash
# Unified dependency management script for InfiniMetrics
# Usage: ./scripts/common/install_deps.sh [component]
#
# Components:
#   operator   - InfiniCore (operator testing)
#   hardware   - CUDA memory benchmark (hardware testing)
#   all        - All components (default)
#
# Each component will:
#   1. Initialize environment variables
#   2. Check if dependencies are installed
#   3. Install if missing

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ========================================
# Dependency Checking Functions
# ========================================

# Check if a command exists
check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Check Python package
check_python_package() {
    python -c "import $1" 2>/dev/null
}

# Check CUDA
check_cuda() {
    echo -n "  CUDA... "
    if check_command nvcc; then
        echo -e "${GREEN}✓${NC} (nvcc $(nvcc --version | awk '{print $5}'))"
        return 0
    else
        echo -e "${RED}✗${NC} not found"
        return 1
    fi
}

# Check InfiniCore
check_infinicore() {
    echo -n "  InfiniCore... "
    if check_python_package infinicore; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} not installed"
        return 1
    fi
}

# ========================================
# InfiniCore (Operator Testing)
# ========================================
install_infinicore() {
    echo ""
    echo "=========================================="
    echo "InfiniCore (Operator)"
    echo "=========================================="
    echo ""

    # 1. Initialize environment variables
    export INFINI_ROOT="$HOME/.infini"
    export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$LD_LIBRARY_PATH"
    echo -e "${BLUE}Environment:${NC}"
    echo "  INFINI_ROOT: $INFINI_ROOT"
    echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    echo ""

    # 2. Check if INFINICORE_PATH is set
    if [ -z "$INFINICORE_PATH" ]; then
        echo -e "${RED}❌ INFINICORE_PATH environment variable not set${NC}"
        echo ""
        echo "Please set INFINICORE_PATH environment variable:"
        echo "  export INFINICORE_PATH=/path/to/InfiniCore"
        echo ""
        echo "Example:"
        echo "  export INFINICORE_PATH=\"\$HOME/workplace/random_input/InfiniCore\""
        echo "  $0 operator"
        exit 1
    fi

    # 3. Check if InfiniCore is installed
    echo -e "${BLUE}Checking InfiniCore...${NC}"
    if python -c "import infinicore" 2>/dev/null; then
        echo -e "${GREEN}✓ InfiniCore already installed${NC}"
        return 0
    fi

    echo -e "${YELLOW}⚠ InfiniCore not found, installing...${NC}"
    echo ""

    # 4. Install InfiniCore
    if [ ! -d "$INFINICORE_PATH" ]; then
        echo -e "${RED}❌ InfiniCore not found at: $INFINICORE_PATH${NC}"
        exit 1
    fi

    echo -e "${BLUE}InfiniCore path: $INFINICORE_PATH${NC}"
    echo ""

    cd "$INFINICORE_PATH"

    echo -e "${BLUE}Step 1: Installing Python dependencies...${NC}"
    # TODO: Currently hardcoded for NVIDIA platform (--nv-gpu=y)
    # Future: Detect hardware platform from environment variables (XPU, NPU, etc.)
    python scripts/install.py --nv-gpu=y
    echo ""

    echo -e "${BLUE}Step 2: Building InfiniCore...${NC}"
    xmake build _infinicore
    echo ""

    echo -e "${BLUE}Step 3: Installing InfiniCore...${NC}"
    xmake install _infinicore
    echo ""

    echo -e "${BLUE}Step 4: Installing Python package...${NC}"
    pip install -e .
    echo ""

    # 5. Verify installation
    echo -e "${BLUE}Verifying installation...${NC}"
    if python -c "import infinicore" 2>/dev/null; then
        echo -e "${GREEN}✅ InfiniCore installation completed successfully!${NC}"
    else
        echo -e "${RED}❌ InfiniCore installation verification failed${NC}"
        exit 1
    fi
}

# ========================================
# CUDA Memory Benchmark (Hardware Testing)
# ========================================
install_hardware() {
    echo ""
    echo "=========================================="
    echo "CUDA Memory Benchmark (Hardware)"
    echo "=========================================="
    echo ""

    # 1. Initialize environment variables
    local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    local BENCHMARK_PATH="$PROJECT_ROOT/infinimetrics/hardware/cuda-memory-benchmark"

    # 2. Check if already built
    echo -e "${BLUE}Checking CUDA memory benchmark...${NC}"
    if [ -f "$BENCHMARK_PATH/build/cu_mem_test" ]; then
        echo -e "${GREEN}✓ CUDA memory benchmark already built${NC}"
        return 0
    fi

    echo -e "${YELLOW}⚠ CUDA memory benchmark not found, building...${NC}"
    echo ""

    # 3. Build
    if [ ! -d "$BENCHMARK_PATH" ]; then
        echo -e "${RED}❌ CUDA memory benchmark not found at: $BENCHMARK_PATH${NC}"
        exit 1
    fi

    echo -e "${BLUE}Benchmark path: $BENCHMARK_PATH${NC}"
    echo ""

    cd "$BENCHMARK_PATH"

    echo -e "${BLUE}Step 1: Cleaning previous build...${NC}"
    rm -rf build
    echo ""

    echo -e "${BLUE}Step 2: Building benchmark...${NC}"
    bash build.sh
    echo ""

    # 4. Verify
    if [ -f "$BENCHMARK_PATH/build/cu_mem_test" ]; then
        echo -e "${GREEN}✅ CUDA memory benchmark built successfully!${NC}"
    else
        echo -e "${RED}❌ CUDA memory benchmark build failed${NC}"
        exit 1
    fi
}

# ========================================
# Main
# ========================================
show_usage() {
    echo "Usage: $0 [component]"
    echo ""
    echo "Components:"
    echo "  operator   - InfiniCore (operator testing)"
    echo "  hardware   - CUDA memory benchmark (hardware testing)"
    echo "  all        - All components (default)"
    echo ""
    echo "Environment variables:"
    echo "  INFINICORE_PATH  - Path to InfiniCore source (required for operator)"
    echo ""
    echo "Examples:"
    echo "  export INFINICORE_PATH=\"\$HOME/workplace/random_input/InfiniCore\""
    echo "  $0 operator   # Install InfiniCore"
    echo "  $0 hardware   # Build CUDA benchmark"
    echo "  $0 all        # Install everything"
    echo "  $0            # Install everything (default)"
}

main() {
    local COMPONENT="${1:-all}"

    echo "=========================================="
    echo "InfiniMetrics Dependency Manager"
    echo "=========================================="
    echo ""

    case "$COMPONENT" in
        operator)
            install_infinicore
            ;;
        hardware)
            install_hardware
            ;;
        all)
            install_infinicore
            install_hardware
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown component: $COMPONENT${NC}"
            echo ""
            show_usage
            exit 1
            ;;
    esac

    echo ""
    echo "=========================================="
    echo -e "${GREEN}✅ All operations completed successfully!${NC}"
    echo "=========================================="
}

# Only run main() if this script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
