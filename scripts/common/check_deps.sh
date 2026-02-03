#!/bin/bash
# Common dependency checking functions
# Source this in test scripts: source scripts/common/check_deps.sh

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

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

# Check NVIDIA Management Library (nvml)
check_nvml() {
    echo -n "  NVIDIA nvml... "
    if check_python_package pynvml; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} not installed (optional)"
        return 0
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

# Check vLLM
check_vllm() {
    echo -n "  vLLM... "
    if check_python_package vllm; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} not installed"
        return 1
    fi
}

# Check InfiniLM
check_infinilm() {
    echo -n "  InfiniLM... "
    if check_python_package infinilm; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} not installed"
        return 1
    fi
}

# General dependency check
check_all_deps() {
    local deps=("$@")
    local missing=0

    echo "Checking dependencies:"

    for dep in "${deps[@]}"; do
        if check_python_package "$dep"; then
            echo -e "  ${GREEN}✓${NC} $dep"
        else
            echo -e "  ${RED}✗${NC} $dep"
            ((missing++))
        fi
    done

    if [ $missing -eq 0 ]; then
        return 0
    else
        return 1
    fi
}
