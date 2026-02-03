#!/bin/bash
# Hardware testing script
# Usage: ./scripts/test_hardware.sh [input.json]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common/check_deps.sh"
source "${SCRIPT_DIR}/common/prepare_env.sh"

TEST_NAME="hardware"

# Parse arguments
INPUT_FILE="${1:-}"

# ========================================
# Part 1: Dependencies
# ========================================
check_hardware_deps() {
    echo "Checking hardware test dependencies:"
    check_cuda
}

# ========================================
# Part 2: Run Tests
# ========================================
run_hardware_tests() {
    log_test_start "$TEST_NAME"

    # Set trap for cleanup on error
    trap 'cleanup_on_error "$TEST_NAME"' ERR

    # Find test input files
    if [ -n "$INPUT_FILE" ]; then
        # Test specific file
        validate_json "$INPUT_FILE" || exit 1
        echo "Running hardware test: $INPUT_FILE"
        python main.py "$INPUT_FILE" --output "./output/${TEST_NAME}/"
    else
        # Find all hardware test files
        echo "Searching for hardware test files..."
        TEST_FILES=$(find . -name "*hardware*.json" -o -name "*mem*.json" | head -5)

        if [ -z "$TEST_FILES" ]; then
            echo "⚠️  No hardware test files found"
            echo "   Looking for: *hardware*.json, *mem*.json"
            return 1
        fi

        echo "Found test files:"
        echo "$TEST_FILES"
        echo ""

        # Run each test
        for file in $TEST_FILES; do
            echo "Running: $file"
            prepare_output_dir "$TEST_NAME"
            python main.py "$file" --output "./output/${TEST_NAME}/"
        done
    fi

    log_test_end "$TEST_NAME" $?
}

# ========================================
# Main
# ========================================
main() {
    echo "=========================================="
    echo "Hardware Testing"
    echo "=========================================="
    echo ""

    # Check dependencies
    check_hardware_deps
    echo ""

    # Run tests
    run_hardware_tests
}

main "$@"
