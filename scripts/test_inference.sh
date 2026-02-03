#!/bin/bash
# Inference testing script
# Usage: ./scripts/test_inference.sh [input.json]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common/check_deps.sh"
source "${SCRIPT_DIR}/common/prepare_env.sh"

TEST_NAME="inference"

# Parse arguments
INPUT_FILE="${1:-}"

# ========================================
# Part 1: Dependencies
# ========================================
check_inference_deps() {
    echo "Checking inference test dependencies:"
    check_vllm
    check_infinilm
}

# ========================================
# Part 2: Run Tests
# ========================================
run_inference_tests() {
    log_test_start "$TEST_NAME"
    trap 'cleanup_on_error "$TEST_NAME"' ERR

    if [ -n "$INPUT_FILE" ]; then
        # Test specific file
        validate_json "$INPUT_FILE" || exit 1
        echo "Running inference test: $INPUT_FILE"
        python main.py "$INPUT_FILE" --output "./output/${TEST_NAME}/"
    else
        # Find all inference test files
        echo "Searching for inference test files..."
        TEST_FILES=$(find . -name "*inference*.json" -o -name "*vllm*.json" -o -name "*infinilm*.json" | head -5)

        if [ -z "$TEST_FILES" ]; then
            echo "⚠️  No inference test files found"
            return 1
        fi

        echo "Found test files:"
        echo "$TEST_FILES"
        echo ""

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
    echo "Inference Testing"
    echo "=========================================="
    echo ""

    check_inference_deps
    echo ""

    run_inference_tests
}

main "$@"
