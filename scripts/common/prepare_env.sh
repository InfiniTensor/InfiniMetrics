#!/usr/bin/env bash
# Common environment preparation functions
# Source this in test scripts: source scripts/common/prepare_env.sh

set -e

# Cleanup function for trap
cleanup_on_error() {
    local test_name="$1"
    echo ""
    echo "=========================================="
    echo "❌ Test failed: $test_name"
    echo "=========================================="
}

# Get current timestamp
get_timestamp() {
    date "+%Y%m%d_%H%M%S"
}

# Log test start
log_test_start() {
    local test_name="$1"
    echo ""
    echo "=========================================="
    echo "Running: $test_name"
    echo "Time: $(get_timestamp)"
    echo "=========================================="
}

# Log test end
log_test_end() {
    local test_name="$1"
    local exit_code=$2

    echo ""
    echo "=========================================="
    if [ $exit_code -eq 0 ]; then
        echo "✅ Test completed: $test_name"
    else
        echo "❌ Test failed: $test_name (exit code: $exit_code)"
    fi
    echo "Time: $(get_timestamp)"
    echo "=========================================="
    echo ""

    return $exit_code
}
