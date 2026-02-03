#!/usr/bin/env bash
# Common environment preparation functions
# Source this in test scripts: source scripts/common/prepare_env.sh

set -e

# Create output directory
prepare_output_dir() {
    local test_name="$1"
    local output_dir="./output/${test_name}"

    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
        echo "Created output directory: $output_dir"
    fi
}

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

# Validate JSON input file
validate_json() {
    local json_file="$1"

    if [ ! -f "$json_file" ]; then
        echo "❌ Error: JSON file not found: $json_file"
        return 1
    fi

    # Check if valid JSON
    if python -c "import json; json.load(open('$json_file'))" 2>/dev/null; then
        return 0
    else
        echo "❌ Error: Invalid JSON file: $json_file"
        return 1
    fi
}
