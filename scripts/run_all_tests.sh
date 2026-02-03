#!/bin/bash
# Run all tests in sequence
# Usage: ./scripts/run_all_tests [test_file]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
INPUT_FILE="${1:-}"

# Test suite
TESTS=(
    "hardware"
    "operator"
    "inference"
    "communication"
)

# Counters
TOTAL=0
PASSED=0
FAILED=0

echo "=========================================="
echo "Running All Tests"
echo "=========================================="
echo ""

# Run each test suite
for test in "${TESTS[@]}"; do
    TEST_SCRIPT="${SCRIPT_DIR}/test_${test}.sh"

    if [ ! -f "$TEST_SCRIPT" ]; then
        echo -e "${YELLOW}⚠️  Test script not found: $TEST_SCRIPT${NC}"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Test Suite: $test"
    echo "=========================================="
    echo ""

    if [ -n "$INPUT_FILE" ]; then
        # Run all tests with same input file
        bash "$TEST_SCRIPT" "$INPUT_FILE"
    else
        # Let each test find its own input files
        bash "$TEST_SCRIPT"
    fi

    EXIT_CODE=$?
    ((TOTAL++))

    if [ $EXIT_CODE -eq 0 ]; then
        ((PASSED++))
        echo -e "${GREEN}✓ $test passed${NC}"
    else
        ((FAILED++))
        echo -e "${RED}✗ $test failed${NC}"
    fi

    echo ""
    echo "----------------------------------------"
    echo ""
done

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total:   $TOTAL"
echo -e "${GREEN}Passed:  $PASSED${NC}"
echo -e "${RED}Failed:  $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo "=========================================="
    exit 0
else
    echo ""
    echo -e "${RED}❌ Some tests failed${NC}"
    echo "=========================================="
    exit 1
fi
