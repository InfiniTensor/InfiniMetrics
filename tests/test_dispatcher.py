#!/usr/bin/env python3
"""Tests for Dispatcher class."""

import pytest

from infinimetrics.dispatcher import Dispatcher
from infinimetrics.common.constants import TestCategory


class TestDispatcherParseTestcase:
    """Tests for _parse_testcase method."""

    def setup_method(self):
        self.dispatcher = Dispatcher()

    def test_valid_infer_testcase(self):
        """Test parsing valid inference testcase."""
        test_type, framework = self.dispatcher._parse_testcase("infer.InfiniLM.Direct")
        assert test_type == "infer"
        assert framework == "infinilm"

    def test_valid_operator_testcase(self):
        """Test parsing valid operator testcase."""
        test_type, framework = self.dispatcher._parse_testcase("operator.InfiniCore.Matmul")
        assert test_type == "operator"
        assert framework == "infinicore"

    def test_valid_hardware_testcase(self):
        """Test parsing valid hardware testcase."""
        test_type, framework = self.dispatcher._parse_testcase("hardware.CudaUnified.Test")
        assert test_type == "hardware"
        assert framework == "cudaunified"

    def test_valid_comm_testcase(self):
        """Test parsing valid communication testcase."""
        test_type, framework = self.dispatcher._parse_testcase("comm.NcclTest.AllReduce")
        assert test_type == "comm"
        assert framework == "nccltest"

    def test_invalid_test_type(self):
        """Test that invalid test type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            self.dispatcher._parse_testcase("invalid.SomeFramework.Test")
        assert "Invalid test_type" in str(exc_info.value)

    def test_single_part_testcase(self):
        """Test testcase with only one part uses defaults."""
        test_type, framework = self.dispatcher._parse_testcase("testcase")
        # Should return defaults
        assert test_type == "operator"
        assert framework == "infinicore"

    def test_case_insensitive_framework(self):
        """Test that framework is converted to lowercase."""
        test_type, framework = self.dispatcher._parse_testcase("infer.INFINILM.Direct")
        assert framework == "infinilm"


class TestDispatcherValidateInput:
    """Tests for validate_input method."""

    def setup_method(self):
        self.dispatcher = Dispatcher()

    def test_valid_input(self):
        """Test valid input with testcase."""
        assert self.dispatcher.validate_input({"testcase": "test"}) is True

    def test_missing_testcase(self):
        """Test input without testcase."""
        assert self.dispatcher.validate_input({}) is False
        assert self.dispatcher.validate_input({"config": {}}) is False

    def test_none_testcase(self):
        """Test input with None testcase - current implementation only checks key existence."""
        # Note: current validate_input only checks if "testcase" key exists
        # This test documents the current behavior
        assert self.dispatcher.validate_input({"testcase": None}) is True

    def test_empty_testcase(self):
        """Test input with empty testcase."""
        # Empty string is truthy, so this passes
        assert self.dispatcher.validate_input({"testcase": ""}) is True


class TestDispatcherAggregateResults:
    """Tests for _aggregate_results method."""

    def setup_method(self):
        self.dispatcher = Dispatcher()

    def test_all_success(self):
        """Test aggregation with all successful results."""
        results = [
            {"run_id": "test1", "testcase": "test", "result_code": 0, "result_file": "file1"},
            {"run_id": "test2", "testcase": "test", "result_code": 0, "result_file": "file2"},
        ]
        aggregated = self.dispatcher._aggregate_results(results)
        assert aggregated["total_tests"] == 2
        assert aggregated["successful_tests"] == 2
        assert aggregated["failed_tests"] == 0

    def test_all_failed(self):
        """Test aggregation with all failed results."""
        results = [
            {"run_id": "test1", "testcase": "test", "result_code": 1, "result_file": None},
            {"run_id": "test2", "testcase": "test", "result_code": 2, "result_file": None},
        ]
        aggregated = self.dispatcher._aggregate_results(results)
        assert aggregated["total_tests"] == 2
        assert aggregated["successful_tests"] == 0
        assert aggregated["failed_tests"] == 2

    def test_mixed_results(self):
        """Test aggregation with mixed results."""
        results = [
            {"run_id": "test1", "testcase": "test", "result_code": 0, "result_file": "file1"},
            {"run_id": "test2", "testcase": "test", "result_code": 1, "result_file": None},
        ]
        aggregated = self.dispatcher._aggregate_results(results)
        assert aggregated["total_tests"] == 2
        assert aggregated["successful_tests"] == 1
        assert aggregated["failed_tests"] == 1

    def test_empty_results(self):
        """Test aggregation with no results."""
        aggregated = self.dispatcher._aggregate_results([])
        assert aggregated["total_tests"] == 0
        assert aggregated["successful_tests"] == 0
        assert aggregated["failed_tests"] == 0
