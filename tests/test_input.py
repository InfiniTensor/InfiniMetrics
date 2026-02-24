#!/usr/bin/env python3
"""Tests for TestInput class."""

import pytest
from datetime import datetime

from infinimetrics.input import TestInput


class TestInputClass:
    """Tests for TestInput dataclass."""

    def test_basic_creation(self):
        """Test basic TestInput creation."""
        test_input = TestInput(testcase="infer.test")
        assert test_input.testcase == "infer.test"
        assert test_input.run_id is None
        assert test_input.config == {}

    def test_auto_time(self):
        """Test that time is auto-generated."""
        test_input = TestInput(testcase="test")
        assert test_input.time is not None
        # Should be parseable as datetime
        parsed = datetime.strptime(test_input.time, "%Y-%m-%d %H:%M:%S")
        assert isinstance(parsed, datetime)

    def test_empty_testcase_raises(self):
        """Test that empty testcase raises ValueError."""
        with pytest.raises(ValueError):
            TestInput(testcase="")

    def test_with_config(self):
        """Test TestInput with config."""
        config = {"model": "test-model", "batch_size": 32}
        test_input = TestInput(testcase="test", config=config)
        assert test_input.config["model"] == "test-model"
        assert test_input.config["batch_size"] == 32

    def test_to_dict(self):
        """Test to_dict method."""
        test_input = TestInput(
            testcase="infer.test",
            run_id="test-123",
            config={"key": "value"}
        )
        result = test_input.to_dict()
        assert result["testcase"] == "infer.test"
        assert result["run_id"] == "test-123"
        assert result["config"]["key"] == "value"

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {
            "testcase": "test.from_dict",
            "run_id": "run-456",
            "config": {"param": 123}
        }
        test_input = TestInput.from_dict(data)
        assert test_input.testcase == "test.from_dict"
        assert test_input.run_id == "run-456"
        assert test_input.config["param"] == 123

    def test_get_config_value(self):
        """Test get_config_value method."""
        test_input = TestInput(
            testcase="test",
            config={"existing_key": "value"}
        )
        assert test_input.get_config_value("existing_key") == "value"
        assert test_input.get_config_value("missing_key") is None
        assert test_input.get_config_value("missing_key", "default") == "default"

    def test_set_config_value(self):
        """Test set_config_value method."""
        test_input = TestInput(testcase="test")
        test_input.set_config_value("new_key", "new_value")
        assert test_input.config["new_key"] == "new_value"

    def test_to_dict_excludes_empty_lists(self):
        """Test that empty lists are excluded from to_dict."""
        test_input = TestInput(testcase="test", metrics=[])
        result = test_input.to_dict()
        assert "metrics" not in result
