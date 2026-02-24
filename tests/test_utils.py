#!/usr/bin/env python3
"""Tests for utility functions."""

import pytest
from datetime import datetime

from infinimetrics.utils.path_utils import sanitize_filename
from infinimetrics.utils.time_utils import get_timestamp


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    # ====== Basic cases ======
    @pytest.mark.parametrize("input_val,expected", [
        ("test_file", "test_file"),
        ("TestFile123", "TestFile123"),
        ("test-file.json", "test-file.json"),
        ("v1.2.3", "v1.2.3"),
    ])
    def test_valid_characters_preserved(self, input_val, expected):
        """Test that valid characters are preserved."""
        assert sanitize_filename(input_val) == expected

    # ====== Special character replacement ======
    @pytest.mark.parametrize("input_val,expected", [
        ("test/file", "test_file"),
        ("test:file", "test_file"),
        ("test file", "test_file"),
        ("test|file", "test_file"),
        ("test*file", "test_file"),
        ("test?file", "test_file"),
        ("test<file>", "test_file"),
    ])
    def test_special_characters_replaced(self, input_val, expected):
        """Test that special characters are replaced with underscore."""
        assert sanitize_filename(input_val) == expected

    # ====== Multiple special chars ======
    @pytest.mark.parametrize("input_val,expected", [
        ("test///file", "test_file"),
        ("test   file", "test_file"),
        ("a///b///c", "a_b_c"),
        ("  test  ", "test"),
    ])
    def test_multiple_special_chars_consolidated(self, input_val, expected):
        """Test multiple consecutive special characters are consolidated."""
        assert sanitize_filename(input_val) == expected

    # ====== Edge cases ======
    @pytest.mark.parametrize("input_val,expected", [
        ("", "unknown"),
        ("   ", "unknown"),
        (None, "unknown"),
        ("___", "unknown"),  # Only underscores stripped -> empty -> unknown
        ("...", "..."),  # Dots are preserved
        ("---", "---"),  # Dashes are preserved
    ])
    def test_edge_cases(self, input_val, expected):
        """Test edge cases return expected values."""
        assert sanitize_filename(input_val) == expected

    # ====== Type coercion ======
    @pytest.mark.parametrize("input_val,expected", [
        (123, "123"),
        (45.67, "45.67"),
        (0, "0"),
        (-1, "-1"),
        (True, "True"),
    ])
    def test_non_string_input(self, input_val, expected):
        """Test that non-string inputs are converted to string."""
        assert sanitize_filename(input_val) == expected

    # ====== Unicode handling ======
    def test_unicode_preserved(self):
        r"""Test unicode characters are preserved (regex \w includes unicode in Python 3)."""
        result = sanitize_filename("测试文件")
        assert result == "测试文件"

    def test_mixed_unicode_and_special(self):
        """Test mixed unicode and special characters."""
        result = sanitize_filename("测试/文件")
        assert result == "测试_文件"


class TestGetTimestamp:
    """Tests for get_timestamp function."""

    def test_format_parseable(self):
        """Test timestamp format is parseable."""
        ts = get_timestamp()
        parsed = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        assert isinstance(parsed, datetime)

    def test_returns_string(self):
        """Test that function returns a string."""
        assert isinstance(get_timestamp(), str)

    def test_length(self):
        """Test timestamp length is 19 characters."""
        assert len(get_timestamp()) == 19

    def test_format_pattern(self):
        """Test timestamp matches expected pattern."""
        import re
        ts = get_timestamp()
        pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
        assert re.match(pattern, ts) is not None

    def test_reasonable_year(self):
        """Test year is reasonable (2020-2100)."""
        ts = get_timestamp()
        year = int(ts[:4])
        assert 2020 <= year <= 2100
