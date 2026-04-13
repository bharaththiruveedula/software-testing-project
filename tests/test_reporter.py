"""
Unit tests for the Terminal Report Printer Module (reporter.py).
"""

import json
from io import StringIO
from unittest.mock import patch

import pytest

from load_runner import LoadTestResult
from metrics import PerformanceTracker
from reporter import print_report
from validator import ValidationResult


def _make_result(
    num_success: int = 5,
    num_fail: int = 0,
    valid_count: int = None,
    latency_ms: float = 250.0,
    ttft_ms: float = 50.0,
) -> LoadTestResult:
    """Helper to build a LoadTestResult for testing."""
    if valid_count is None:
        valid_count = num_success

    tracker = PerformanceTracker()

    for i in range(num_success):
        tracker.record(
            request_id=i + 1,
            latency_ms=latency_ms + i * 10,
            ttft_ms=ttft_ms + i * 5,
            tpot_ms=None,
            success=True,
        )
    for i in range(num_fail):
        tracker.record(
            request_id=num_success + i + 1,
            latency_ms=5000.0,
            ttft_ms=None,
            tpot_ms=None,
            success=False,
            error="Timeout",
        )

    # Build validation results
    validation_results = []
    for i in range(valid_count):
        validation_results.append(ValidationResult(is_valid=True))
    invalid_count = num_success - valid_count
    for i in range(invalid_count):
        validation_results.append(
            ValidationResult(is_valid=False, error_message="Schema validation failed")
        )
    # Failed requests also get validation results
    for i in range(num_fail):
        validation_results.append(
            ValidationResult(is_valid=False, error_message="Request failed")
        )

    return LoadTestResult(
        tracker=tracker,
        validation_results=validation_results,
        raw_responses=[None] * (num_success + num_fail),
    )


class TestPrintReportOutput:
    """Test that print_report produces expected terminal output."""

    def test_all_passed(self, capsys):
        result = _make_result(num_success=10, valid_count=10)
        passed = print_report(result, pass_rate_threshold=0.95)

        assert passed is True
        output = capsys.readouterr().out
        assert "Performance Metrics" in output
        assert "Structural Validation" in output
        assert "PASSED" in output

    def test_below_threshold_fails(self, capsys):
        # 8/10 valid = 80%, below 95%
        result = _make_result(num_success=10, valid_count=8)
        passed = print_report(result, pass_rate_threshold=0.95)

        assert passed is False
        output = capsys.readouterr().out
        assert "FAILED" in output

    def test_all_requests_failed(self, capsys):
        result = _make_result(num_success=0, num_fail=5)
        passed = print_report(result, pass_rate_threshold=0.95)

        assert passed is False
        output = capsys.readouterr().out
        assert "FAILED" in output

    def test_mixed_results(self, capsys):
        result = _make_result(num_success=8, num_fail=2, valid_count=8)
        passed = print_report(result, pass_rate_threshold=0.80)

        # 8/10 total validated, 8 valid = 80%, threshold is 80%
        assert passed is True
        output = capsys.readouterr().out
        assert "Performance Metrics" in output

    def test_custom_threshold_passes(self, capsys):
        # 9/10 valid = 90%, threshold at 90% should pass
        result = _make_result(num_success=10, valid_count=9)
        passed = print_report(result, pass_rate_threshold=0.90)
        assert passed is True

    def test_custom_threshold_fails(self, capsys):
        # 9/10 valid = 90%, threshold at 91% should fail
        result = _make_result(num_success=10, valid_count=9)
        passed = print_report(result, pass_rate_threshold=0.91)
        assert passed is False

    def test_shows_failed_validation_details(self, capsys):
        result = _make_result(num_success=5, valid_count=3)
        print_report(result, pass_rate_threshold=0.95)

        output = capsys.readouterr().out
        assert "Schema validation failed" in output

    def test_shows_total_requests(self, capsys):
        result = _make_result(num_success=20)
        print_report(result)

        output = capsys.readouterr().out
        assert "20" in output


class TestPrintReportReturnValue:
    """Test the boolean return value of print_report."""

    def test_returns_true_on_pass(self, capsys):
        result = _make_result(num_success=10, valid_count=10)
        assert print_report(result, pass_rate_threshold=0.95) is True

    def test_returns_false_on_fail(self, capsys):
        result = _make_result(num_success=10, valid_count=5)
        assert print_report(result, pass_rate_threshold=0.95) is False

    def test_returns_false_when_no_requests(self, capsys):
        result = _make_result(num_success=0, num_fail=0)
        assert print_report(result, pass_rate_threshold=0.95) is False
