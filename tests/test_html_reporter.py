"""
Unit tests for the HTML Reporter Module.
"""

import os
from unittest.mock import MagicMock
from html_reporter import generate_html_report
from load_runner import LoadTestResult
from metrics import PerformanceTracker
from validator import ValidationResult


def test_generate_html_report_success(tmp_path):
    """Test generating an HTML report with valid data."""
    report_file = tmp_path / "test_report.html"
    
    tracker = PerformanceTracker()
    tracker.record(1, 100.0, 20.0, None, True, prompt_tokens=10, completion_tokens=20)
    tracker.record(2, 200.0, 40.0, None, True, prompt_tokens=10, completion_tokens=20)
    
    val_results = [
        ValidationResult(is_valid=True),
        ValidationResult(is_valid=False, error_message="Missing key"),
    ]
    
    result = LoadTestResult(
        tracker=tracker,
        validation_results=val_results,
        raw_responses=["res1", "res2"],
    )
    
    generate_html_report(result, output_path=str(report_file))
    
    assert report_file.exists()
    content = report_file.read_text()
    
    # Check if necessary components are injected
    assert "<title>LLM Load Test Report</title>" in content
    assert "chart.js" in content
    # Values check
    assert "150.0ms" in content  # avg latency
    assert "Missing key" in content  # failed validation


def test_generate_html_report_empty(tmp_path):
    """Test generating a report when no data is available."""
    report_file = tmp_path / "empty_report.html"
    
    tracker = PerformanceTracker()
    result = LoadTestResult(
        tracker=tracker,
        validation_results=[],
        raw_responses=[],
    )
    
    generate_html_report(result, output_path=str(report_file))
    
    assert report_file.exists()
    content = report_file.read_text()
    assert "LLM Inference Test Report" in content
    assert "No validation failures." in content
