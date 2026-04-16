"""
Unit tests for the HTML Reporter Module.
"""

import os
from unittest.mock import MagicMock
from html_reporter import generate_html_report, generate_comparison_report
from comparator import ComparisonMetric
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


def test_generate_html_report_escapes_error_messages(tmp_path):
    """Validation error text should be HTML-escaped in report output."""
    report_file = tmp_path / "xss_report.html"

    tracker = PerformanceTracker()
    tracker.record(1, 100.0, 20.0, None, True)
    val_results = [
        ValidationResult(
            is_valid=False,
            error_message="<script>alert('xss')</script>",
        )
    ]
    result = LoadTestResult(
        tracker=tracker,
        validation_results=val_results,
        raw_responses=["res1"],
    )

    generate_html_report(result, output_path=str(report_file))

    content = report_file.read_text(encoding="utf-8")
    assert "<script>alert('xss')</script>" not in content
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in content


def test_generate_comparison_report_escapes_labels(tmp_path):
    """Scenario/model names should be escaped in the comparison table."""
    report_file = tmp_path / "comparison_xss.html"
    metrics = [
        ComparisonMetric(
            scenario_name="<img src=x onerror=alert(1)>",
            target="<b>model</b>",
            users=1,
            throughput=12.3,
            avg_latency=123.0,
            pass_rate=1.0,
            passed=True,
        )
    ]

    generate_comparison_report(metrics, output_path=str(report_file))

    content = report_file.read_text(encoding="utf-8")
    assert "<img src=x onerror=alert(1)>" not in content
    assert "<b>model</b>" not in content
    assert "&lt;img src=x onerror=alert(1)&gt;" in content
    assert "&lt;b&gt;model&lt;/b&gt;" in content
