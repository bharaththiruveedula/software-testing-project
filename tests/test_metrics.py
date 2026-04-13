"""
Unit tests for the Performance Tracking Module (metrics.py).
"""

import pytest
from metrics import PerformanceTracker


class TestPerformanceTrackerEmpty:
    """Test behavior with no recorded results."""

    def test_total_requests_empty(self):
        tracker = PerformanceTracker()
        assert tracker.total_requests == 0

    def test_successful_requests_empty(self):
        tracker = PerformanceTracker()
        assert tracker.successful_requests == 0

    def test_failed_requests_empty(self):
        tracker = PerformanceTracker()
        assert tracker.failed_requests == 0

    def test_average_latency_empty(self):
        tracker = PerformanceTracker()
        assert tracker.average_latency() == 0.0

    def test_max_latency_empty(self):
        tracker = PerformanceTracker()
        assert tracker.max_latency() == 0.0

    def test_min_latency_empty(self):
        tracker = PerformanceTracker()
        assert tracker.min_latency() == 0.0

    def test_average_ttft_empty(self):
        tracker = PerformanceTracker()
        assert tracker.average_ttft() == 0.0

    def test_max_ttft_empty(self):
        tracker = PerformanceTracker()
        assert tracker.max_ttft() == 0.0

    def test_timeout_violations_empty(self):
        tracker = PerformanceTracker()
        assert tracker.timeout_violations(1000) == []

    def test_summary_empty(self):
        tracker = PerformanceTracker()
        s = tracker.summary()
        assert s["total_requests"] == 0
        assert s["successful_requests"] == 0
        assert s["average_latency_ms"] == 0.0
        assert s["total_prompt_tokens"] == 0
        assert s["total_completion_tokens"] == 0
        assert s["tokens_per_second"] == 0.0


class TestPerformanceTrackerSingleRequest:
    """Test behavior with a single recorded result."""

    def test_single_successful_request(self):
        tracker = PerformanceTracker()
        tracker.record(request_id=1, latency_ms=250.0, ttft_ms=50.0, tpot_ms=10.0, success=True)

        assert tracker.total_requests == 1
        assert tracker.successful_requests == 1
        assert tracker.failed_requests == 0
        assert tracker.average_latency() == 250.0
        assert tracker.max_latency() == 250.0
        assert tracker.min_latency() == 250.0
        assert tracker.average_ttft() == 50.0

    def test_single_failed_request(self):
        tracker = PerformanceTracker()
        tracker.record(
            request_id=1,
            latency_ms=5000.0,
            ttft_ms=None,
            tpot_ms=None,
            success=False,
            error="Timeout",
        )

        assert tracker.total_requests == 1
        assert tracker.successful_requests == 0
        assert tracker.failed_requests == 1
        # Average latency should be 0 since no successful requests
        assert tracker.average_latency() == 0.0
        assert tracker.average_ttft() == 0.0


class TestPerformanceTrackerMultipleRequests:
    """Test behavior with multiple recorded results."""

    @pytest.fixture
    def tracker_with_data(self):
        tracker = PerformanceTracker()
        tracker.record(request_id=1, latency_ms=100.0, ttft_ms=20.0, tpot_ms=10.0, success=True)
        tracker.record(request_id=2, latency_ms=200.0, ttft_ms=40.0, tpot_ms=20.0, success=True)
        tracker.record(request_id=3, latency_ms=300.0, ttft_ms=60.0, tpot_ms=None, success=True)
        tracker.record(
            request_id=4,
            latency_ms=5000.0,
            ttft_ms=None,
            tpot_ms=None,
            success=False,
            error="Timeout",
        )
        tracker.record(request_id=5, latency_ms=150.0, ttft_ms=30.0, tpot_ms=30.0, success=True)
        return tracker

    def test_total_requests(self, tracker_with_data):
        assert tracker_with_data.total_requests == 5

    def test_successful_requests(self, tracker_with_data):
        assert tracker_with_data.successful_requests == 4

    def test_failed_requests(self, tracker_with_data):
        assert tracker_with_data.failed_requests == 1

    def test_average_latency(self, tracker_with_data):
        # Average of successful: (100 + 200 + 300 + 150) / 4 = 187.5
        assert tracker_with_data.average_latency() == pytest.approx(187.5)

    def test_max_latency(self, tracker_with_data):
        # Max of successful: 300
        assert tracker_with_data.max_latency() == 300.0

    def test_min_latency(self, tracker_with_data):
        # Min of successful: 100
        assert tracker_with_data.min_latency() == 100.0

    def test_average_ttft(self, tracker_with_data):
        # Average of successful TTFT: (20 + 40 + 60 + 30) / 4 = 37.5
        assert tracker_with_data.average_ttft() == pytest.approx(37.5)

    def test_max_ttft(self, tracker_with_data):
        assert tracker_with_data.max_ttft() == 60.0

    def test_timeout_violations(self, tracker_with_data):
        violations = tracker_with_data.timeout_violations(1000)
        assert len(violations) == 1
        assert violations[0].request_id == 4

    def test_timeout_violations_low_threshold(self, tracker_with_data):
        violations = tracker_with_data.timeout_violations(150)
        # Request #2 (200ms), #3 (300ms), #4 (5000ms) exceed 150ms
        assert len(violations) == 3

    def test_summary_keys(self, tracker_with_data):
        s = tracker_with_data.summary()
        expected_keys = {
            "total_requests",
            "successful_requests",
            "failed_requests",
            "average_latency_ms",
            "max_latency_ms",
            "min_latency_ms",
            "average_ttft_ms",
            "max_ttft_ms",
            "average_tpot_ms",
            "total_prompt_tokens",
            "total_completion_tokens",
            "tokens_per_second",
        }
        assert set(s.keys()) == expected_keys

    def test_tokens_per_second(self, tracker_with_data):
        # 4 successful requests: #1(100ms), #2(200ms), #3(300ms), #5(150ms)
        # Total time = 750ms = 0.75s
        # Let's say we had completion tokens
        tracker_with_data.record(request_id=6, latency_ms=250.0, ttft_ms=50.0, tpot_ms=None, success=True, completion_tokens=50)
        # Now we have 5 successful requests. 
        # Total latency = 750 + 250 = 1000ms = 1.0s
        # Total completion tokens = 50 (since previous ones defaulted to 0)
        # throughput = 50 / 1.0 = 50.0
        assert tracker_with_data.tokens_per_second() == pytest.approx(50.0)

    def test_results_property(self, tracker_with_data):
        results = tracker_with_data.results
        assert len(results) == 5
        # Verify it's a copy (modifying returned list shouldn't affect tracker)
        results.pop()
        assert tracker_with_data.total_requests == 5


class TestPerformanceTrackerEdgeCases:
    """Edge case tests."""

    def test_all_failed_requests(self):
        tracker = PerformanceTracker()
        for i in range(5):
            tracker.record(
                request_id=i, latency_ms=1000.0, ttft_ms=None, tpot_ms=None, success=False, error="err"
            )
        assert tracker.successful_requests == 0
        assert tracker.average_latency() == 0.0
        assert tracker.average_ttft() == 0.0

    def test_none_ttft_not_counted(self):
        tracker = PerformanceTracker()
        tracker.record(request_id=1, latency_ms=100.0, ttft_ms=None, tpot_ms=None, success=True)
        tracker.record(request_id=2, latency_ms=200.0, ttft_ms=50.0, tpot_ms=None, success=True)
        # Only request #2 has TTFT, so average should be 50.0
        assert tracker.average_ttft() == 50.0

    def test_summary_rounding(self):
        tracker = PerformanceTracker()
        tracker.record(request_id=1, latency_ms=100.333, ttft_ms=20.666, tpot_ms=None, success=True)
        tracker.record(request_id=2, latency_ms=200.777, ttft_ms=40.111, tpot_ms=None, success=True)
        s = tracker.summary()
        # Values should be rounded to 2 decimal places
        assert s["average_latency_ms"] == 150.56  # (100.333+200.777)/2 = 150.555
        assert s["average_ttft_ms"] == 30.39  # (20.666+40.111)/2 = 30.3885
