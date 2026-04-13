"""
Performance Tracking Module

Tracks per-request metrics (latency, TTFT, success/failure) and computes
aggregate statistics for the load test run.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RequestResult:
    """Stores the result of a single HTTP request."""
    request_id: int
    latency_ms: float
    ttft_ms: Optional[float]  # Time-To-First-Token in milliseconds
    tpot_ms: Optional[float]  # Time-Per-Output-Token in milliseconds
    success: bool
    error: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class PerformanceTracker:
    """Collects and aggregates performance metrics from load test requests."""

    def __init__(self):
        self._results: List[RequestResult] = []

    def record(
        self,
        request_id: int,
        latency_ms: float,
        ttft_ms: Optional[float],
        tpot_ms: Optional[float],
        success: bool,
        error: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record a single request result."""
        self._results.append(
            RequestResult(
                request_id=request_id,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                success=success,
                error=error,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    @property
    def results(self) -> List[RequestResult]:
        """Return all recorded results."""
        return list(self._results)

    @property
    def total_requests(self) -> int:
        """Total number of recorded requests."""
        return len(self._results)

    @property
    def successful_requests(self) -> int:
        """Number of successful requests."""
        return sum(1 for r in self._results if r.success)

    @property
    def failed_requests(self) -> int:
        """Number of failed requests."""
        return sum(1 for r in self._results if not r.success)

    def average_latency(self) -> float:
        """Compute average latency across all successful requests (ms)."""
        successful = [r.latency_ms for r in self._results if r.success]
        if not successful:
            return 0.0
        return sum(successful) / len(successful)

    def max_latency(self) -> float:
        """Find the maximum latency across all successful requests (ms)."""
        successful = [r.latency_ms for r in self._results if r.success]
        if not successful:
            return 0.0
        return max(successful)

    def min_latency(self) -> float:
        """Find the minimum latency across all successful requests (ms)."""
        successful = [r.latency_ms for r in self._results if r.success]
        if not successful:
            return 0.0
        return min(successful)

    def average_ttft(self) -> float:
        """Compute average Time-To-First-Token across successful requests (ms)."""
        ttft_values = [
            r.ttft_ms for r in self._results if r.success and r.ttft_ms is not None
        ]
        if not ttft_values:
            return 0.0
        return sum(ttft_values) / len(ttft_values)

    def max_ttft(self) -> float:
        """Find the maximum TTFT across all successful requests (ms)."""
        ttft_values = [
            r.ttft_ms for r in self._results if r.success and r.ttft_ms is not None
        ]
        if not ttft_values:
            return 0.0
        return max(ttft_values)

    def average_tpot(self) -> float:
        """Compute average Time-Per-Output-Token (ms)."""
        tpot_values = [
            r.tpot_ms for r in self._results if r.success and r.tpot_ms is not None
        ]
        if not tpot_values:
            return 0.0
        return sum(tpot_values) / len(tpot_values)

    def timeout_violations(self, threshold_ms: float) -> List[RequestResult]:
        """Return list of requests whose latency exceeds the given threshold."""
        return [r for r in self._results if r.latency_ms > threshold_ms]

    def total_prompt_tokens(self) -> int:
        """Total prompt tokens across all successful requests."""
        return sum(r.prompt_tokens for r in self._results if r.success)

    def total_completion_tokens(self) -> int:
        """Total completion tokens across all successful requests."""
        return sum(r.completion_tokens for r in self._results if r.success)

    def total_tokens(self) -> int:
        """Total tokens (prompt + completion) across all successful requests."""
        return self.total_prompt_tokens() + self.total_completion_tokens()

    def tokens_per_second(self) -> float:
        """Compute aggregate completion tokens per second across all successful requests."""
        successful = [r for r in self._results if r.success]
        if not successful:
            return 0.0
        total_completion = sum(r.completion_tokens for r in successful)
        total_time_s = sum(r.latency_ms for r in successful) / 1000.0
        if total_time_s == 0:
            return 0.0
        return total_completion / total_time_s

    def summary(self) -> dict:
        """Return a dictionary summarizing all key performance metrics."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "average_latency_ms": round(self.average_latency(), 2),
            "max_latency_ms": round(self.max_latency(), 2),
            "min_latency_ms": round(self.min_latency(), 2),
            "average_ttft_ms": round(self.average_ttft(), 2),
            "max_ttft_ms": round(self.max_ttft(), 2),
            "average_tpot_ms": round(self.average_tpot(), 2),
            "total_prompt_tokens": self.total_prompt_tokens(),
            "total_completion_tokens": self.total_completion_tokens(),
            "tokens_per_second": round(self.tokens_per_second(), 2),
        }
