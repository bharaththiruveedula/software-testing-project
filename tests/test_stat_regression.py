"""
Test #5 — Statistical performance regression.

Methodology: Statistical hypothesis testing.
Precedent: Hugging Face ``text-generation-inference``'s
``inference-benchmarker`` fixed-request protocol: repeat a workload, compare
p95 latency / TTFT against a saved baseline, and flag a *statistically
significant* regression — not a single-run spike.

Assertion: given a saved baseline of TTFT samples and a new run, the
Mann-Whitney U test must not reject "distributions are equal" with p < 0.05
in the direction "new run slower than baseline". A bootstrap CI on the p95
is reported for context.

This decouples "random-noise blip" from "real regression" — the core reason
flaky perf CI tests end up ignored.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from stat_regression import (
    bootstrap_ci,
    load_baseline,
    mann_whitney_u,
    percentile,
    save_baseline,
)


def gen_latencies(n: int, mean: float, stdev: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [max(1.0, rng.gauss(mean, stdev)) for _ in range(n)]


def assess_regression(
    baseline: list[float],
    new_run: list[float],
    *,
    alpha: float = 0.05,
) -> dict:
    """
    Return an assessment dict.

    - regression: True if new_run is *significantly slower* than baseline at
      level ``alpha`` (one-sided Mann-Whitney greater).
    - p_value: the one-sided p-value for H1: new_run > baseline.
    - baseline_p95 / new_p95: point estimates.
    - new_p95_ci: 95% bootstrap CI on p95 of the new run.
    """
    _, p_value = mann_whitney_u(new_run, baseline, alternative="greater")
    lower, upper = bootstrap_ci(
        new_run,
        statistic=lambda xs: percentile(xs, 95),
        resamples=500,
        rng=random.Random(7),
    )
    return {
        "regression": p_value < alpha,
        "p_value": p_value,
        "baseline_p95": percentile(baseline, 95),
        "new_p95": percentile(new_run, 95),
        "new_p95_ci": (lower, upper),
    }


class TestStatRegression:

    def test_clean_rerun_is_not_a_regression(self):
        baseline = gen_latencies(50, mean=100.0, stdev=10.0, seed=1)
        new_run = gen_latencies(50, mean=100.0, stdev=10.0, seed=2)
        r = assess_regression(baseline, new_run)
        assert not r["regression"], (
            f"False positive: p={r['p_value']}, p95={r['new_p95']:.1f} vs "
            f"baseline {r['baseline_p95']:.1f}"
        )

    def test_real_regression_detected(self):
        baseline = gen_latencies(50, mean=100.0, stdev=10.0, seed=1)
        new_run = gen_latencies(50, mean=130.0, stdev=10.0, seed=2)
        r = assess_regression(baseline, new_run)
        assert r["regression"], f"False negative: p={r['p_value']}"

    def test_improvement_is_not_flagged_as_regression(self):
        baseline = gen_latencies(50, mean=100.0, stdev=10.0, seed=1)
        new_run = gen_latencies(50, mean=70.0, stdev=10.0, seed=2)
        r = assess_regression(baseline, new_run)
        assert not r["regression"], (
            "Speedup was misclassified as a regression — one-sided test is wrong direction"
        )

    def test_single_outlier_doesnt_trip_the_test(self):
        """A handful of slow requests shouldn't fail CI — that's why we use a
        distributional test, not a threshold on max/p99."""
        baseline = gen_latencies(60, mean=100.0, stdev=10.0, seed=1)
        new_run = gen_latencies(55, mean=100.0, stdev=10.0, seed=2)
        new_run += [500.0, 480.0, 520.0, 510.0, 490.0]  # 5 slow outliers
        r = assess_regression(baseline, new_run)
        assert not r["regression"], (
            "Outliers alone triggered regression — test is too sensitive"
        )


class TestBaselineWorkflow:
    """Demonstrates the save-baseline-then-compare workflow end-to-end."""

    def test_roundtrip_and_compare(self, tmp_path: Path):
        baseline_samples = gen_latencies(40, mean=100.0, stdev=12.0, seed=1)
        path = tmp_path / "baselines" / "demo.json"
        save_baseline(path, {
            "scenario": "demo",
            "metric": "ttft_ms",
            "samples": baseline_samples,
        })
        assert path.exists()
        loaded = load_baseline(path)
        assert loaded["samples"] == baseline_samples

        new_run = gen_latencies(40, mean=102.0, stdev=12.0, seed=99)
        r = assess_regression(loaded["samples"], new_run)
        assert not r["regression"]
