"""
Unit tests for the pure statistical harness (src/stat_regression.py).

These tests verify the math itself — bootstrap CI, Mann-Whitney U, percentile,
and baseline save/load. Server-centric performance-regression behaviour that
*uses* this harness is covered by tests/test_stat_regression.py.

``TestStatHarnessMutationKills`` pins reference values computed from the
textbook Mann-Whitney formula and percentile interpolation. These are what
give the mutation-testing run its teeth: the loose "p < 0.05 for different
distributions" style of check can't tell the difference between a correct
formula and one with the wrong tie-correction denominator, but a pinned
p-value can.
"""

import json
import math
import random
import statistics
from pathlib import Path

import pytest

from stat_regression import (
    bootstrap_ci,
    load_baseline,
    mann_whitney_u,
    percentile,
    save_baseline,
)


class TestPercentile:
    def test_single_value(self):
        assert percentile([42.0], 50) == 42.0

    def test_median_of_known_sample(self):
        assert percentile([1, 2, 3, 4, 5], 50) == 3.0

    def test_boundaries(self):
        xs = [10, 20, 30, 40, 50]
        assert percentile(xs, 0) == 10
        assert percentile(xs, 100) == 50

    def test_interpolates_between_values(self):
        # p=25 on [1,2,3,4,5] → k = 4 * 0.25 = 1.0 → exactly xs[1]=2
        assert percentile([1, 2, 3, 4, 5], 25) == 2.0
        # p=75 → k = 3.0 → exactly xs[3]=4
        assert percentile([1, 2, 3, 4, 5], 75) == 4.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty sample"):
            percentile([], 50)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"p must be in \[0, 100\]"):
            percentile([1, 2, 3], 101)
        with pytest.raises(ValueError, match=r"p must be in \[0, 100\]"):
            percentile([1, 2, 3], -1)


class TestBootstrapCI:
    def test_ci_covers_true_mean_for_normal_like_data(self):
        rng = random.Random(12345)
        samples = [rng.gauss(100.0, 10.0) for _ in range(200)]
        true_mean = statistics.mean(samples)
        lower, upper = bootstrap_ci(samples, resamples=500, rng=random.Random(1))
        assert lower <= true_mean <= upper
        # CI for the mean of 200 samples should be relatively tight
        assert (upper - lower) < 10.0

    def test_ci_on_constant_sample_has_zero_width(self):
        lower, upper = bootstrap_ci([5.0] * 50, resamples=200, rng=random.Random(1))
        assert lower == 5.0
        assert upper == 5.0

    def test_custom_statistic_median(self):
        rng = random.Random(7)
        samples = [rng.gauss(50.0, 5.0) for _ in range(100)]
        lower, upper = bootstrap_ci(
            samples,
            statistic=statistics.median,
            resamples=300,
            rng=random.Random(2),
        )
        assert lower <= statistics.median(samples) <= upper

    def test_empty_sample_raises(self):
        with pytest.raises(ValueError, match="empty sample"):
            bootstrap_ci([])

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match=r"confidence must be in \(0, 1\)"):
            bootstrap_ci([1, 2, 3], confidence=0.0)
        with pytest.raises(ValueError, match=r"confidence must be in \(0, 1\)"):
            bootstrap_ci([1, 2, 3], confidence=1.0)

    def test_invalid_resamples_raises(self):
        with pytest.raises(ValueError, match="resamples must be >= 1"):
            bootstrap_ci([1, 2, 3], resamples=0)


class TestMannWhitneyU:
    def test_clearly_different_distributions_significant(self):
        rng = random.Random(42)
        a = [rng.gauss(100.0, 5.0) for _ in range(30)]
        b = [rng.gauss(130.0, 5.0) for _ in range(30)]
        _, p = mann_whitney_u(a, b)
        assert p < 0.001

    def test_identical_distributions_not_significant(self):
        rng = random.Random(42)
        base = [rng.gauss(50.0, 5.0) for _ in range(200)]
        half = len(base) // 2
        _, p = mann_whitney_u(base[:half], base[half:])
        assert p > 0.05

    def test_one_sided_greater(self):
        rng = random.Random(11)
        a = [rng.gauss(110.0, 3.0) for _ in range(25)]
        b = [rng.gauss(100.0, 3.0) for _ in range(25)]
        _, p_two = mann_whitney_u(a, b, alternative="two-sided")
        _, p_greater = mann_whitney_u(a, b, alternative="greater")
        _, p_less = mann_whitney_u(a, b, alternative="less")
        assert p_greater < 0.01
        assert p_less > 0.99
        # one-sided p should be ~half of two-sided when extreme
        assert p_greater < p_two

    def test_all_ties_returns_nonsignificant(self):
        _, p = mann_whitney_u([5.0] * 10, [5.0] * 10)
        assert p == 1.0

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="both samples must be non-empty"):
            mann_whitney_u([], [1, 2, 3])
        with pytest.raises(ValueError, match="both samples must be non-empty"):
            mann_whitney_u([1, 2, 3], [])

    def test_rejects_bad_alternative(self):
        with pytest.raises(
            ValueError,
            match="alternative must be two-sided, less, or greater",
        ):
            mann_whitney_u([1, 2], [3, 4], alternative="bogus")


class TestBaselineIO:
    def test_roundtrip(self, tmp_path: Path):
        path = tmp_path / "nested" / "baseline.json"
        baseline = {
            "scenario": "demo",
            "p95_ttft_ms": 123.4,
            "n": 50,
            "samples": [1.0, 2.0, 3.0],
        }
        save_baseline(path, baseline)
        assert path.exists()
        loaded = load_baseline(path)
        assert loaded == baseline

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "a" / "b" / "c" / "base.json"
        save_baseline(path, {"x": 1})
        assert path.exists()

    def test_output_is_sorted_and_indented(self, tmp_path: Path):
        path = tmp_path / "base.json"
        save_baseline(path, {"z": 1, "a": 2})
        text = path.read_text()
        # Sort order: "a" should appear before "z"
        assert text.index('"a"') < text.index('"z"')
        # indent=2 → exactly two-space leading indent on keys
        assert '\n  "a"' in text
        assert '\n  "z"' in text


# ── Mutation-kill tests: pin exact values instead of loose inequalities ─────


class TestStatHarnessMutationKills:
    """Pins the exact formulas so off-by-one / wrong-constant mutations die.

    Reference values are computed by hand from the Mann-Whitney definition
    and the linear-interpolation percentile. No scipy dependency.
    """

    # ── percentile: interpolation branches ─────────────────────────────────

    def test_percentile_two_element_sample_interpolates(self):
        # k = (2-1) * 0.5 = 0.5; f=0, c=1, weight 0.5 → 10.0.
        # Kills "len(xs) == 1 → == 2" which would return xs[0]=5.0.
        assert percentile([5.0, 15.0], 50) == 10.0

    def test_percentile_non_integer_k_uses_interpolation_branch(self):
        # p=30 on [0, 100] → k=0.3 → result 30.0.
        # Kills "f == c → f != c": inverted branch returns xs[0]=0.0.
        assert percentile([0.0, 100.0], 30) == pytest.approx(30.0)

    def test_percentile_interpolation_weighted_correctly(self):
        # xs=[0, 100, 200]. p=25 → k = 2 * 0.25 = 0.5.
        # f=0, c=1. Result: xs[0] + (xs[1] - xs[0]) * (0.5 - 0) = 50.0.
        assert percentile([0.0, 100.0, 200.0], 25) == pytest.approx(50.0)

    # ── bootstrap_ci: default-arg / short-circuit traps ────────────────────

    def test_bootstrap_ci_with_no_rng_does_not_crash(self):
        # Kills "rng = rng or random.Random()" → "rng and random.Random()".
        # That mutation would leave rng=None and raise AttributeError on
        # rng.randrange.
        lower, upper = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], resamples=50)
        assert lower <= upper

    def test_bootstrap_ci_accepts_resamples_equal_to_one(self):
        # Kills "resamples < 1" → "resamples <= 1" or "< 2", both of which
        # would reject the legitimate edge case of a single resample.
        lower, upper = bootstrap_ci(
            [1.0, 2.0, 3.0], resamples=1, rng=random.Random(0)
        )
        assert lower == upper  # single resample produces a point

    def test_bootstrap_ci_default_confidence_is_valid(self):
        # Kills "confidence = 0.95" → "1.95", which would flip the validator
        # and raise ValueError on every default-arg call.
        lower, upper = bootstrap_ci(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            resamples=50,
            rng=random.Random(0),
        )
        assert lower <= upper

    # ── mann_whitney_u: pinned reference values ────────────────────────────

    def test_mw_reference_no_ties_two_sided(self):
        # a = 1..10, b = 11..20. No ties.
        # rank_sum_a = 1+2+...+10 = 55.  U_a = 55 - 10*11/2 = 0.
        # mu = 10*10/2 = 50. sigma^2 = (100/12) * 21 = 175.
        # sigma = sqrt(175). z = (|0-50| - 0.5)/sqrt(175) ≈ 3.74076.
        # p_two = 2 * (1 - Phi(z)) ≈ 1.8338e-4.
        a = list(range(1, 11))
        b = list(range(11, 21))
        u_a, p = mann_whitney_u(a, b, alternative="two-sided")
        assert u_a == 0.0
        expected_z = (50.0 - 0.5) / math.sqrt(175.0)
        expected_p = 2 * (1 - 0.5 * (1 + math.erf(expected_z / math.sqrt(2))))
        assert p == pytest.approx(expected_p, rel=1e-9)
        # sanity: pinned numeric value too
        assert p == pytest.approx(1.827e-4, rel=1e-2)

    def test_mw_reference_one_sided_less_and_greater(self):
        # Same samples. one-sided 'less' for a<b: z = (0 - 50 + 0.5)/sigma
        # ≈ -3.74037. p = Phi(z) ≈ 9.17e-5.
        # one-sided 'greater' for b>a (swap): same magnitude.
        a = list(range(1, 11))
        b = list(range(11, 21))
        sigma = math.sqrt(175.0)
        z_less = (0.0 - 50.0 + 0.5) / sigma
        expected_less = 0.5 * (1 + math.erf(z_less / math.sqrt(2)))
        _, p_less = mann_whitney_u(a, b, alternative="less")
        assert p_less == pytest.approx(expected_less, rel=1e-9)

        z_greater = (100.0 - 50.0 - 0.5) / sigma  # U_a when swapped = 100
        expected_greater = 1 - 0.5 * (1 + math.erf(z_greater / math.sqrt(2)))
        _, p_greater = mann_whitney_u(b, a, alternative="greater")
        assert p_greater == pytest.approx(expected_greater, rel=1e-9)

    def test_mw_reference_with_ties(self):
        # a = [1,1,2,3,5], b = [1,2,4,5,6]. Ties at values 1,2,5.
        # Worked example (see module docstring in tests): rank_sum_a = 23,
        # U_a = 8. tie_term = (3^3-3) + (2^3-2) + (2^3-2) = 24+6+6 = 36.
        # n=10, sigma^2 = (25/12)*(11 - 36/90) = (25/12)*10.6 = 22.0833...
        a = [1.0, 1.0, 2.0, 3.0, 5.0]
        b = [1.0, 2.0, 4.0, 5.0, 6.0]
        u_a, p = mann_whitney_u(a, b, alternative="two-sided")
        assert u_a == 8.0
        sigma2 = (25.0 / 12.0) * (11.0 - 36.0 / 90.0)
        sigma = math.sqrt(sigma2)
        z = (abs(8.0 - 12.5) - 0.5) / sigma
        expected_p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
        assert p == pytest.approx(expected_p, rel=1e-9)

    def test_mw_default_alternative_is_two_sided(self):
        # Kills mutations on the default-arg string ("two-sided" → "XXtwo-sidedXX"
        # / "TWO-SIDED"): the mutated default would fall through the
        # {"two-sided","less","greater"} guard and raise ValueError.
        a = [1.0, 2.0, 3.0, 4.0]
        b = [5.0, 6.0, 7.0, 8.0]
        _, p_default = mann_whitney_u(a, b)
        _, p_two = mann_whitney_u(a, b, alternative="two-sided")
        assert p_default == p_two
