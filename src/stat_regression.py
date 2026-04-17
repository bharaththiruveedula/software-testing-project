"""
Statistical Regression Harness

Pure-stdlib utilities used by performance-regression tests (Test #5) and
determinism tests (Test #4) to make defensible pass/fail claims instead of
eyeballing single-run numbers.

Provides:
    - percentile(samples, p)
    - bootstrap_ci(samples, statistic, resamples, confidence)
    - mann_whitney_u(a, b, alternative)
    - save_baseline(path, baseline) / load_baseline(path)
"""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union


PathLike = Union[str, Path]


def percentile(samples: Sequence[float], p: float) -> float:
    """Return the pth percentile (0 ≤ p ≤ 100) using linear interpolation."""
    if not samples:
        raise ValueError("empty sample")
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    xs = sorted(samples)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[int(k)])
    return xs[int(f)] + (xs[int(c)] - xs[int(f)]) * (k - f)


def bootstrap_ci(
    samples: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = statistics.mean,
    resamples: int = 1000,
    confidence: float = 0.95,
    rng: Optional[random.Random] = None,
) -> Tuple[float, float]:
    """
    Percentile-method bootstrap confidence interval for ``statistic``.

    Resample ``samples`` with replacement ``resamples`` times, compute the
    statistic on each resample, then return the (alpha/2, 1-alpha/2)
    percentiles of the resulting distribution.
    """
    if not samples:
        raise ValueError("empty sample")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    rng = rng or random.Random()
    n = len(samples)
    values: List[float] = []
    for _ in range(resamples):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        values.append(statistic(resample))
    alpha = (1 - confidence) / 2
    lower = percentile(values, 100 * alpha)
    upper = percentile(values, 100 * (1 - alpha))
    return lower, upper


def _phi(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mann_whitney_u(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Mann-Whitney U test with continuity-corrected normal approximation.

    Returns ``(u_a, p_value)`` where ``u_a`` is the U statistic for sample
    ``a``. The normal approximation is appropriate for n1, n2 ≳ 8; for
    regression tests, prefer n ≥ 10 per group.
    """
    if not a or not b:
        raise ValueError("both samples must be non-empty")
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be two-sided, less, or greater")

    na, nb = len(a), len(b)
    combined = [(v, 0) for v in a] + [(v, 1) for v in b]
    combined.sort(key=lambda t: t[0])

    ranks: List[float] = [0.0] * len(combined)
    tie_term = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        t = j - i
        if t > 1:
            tie_term += (t ** 3 - t)
        i = j

    rank_sum_a = sum(r for r, (_, g) in zip(ranks, combined) if g == 0)
    u_a = rank_sum_a - na * (na + 1) / 2.0

    n = na + nb
    mu = na * nb / 2.0
    sigma2 = (na * nb / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    sigma = math.sqrt(sigma2) if sigma2 > 0 else 0.0

    if sigma == 0:
        return u_a, 1.0

    if alternative == "two-sided":
        z = (abs(u_a - mu) - 0.5) / sigma
        p = 2 * (1 - _phi(z))
    elif alternative == "less":
        z = (u_a - mu + 0.5) / sigma
        p = _phi(z)
    else:
        z = (u_a - mu - 0.5) / sigma
        p = 1 - _phi(z)
    return u_a, max(0.0, min(1.0, p))


def save_baseline(path: PathLike, baseline: dict) -> None:
    """Write a baseline JSON document to ``path`` (creating parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, sort_keys=True)


def load_baseline(path: PathLike) -> dict:
    """Load a baseline JSON document from ``path``."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
