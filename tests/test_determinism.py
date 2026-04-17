"""
Test #4 — Determinism at temperature=0.

Methodology: Metamorphic / idempotency testing.
Precedent: vLLM ``tests/v1/determinism/`` — the entire directory asserts that
identical inputs at temperature 0 produce byte-identical outputs.

Assertion: For N repeats of the same prompt at temp=0, determinism rate
(fraction of repeats that match the first response verbatim) has a
bootstrap-CI lower bound ≥ 0.9.

Unit mode runs against ``aioresponses`` so the methodology is demo-able
without a live server; integration mode hits the live server.
"""

from __future__ import annotations

import asyncio
import random
from typing import List, Sequence

import pytest
from aioresponses import aioresponses

from load_runner import run_load_test
from stat_regression import bootstrap_ci

from tests.conftest import make_chat_completion


DETERMINISM_CI_LOWER_BOUND = 0.9
N_REPEATS = 20
BOOTSTRAP_RESAMPLES = 500


def determinism_rate(responses: Sequence[str]) -> float:
    """Fraction of responses that match the first one verbatim."""
    if not responses:
        return 0.0
    first = responses[0]
    return sum(1 for r in responses if r == first) / len(responses)


async def _collect_responses(url: str, n: int, **kwargs) -> List[str]:
    """Run ``n`` sequential requests and return their extracted texts."""
    results: List[str] = []
    for _ in range(n):
        res = await run_load_test(
            url=url,
            num_users=1,
            prompt="What is 2 + 2?",
            schema=None,
            temperature=0.0,
            **kwargs,
        )
        raw = res.raw_responses[0]
        assert raw is not None, "request failed"
        results.append(raw)
    return results


# ── Unit mode ────────────────────────────────────────────────────────────────


class TestDeterminismUnit:
    """Exercises the assertion using aioresponses — no live server needed."""

    @pytest.mark.asyncio
    async def test_perfectly_deterministic_server_passes(self):
        content = "The answer is 4."
        mock = make_chat_completion(content)
        url = "http://localhost:0/v1/chat/completions"
        with aioresponses() as m:
            for _ in range(N_REPEATS):
                m.post(url, payload=mock)
            responses = await _collect_responses(url, N_REPEATS)

        rate = determinism_rate(responses)
        lower, _ = bootstrap_ci(
            [1.0 if r == responses[0] else 0.0 for r in responses],
            resamples=BOOTSTRAP_RESAMPLES,
            rng=random.Random(1),
        )
        assert rate == 1.0
        assert lower >= DETERMINISM_CI_LOWER_BOUND

    @pytest.mark.asyncio
    async def test_nondeterministic_server_fails_ci_bound(self):
        # Alternate two distinct contents → rate = 0.5 (first is "A", half will match)
        url = "http://localhost:0/v1/chat/completions"
        with aioresponses() as m:
            for i in range(N_REPEATS):
                content = "A" if i % 2 == 0 else "B"
                m.post(url, payload=make_chat_completion(content))
            responses = await _collect_responses(url, N_REPEATS)

        matches = [1.0 if r == responses[0] else 0.0 for r in responses]
        lower, upper = bootstrap_ci(
            matches,
            resamples=BOOTSTRAP_RESAMPLES,
            rng=random.Random(1),
        )
        # Rate is 0.5; the CI must clearly be below 0.9, which is the
        # pass/fail threshold — this is the methodology's failure mode.
        assert lower < DETERMINISM_CI_LOWER_BOUND


# ── Integration mode ─────────────────────────────────────────────────────────


@pytest.mark.integration
class TestDeterminismIntegration:
    """Exercises the assertion against a live OpenAI-compatible backend."""

    @pytest.mark.asyncio
    async def test_determinism_rate_ci_lower_bound(
        self,
        chat_completions_url: str,
        model_name: str,
    ):
        responses = await _collect_responses(
            chat_completions_url,
            N_REPEATS,
            model=model_name,
        )
        matches = [1.0 if r == responses[0] else 0.0 for r in responses]
        lower, _ = bootstrap_ci(
            matches,
            resamples=BOOTSTRAP_RESAMPLES,
            rng=random.Random(1),
        )
        assert lower >= DETERMINISM_CI_LOWER_BOUND, (
            f"Determinism CI lower bound {lower:.3f} < "
            f"{DETERMINISM_CI_LOWER_BOUND}; server is not deterministic at "
            f"temperature=0."
        )
