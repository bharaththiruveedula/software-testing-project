"""
Test #6 — Differential / N-version testing.

Methodology: N-version / differential testing across independent
implementations.
Precedent: vLLM's reference comparison against HF Transformers — same
prompt, multiple backends, assert they agree on *structural* properties.

Assertion: For the same prompt run against several OpenAI-compatible
backends (e.g. vLLM, llama.cpp, Gemini-compat shim), all backends must:

    - Produce responses validating against the OpenAI envelope schema.
    - Report sane ``completion_tokens`` (0 < n ≤ max_tokens).
    - Agree on schema-validity rate within ±5 percentage points.

We explicitly do NOT assert latency parity (servers legitimately differ) or
content parity (different models produce different text). Quality is
out of scope per the proposal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import pytest
from aioresponses import aioresponses

from load_runner import run_load_test
from validator import validate_response

from tests.conftest import make_chat_completion


@dataclass
class BackendReport:
    name: str
    schema_valid_rate: float
    avg_completion_tokens: float


def assess_differential(
    reports: List[BackendReport],
    *,
    min_schema_valid_rate: float = 0.95,
    max_rate_delta: float = 0.05,
    max_tokens: int = 64,
) -> None:
    assert reports, "No backend reports provided"
    for r in reports:
        assert r.schema_valid_rate >= min_schema_valid_rate, (
            f"{r.name}: schema-valid rate {r.schema_valid_rate:.2%} < "
            f"{min_schema_valid_rate:.0%}"
        )
        assert 0 < r.avg_completion_tokens <= max_tokens, (
            f"{r.name}: avg completion_tokens {r.avg_completion_tokens} "
            f"outside (0, {max_tokens}]"
        )
    rates = [r.schema_valid_rate for r in reports]
    spread = max(rates) - min(rates)
    assert spread <= max_rate_delta, (
        f"Schema-valid rate disagreement across backends: "
        f"spread={spread:.2%} > {max_rate_delta:.0%} ({reports})"
    )


async def _probe_backend(
    url: str,
    payloads: List[dict],
    n: int,
    schema: dict,
    max_tokens: int,
    name: str,
) -> BackendReport:
    """Hit one backend ``n`` times and produce a BackendReport."""
    valid = 0
    completion_tokens_sum = 0
    counted = 0
    with aioresponses() as m:
        for p in payloads:
            m.post(url, payload=p)
        for _ in range(n):
            res = await run_load_test(
                url=url,
                num_users=1,
                prompt="Say hello.",
                schema=None,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            raw = res.raw_responses[0]
            if raw is None:
                continue
            if validate_response(raw, schema).is_valid:
                valid += 1
            try:
                body = json.loads(raw)
                ct = body.get("usage", {}).get("completion_tokens", 0)
                if ct > 0:
                    completion_tokens_sum += ct
                    counted += 1
            except json.JSONDecodeError:
                pass
    avg_ct = completion_tokens_sum / counted if counted else 0.0
    return BackendReport(
        name=name,
        schema_valid_rate=valid / n,
        avg_completion_tokens=avg_ct,
    )


# ── Unit mode ────────────────────────────────────────────────────────────────


class TestDifferentialUnit:

    @pytest.mark.asyncio
    async def test_three_healthy_backends_agree(self, chat_completion_schema: dict):
        n = 10
        backends: Dict[str, dict] = {
            "vllm":   make_chat_completion("Hello from vLLM!", completion_id="chatcmpl-v", completion_tokens=4),
            "llamacpp": make_chat_completion("Greetings!",      completion_id="chatcmpl-l", completion_tokens=2),
            "gemini": make_chat_completion("Hi there.",         completion_id="chatcmpl-g", completion_tokens=3),
        }
        reports = []
        for i, (name, body) in enumerate(backends.items()):
            url = f"http://localhost:0/backend-{i}/v1/chat/completions"
            reports.append(
                await _probe_backend(url, [body] * n, n, chat_completion_schema, 16, name)
            )
        assess_differential(reports, max_tokens=16)

    @pytest.mark.asyncio
    async def test_one_backend_violates_envelope_detected(
        self, chat_completion_schema: dict
    ):
        n = 10
        good = make_chat_completion("hi", completion_id="chatcmpl-g", completion_tokens=2)
        bad = make_chat_completion("hi", completion_id="chatcmpl-b", completion_tokens=2)
        bad["object"] = "not.chat.completion"  # envelope violation
        reports = [
            await _probe_backend(
                "http://localhost:0/good/v1/chat/completions",
                [good] * n, n, chat_completion_schema, 16, "good"
            ),
            await _probe_backend(
                "http://localhost:0/bad/v1/chat/completions",
                [bad] * n, n, chat_completion_schema, 16, "bad"
            ),
        ]
        with pytest.raises(AssertionError):
            assess_differential(reports, max_tokens=16)

    @pytest.mark.asyncio
    async def test_completion_tokens_sanity_catches_zero(
        self, chat_completion_schema: dict
    ):
        n = 10
        zero_tokens = make_chat_completion("", completion_id="chatcmpl-z", completion_tokens=0)
        # Force a zero — make_chat_completion would coerce to >= 1 otherwise
        zero_tokens["usage"]["completion_tokens"] = 0
        reports = [
            await _probe_backend(
                "http://localhost:0/zero/v1/chat/completions",
                [zero_tokens] * n, n, chat_completion_schema, 16, "zero"
            )
        ]
        with pytest.raises(AssertionError, match="completion_tokens"):
            assess_differential(reports, max_tokens=16)
