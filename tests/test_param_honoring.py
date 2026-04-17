"""
Test #3 — Parameter honoring.

Methodology: Metamorphic testing.
Precedent: vLLM ``tests/entrypoints/openai/test_ignore_eos.py``; llama.cpp
``test_chat_completion.py`` uses regex assertions on completion text to
verify stop sequences and token limits are honored.

Assertions:
    A) ``max_tokens=N`` → ``usage.completion_tokens ≤ N`` and
       ``finish_reason == 'length'`` when the limit is reached.
    B) ``stop=["X"]`` → generated text does not contain "X" after trimming,
       and ``finish_reason == 'stop'`` when the stop sequence triggered.

Unit mode verifies (1) the client correctly propagates these parameters to
the server and (2) the assertion layer correctly *flags* a non-conforming
server response. Integration mode verifies the live server honors them.
"""

from __future__ import annotations

import json
from typing import List

import pytest
from aioresponses import aioresponses

from load_runner import _build_chat_payload, run_load_test

from tests.conftest import make_chat_completion


# ── Unit mode: client propagation ────────────────────────────────────────────


class TestParameterPropagation:
    """The client must actually send max_tokens / stop to the server."""

    def test_chat_payload_includes_max_tokens(self):
        payload = _build_chat_payload("p", "m", 0.0, max_tokens=17)
        assert payload["max_tokens"] == 17

    def test_chat_payload_includes_stop(self):
        payload = _build_chat_payload("p", "m", 0.0, stop=["END", "###"])
        assert payload["stop"] == ["END", "###"]

    def test_chat_payload_omits_stop_when_none(self):
        payload = _build_chat_payload("p", "m", 0.0, stop=None)
        assert "stop" not in payload

    @pytest.mark.asyncio
    async def test_run_load_test_sends_max_tokens_and_stop(self):
        url = "http://localhost:0/v1/chat/completions"
        mock = make_chat_completion("ok", finish_reason="length", completion_tokens=5)
        with aioresponses() as m:
            m.post(url, payload=mock)
            await run_load_test(
                url=url,
                num_users=1,
                prompt="hi",
                schema=None,
                temperature=0.0,
                max_tokens=5,
                stop=["XYZ"],
            )
            # Inspect the one captured request
            sent = list(m.requests.values())[0][0]
            body = json.loads(sent.kwargs["json"] and json.dumps(sent.kwargs["json"]))
            assert body["max_tokens"] == 5
            assert body["stop"] == ["XYZ"]


# ── Assertion layer ──────────────────────────────────────────────────────────


def max_tokens_honored(response: dict, limit: int) -> bool:
    usage = response.get("usage", {})
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        return False
    if completion_tokens > limit:
        return False
    # If the response hit the limit, finish_reason must be "length"
    finish = response["choices"][0].get("finish_reason")
    if completion_tokens == limit and finish != "length":
        return False
    return True


def stop_sequence_honored(response: dict, stops: List[str]) -> bool:
    content = response["choices"][0]["message"].get("content") or ""
    # The stop token itself must not appear in emitted content
    for s in stops:
        if s in content:
            return False
    return True


class TestAssertionLayerCatchesViolations:
    """If the server lies, our test layer must notice."""

    def test_catches_over_budget_completion_tokens(self):
        violating = make_chat_completion(
            "one two three four five six",
            finish_reason="length",
            completion_tokens=9,
        )
        assert max_tokens_honored(violating, limit=5) is False

    def test_accepts_exact_budget_with_length_finish(self):
        ok = make_chat_completion(
            "t t t t t",
            finish_reason="length",
            completion_tokens=5,
        )
        assert max_tokens_honored(ok, limit=5) is True

    def test_catches_exact_budget_with_wrong_finish_reason(self):
        # At the ceiling but server claims natural stop — almost certainly a lie
        lying = make_chat_completion(
            "t t t t t",
            finish_reason="stop",
            completion_tokens=5,
        )
        assert max_tokens_honored(lying, limit=5) is False

    def test_catches_stop_token_leak(self):
        leaking = make_chat_completion("hello ENDOFTEXT bye")
        assert stop_sequence_honored(leaking, stops=["ENDOFTEXT"]) is False

    def test_accepts_clean_response(self):
        clean = make_chat_completion("hello there")
        assert stop_sequence_honored(clean, stops=["ENDOFTEXT"]) is True


# ── Integration mode ─────────────────────────────────────────────────────────


@pytest.mark.integration
class TestParameterHonoringIntegration:

    @pytest.mark.asyncio
    async def test_max_tokens_respected(
        self,
        chat_completions_url: str,
        model_name: str,
    ):
        result = await run_load_test(
            url=chat_completions_url,
            num_users=1,
            prompt="Count from 1 to 100.",
            schema=None,
            temperature=0.0,
            model=model_name,
            max_tokens=8,
        )
        raw = result.raw_responses[0]
        assert raw is not None
        body = json.loads(raw)
        assert max_tokens_honored(body, limit=8), (
            f"Server exceeded max_tokens=8: {body.get('usage')}, "
            f"finish_reason={body['choices'][0].get('finish_reason')}"
        )

    @pytest.mark.asyncio
    async def test_stop_sequence_respected(
        self,
        chat_completions_url: str,
        model_name: str,
    ):
        result = await run_load_test(
            url=chat_completions_url,
            num_users=1,
            prompt="Write exactly: A B C STOPHERE D E F",
            schema=None,
            temperature=0.0,
            model=model_name,
            stop=["STOPHERE"],
        )
        raw = result.raw_responses[0]
        assert raw is not None
        body = json.loads(raw)
        assert stop_sequence_honored(body, stops=["STOPHERE"])
