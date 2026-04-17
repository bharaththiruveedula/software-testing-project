"""
Test #7 — Concurrency isolation.

Methodology: Isolation invariants under concurrency.
Precedent: llama.cpp parallel tests implicitly assume per-request isolation;
this test makes that assumption an explicit invariant. The failure mode is
context/session bleed between simultaneous requests in a batched server.

Assertion: When N concurrent requests each carry a unique marker (e.g.
``MARKER-XXXXXXXX``), each response contains its own marker and none of the
N-1 foreign markers. Any cross-talk indicates a session-isolation bug.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from typing import Dict, List
from urllib.parse import urlparse

import aiohttp
import pytest
from aioresponses import aioresponses, CallbackResult

from load_runner import run_load_test

from tests.conftest import make_chat_completion


MARKER_PREFIX = "MARKER-"


def make_marker() -> str:
    return f"{MARKER_PREFIX}{secrets.token_hex(4).upper()}"


def build_marker_prompt(marker: str) -> str:
    return (
        f"Please repeat the following token verbatim, with no additional text: "
        f"{marker}"
    )


def assert_isolation(prompts: List[str], responses: List[str]) -> None:
    """Each response must contain its own marker and no foreign marker."""
    assert len(prompts) == len(responses)
    markers = [p.split()[-1] for p in prompts]  # marker is the last word
    foreign_marker_hits: Dict[int, List[str]] = {}
    own_marker_misses: List[int] = []

    for i, (own, resp) in enumerate(zip(markers, responses)):
        if own not in resp:
            own_marker_misses.append(i)
        foreign = [m for j, m in enumerate(markers) if j != i and m in resp]
        if foreign:
            foreign_marker_hits[i] = foreign

    assert not own_marker_misses, (
        f"Responses missing their own marker: indices {own_marker_misses}"
    )
    assert not foreign_marker_hits, (
        f"Cross-contamination: {foreign_marker_hits}"
    )


# ── Unit mode: honest mock server ────────────────────────────────────────────


def _honest_echo_callback(url, **kwargs):
    """Return a response containing the prompt's marker — simulates an
    isolated server."""
    body = kwargs.get("json") or {}
    prompt = body["messages"][0]["content"]
    marker = prompt.split()[-1]
    return CallbackResult(
        status=200,
        payload=make_chat_completion(f"Sure: {marker}", completion_id="chatcmpl-test"),
    )


def _bleeding_callback_factory(leak_marker: str):
    """Return a callback that always echoes a fixed foreign marker — simulates
    a server with broken isolation."""
    def cb(url, **kwargs):
        body = kwargs.get("json") or {}
        prompt = body["messages"][0]["content"]
        own = prompt.split()[-1]
        return CallbackResult(
            status=200,
            payload=make_chat_completion(
                f"Sure: {own} (and also {leak_marker})",
                completion_id="chatcmpl-test",
            ),
        )
    return cb


async def _run_with_callback(url: str, markers: List[str], callback) -> List[str]:
    """Launch len(markers) concurrent requests against aioresponses returning
    ``callback``'s payload; return generated contents."""
    prompts = [build_marker_prompt(m) for m in markers]
    with aioresponses() as m:
        # Register a single matcher that handles every call with the callback.
        for _ in markers:
            m.post(url, callback=callback)
        async def one(p: str) -> str:
            res = await run_load_test(
                url=url,
                num_users=1,
                prompt=p,
                schema=None,
                temperature=0.0,
            )
            return res.raw_responses[0]
        raws = await asyncio.gather(*(one(p) for p in prompts))
    return [json.loads(r)["choices"][0]["message"]["content"] for r in raws]


class TestIsolationUnit:

    @pytest.mark.asyncio
    async def test_honest_server_passes(self):
        markers = [make_marker() for _ in range(5)]
        url = "http://localhost:0/v1/chat/completions"
        contents = await _run_with_callback(url, markers, _honest_echo_callback)
        prompts = [build_marker_prompt(m) for m in markers]
        assert_isolation(prompts, contents)

    @pytest.mark.asyncio
    async def test_bleeding_server_detected(self):
        markers = [make_marker() for _ in range(3)]
        leak = markers[0]
        url = "http://localhost:0/v1/chat/completions"
        contents = await _run_with_callback(
            url, markers, _bleeding_callback_factory(leak)
        )
        prompts = [build_marker_prompt(m) for m in markers]
        with pytest.raises(AssertionError, match="Cross-contamination"):
            assert_isolation(prompts, contents)


# ── Integration ──────────────────────────────────────────────────────────────


async def _one_live_request(url: str, model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": False,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            body = await resp.json()
    return body["choices"][0]["message"]["content"]


@pytest.mark.integration
class TestIsolationIntegration:

    @pytest.mark.asyncio
    async def test_concurrent_markers_are_isolated(
        self,
        chat_completions_url: str,
        model_name: str,
    ):
        n = 8
        markers = [make_marker() for _ in range(n)]
        prompts = [build_marker_prompt(m) for m in markers]
        responses = await asyncio.gather(
            *(_one_live_request(chat_completions_url, model_name, p) for p in prompts)
        )
        assert_isolation(prompts, responses)
