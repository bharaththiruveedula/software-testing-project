"""
Test #2 — Streaming protocol compliance.

Methodology: Protocol compliance.
Precedent: llama.cpp ``test_chat_completion.py`` streaming variant asserts
role-only first chunk, stable id across chunks, and terminal finish_reason.

Assertions:
    1. First delta carries ``role`` (or is empty), not content-before-role.
    2. ``id`` is stable across every chunk in a response.
    3. Exactly one chunk carries a terminal ``finish_reason``; it is the
       last content-bearing chunk (or a trailing usage chunk).
    4. ``data: [DONE]`` sentinel is emitted.
    5. When ``stream_options.include_usage`` is set, a ``usage`` block is
       emitted in one of the chunks.
    6. Each chunk validates against the chunk JSON schema.
"""

from __future__ import annotations

import json
from typing import Iterable, List, Optional, Tuple

import aiohttp
import pytest
from aioresponses import aioresponses

from load_runner import SSEParser, parse_sse_stream
from validator import validate_response

from tests.conftest import make_sse_body


# ── Assertions layer ─────────────────────────────────────────────────────────


def split_events(sse_bytes_iter: Iterable[bytes]) -> Tuple[List[dict], bool]:
    """Parse the SSE stream; return (chunk_payloads, saw_done)."""
    chunks: List[dict] = []
    saw_done = False
    for ev in parse_sse_stream(sse_bytes_iter):
        if ev.data == "[DONE]":
            saw_done = True
            continue
        try:
            chunks.append(json.loads(ev.data))
        except json.JSONDecodeError:
            pytest.fail(f"chunk is not valid JSON: {ev.data!r}")
    return chunks, saw_done


def assert_protocol_compliance(
    chunks: List[dict],
    saw_done: bool,
    chunk_schema: dict,
    *,
    expect_usage: bool,
) -> None:
    assert saw_done, "Stream ended without `data: [DONE]` sentinel"
    assert chunks, "Stream yielded no data chunks"

    # Every chunk validates
    for c in chunks:
        r = validate_response(json.dumps(c), chunk_schema)
        assert r.is_valid, f"Chunk schema violation: {r.error_message} ({c})"

    # Stable id
    ids = {c["id"] for c in chunks}
    assert len(ids) == 1, f"Non-stable id across chunks: {ids}"

    # The first *choice-bearing* chunk's delta must not carry content without role
    choice_chunks = [c for c in chunks if c.get("choices")]
    assert choice_chunks, "No chunk with choices[] present"
    first_delta = choice_chunks[0]["choices"][0].get("delta", {})
    assert "content" not in first_delta or first_delta.get("role") == "assistant", (
        f"First chunk carried content before role: {first_delta}"
    )

    # Exactly one terminal finish_reason among choice chunks
    finish_reasons = [
        c["choices"][0].get("finish_reason")
        for c in choice_chunks
        if c["choices"]
    ]
    terminals = [f for f in finish_reasons if f is not None]
    assert len(terminals) == 1, (
        f"Expected exactly 1 terminal finish_reason, got {len(terminals)}: {terminals}"
    )
    # Terminal must be the last finish_reason in order
    assert finish_reasons[-1] is not None, (
        "Terminal finish_reason did not appear on the last choice-bearing chunk"
    )

    if expect_usage:
        has_usage = any(c.get("usage") for c in chunks)
        assert has_usage, "include_usage requested but no chunk carried usage block"


# ── Unit mode ────────────────────────────────────────────────────────────────


class TestStreamingUnit:
    """Protocol assertions against synthetic SSE bodies from conftest."""

    def test_well_formed_stream_passes(self, chat_completion_chunk_schema: dict):
        body = make_sse_body(["Hello", " ", "world"], include_usage=True)
        chunks, saw_done = split_events([body.encode()])
        assert_protocol_compliance(
            chunks, saw_done, chat_completion_chunk_schema, expect_usage=True
        )

    def test_missing_done_fails(self, chat_completion_chunk_schema: dict):
        body = make_sse_body(["x"], include_usage=True)
        body = body.replace("data: [DONE]\n\n", "")
        chunks, saw_done = split_events([body.encode()])
        with pytest.raises(AssertionError, match=r"\[DONE\]"):
            assert_protocol_compliance(
                chunks, saw_done, chat_completion_chunk_schema, expect_usage=True
            )

    def test_unstable_id_fails(self, chat_completion_chunk_schema: dict):
        body = make_sse_body(["x", "y"], completion_id="chatcmpl-a", include_usage=False)
        # Flip the id mid-stream by doing a naive string replace on the second token chunk
        body = body.replace("chatcmpl-a", "chatcmpl-b", 1).replace("chatcmpl-a", "chatcmpl-a")
        chunks, saw_done = split_events([body.encode()])
        with pytest.raises(AssertionError, match="Non-stable id"):
            assert_protocol_compliance(
                chunks, saw_done, chat_completion_chunk_schema, expect_usage=False
            )

    def test_multiple_finish_reasons_fails(self, chat_completion_chunk_schema: dict):
        body = make_sse_body(["x", "y"], include_usage=False)
        # Inject a second finish_reason by adding a bogus chunk before [DONE]
        rogue = {
            "id": "chatcmpl-mock-1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        body = body.replace(
            "data: [DONE]\n\n",
            f"data: {json.dumps(rogue)}\n\ndata: [DONE]\n\n",
        )
        chunks, saw_done = split_events([body.encode()])
        with pytest.raises(AssertionError, match="exactly 1 terminal"):
            assert_protocol_compliance(
                chunks, saw_done, chat_completion_chunk_schema, expect_usage=False
            )

    def test_event_split_across_chunk_boundaries(
        self, chat_completion_chunk_schema: dict
    ):
        """The buffered parser must reassemble events fragmented across reads."""
        body = make_sse_body(["hello"], include_usage=False).encode()
        # Slice mid-event
        mid = len(body) // 3
        parts = [body[:mid], body[mid:mid * 2], body[mid * 2:]]
        chunks, saw_done = split_events(parts)
        assert_protocol_compliance(
            chunks, saw_done, chat_completion_chunk_schema, expect_usage=False
        )


# ── Against aioresponses ────────────────────────────────────────────────────


class TestStreamingViaAioresponses:
    """The client receives a compliant stream end-to-end over aiohttp+SSE."""

    @pytest.mark.asyncio
    async def test_aiohttp_consumes_well_formed_stream(
        self, chat_completion_chunk_schema: dict
    ):
        url = "http://localhost:0/v1/chat/completions"
        body = make_sse_body(["foo", " bar"], include_usage=True)
        with aioresponses() as m:
            m.post(url, status=200, body=body, headers={"Content-Type": "text/event-stream"})
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={}) as resp:
                    parser = SSEParser()
                    chunks: List[dict] = []
                    saw_done = False
                    async for chunk_bytes in resp.content:
                        for ev in parser.feed(chunk_bytes):
                            if ev.data == "[DONE]":
                                saw_done = True
                            else:
                                chunks.append(json.loads(ev.data))
                    for ev in parser.flush():
                        if ev.data == "[DONE]":
                            saw_done = True
                        else:
                            chunks.append(json.loads(ev.data))

        assert_protocol_compliance(
            chunks, saw_done, chat_completion_chunk_schema, expect_usage=True
        )


# ── Integration mode ─────────────────────────────────────────────────────────


async def _stream_protocol(url: str, model: str, include_usage: bool) -> Tuple[List[dict], bool]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": True,
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    parser = SSEParser()
    chunks: List[dict] = []
    saw_done = False
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            async for chunk_bytes in resp.content:
                for ev in parser.feed(chunk_bytes):
                    if ev.data == "[DONE]":
                        saw_done = True
                    else:
                        chunks.append(json.loads(ev.data))
            for ev in parser.flush():
                if ev.data == "[DONE]":
                    saw_done = True
                else:
                    chunks.append(json.loads(ev.data))
    return chunks, saw_done


@pytest.mark.integration
class TestStreamingIntegration:

    @pytest.mark.asyncio
    async def test_live_stream_is_protocol_compliant(
        self,
        chat_completions_url: str,
        chat_completion_chunk_schema: dict,
        model_name: str,
    ):
        chunks, saw_done = await _stream_protocol(
            chat_completions_url, model_name, include_usage=True
        )
        assert_protocol_compliance(
            chunks, saw_done, chat_completion_chunk_schema, expect_usage=True
        )
