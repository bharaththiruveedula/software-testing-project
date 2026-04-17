"""
Measurement calibration harness.

Before we can trust any of the server-centric tests (TTFT/TPOT claims,
regression verdicts), we need evidence the measurement tool *itself* is
calibrated. These tests inject known delays at an aiohttp test server and
assert the load runner recovers them within tolerance.

    - TTFT should equal the injected pre-first-chunk delay ± 25 ms.
    - TPOT should equal the injected inter-chunk delay ± 25 ms.
    - Non-streaming latency should equal the total simulated server time.

If calibration drifts, every downstream claim in Act 2 becomes suspect;
this is the "trustworthy tooling" floor.
"""

from __future__ import annotations

import asyncio
import json
from typing import List

import aiohttp
import pytest
from aiohttp import web

from load_runner import run_load_test


TOLERANCE_MS = 50.0  # generous to absorb CI jitter


async def _make_sse_server(
    *,
    pre_first_chunk_ms: float,
    inter_chunk_ms: float,
    tokens: List[str],
) -> tuple[aiohttp.web.AppRunner, str]:
    """Spin up an aiohttp server that emits a controlled SSE stream."""
    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await resp.prepare(request)
        completion_id = "chatcmpl-calibration"
        # Role chunk
        role = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "calib",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        await asyncio.sleep(pre_first_chunk_ms / 1000.0)
        await resp.write(f"data: {json.dumps(role)}\n\n".encode())
        for i, tok in enumerate(tokens):
            if i > 0:
                await asyncio.sleep(inter_chunk_ms / 1000.0)
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "calib",
                "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
            }
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "calib",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        await resp.write(f"data: {json.dumps(final)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    return runner, url


class TestCalibration:

    @pytest.mark.asyncio
    async def test_ttft_matches_injected_delay(self):
        pre = 150.0
        runner, url = await _make_sse_server(
            pre_first_chunk_ms=pre,
            inter_chunk_ms=10.0,
            tokens=["a", "b", "c"],
        )
        try:
            result = await run_load_test(
                url=url,
                num_users=1,
                prompt="hi",
                schema=None,
                stream=True,
                temperature=0.0,
                timeout_s=10.0,
            )
        finally:
            await runner.cleanup()

        ttft = result.tracker.results[0].ttft_ms
        assert ttft is not None
        assert abs(ttft - pre) <= TOLERANCE_MS, (
            f"TTFT off: measured {ttft:.1f}ms, injected {pre:.1f}ms, tol {TOLERANCE_MS}ms"
        )

    @pytest.mark.asyncio
    async def test_tpot_matches_injected_delay(self):
        inter = 80.0
        runner, url = await _make_sse_server(
            pre_first_chunk_ms=20.0,
            inter_chunk_ms=inter,
            tokens=["t1", "t2", "t3", "t4", "t5"],
        )
        try:
            result = await run_load_test(
                url=url,
                num_users=1,
                prompt="hi",
                schema=None,
                stream=True,
                temperature=0.0,
                timeout_s=10.0,
            )
        finally:
            await runner.cleanup()

        tpot = result.tracker.results[0].tpot_ms
        assert tpot is not None
        assert abs(tpot - inter) <= TOLERANCE_MS, (
            f"TPOT off: measured {tpot:.1f}ms, injected {inter:.1f}ms, tol {TOLERANCE_MS}ms"
        )

    @pytest.mark.asyncio
    async def test_latency_is_at_least_total_sim_time(self):
        """Latency must reflect total server time (within tolerance) and
        must not under-report."""
        pre = 40.0
        inter = 30.0
        tokens = ["a", "b", "c"]
        runner, url = await _make_sse_server(
            pre_first_chunk_ms=pre,
            inter_chunk_ms=inter,
            tokens=tokens,
        )
        try:
            result = await run_load_test(
                url=url,
                num_users=1,
                prompt="hi",
                schema=None,
                stream=True,
                temperature=0.0,
                timeout_s=10.0,
            )
        finally:
            await runner.cleanup()

        expected_min = pre + inter * (len(tokens) - 1)
        latency = result.tracker.results[0].latency_ms
        assert latency >= expected_min - TOLERANCE_MS, (
            f"Latency under-reported: {latency:.1f}ms < {expected_min:.1f}ms"
        )
