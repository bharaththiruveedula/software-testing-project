"""
Test #8 — Auth + error-path conformance.

Methodology: Negative / error-path testing.
Precedent: llama.cpp ``tools/server/tests/unit/test_security.py`` verifies
the server returns 401 on bad auth and 4xx (not 5xx, not hang) on malformed
input.

Assertions:
    1. Invalid bearer token → HTTP 401.
    2. Malformed JSON body → HTTP 4xx (not 5xx).
    3. Over-context or invalid request → HTTP 4xx.
    4. The load runner correctly records these as failures with
       recognizable error messages (not silent success, not 5xx-shaped).
"""

from __future__ import annotations

import re

import aiohttp
import pytest
from aioresponses import aioresponses

from load_runner import run_load_test


URL = "http://localhost:0/v1/chat/completions"


def extract_http_status(error: str) -> int | None:
    m = re.search(r"HTTP (\d+)", error or "")
    return int(m.group(1)) if m else None


class TestNegativeUnit:

    @pytest.mark.asyncio
    async def test_401_is_recorded_as_failure(self):
        with aioresponses() as m:
            m.post(URL, status=401, body='{"error":"invalid api key"}')
            result = await run_load_test(
                url=URL,
                num_users=1,
                prompt="hi",
                schema=None,
                api_key="bogus",
                temperature=0.0,
            )
        assert result.tracker.failed_requests == 1
        assert result.tracker.successful_requests == 0
        err = result.tracker.results[0].error or ""
        assert extract_http_status(err) == 401

    @pytest.mark.asyncio
    async def test_400_on_malformed_input_is_client_error_not_server_error(self):
        with aioresponses() as m:
            m.post(URL, status=400, body='{"error":"malformed"}')
            result = await run_load_test(
                url=URL,
                num_users=1,
                prompt="hi",
                schema=None,
                temperature=0.0,
            )
        err = result.tracker.results[0].error or ""
        status = extract_http_status(err)
        assert status is not None and 400 <= status < 500, (
            f"Expected 4xx for malformed input, got {status}"
        )

    @pytest.mark.asyncio
    async def test_5xx_is_flagged_as_failure(self):
        """A 500 must not be silently accepted as success."""
        with aioresponses() as m:
            m.post(URL, status=500, body="boom")
            result = await run_load_test(
                url=URL,
                num_users=1,
                prompt="hi",
                schema=None,
                temperature=0.0,
            )
        assert result.tracker.failed_requests == 1
        err = result.tracker.results[0].error or ""
        assert extract_http_status(err) == 500

    @pytest.mark.asyncio
    async def test_timeout_is_flagged_as_failure(self):
        """A server hang must not wedge the client — timeout must surface."""
        with aioresponses() as m:
            m.post(URL, exception=aiohttp.ServerTimeoutError())
            result = await run_load_test(
                url=URL,
                num_users=1,
                prompt="hi",
                schema=None,
                timeout_s=0.05,
                temperature=0.0,
            )
        assert result.tracker.failed_requests == 1


# ── Integration ──────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestNegativeIntegration:

    @pytest.mark.asyncio
    async def test_bogus_api_key_yields_401(
        self,
        chat_completions_url: str,
        model_name: str,
    ):
        result = await run_load_test(
            url=chat_completions_url,
            num_users=1,
            prompt="hi",
            schema=None,
            model=model_name,
            api_key="definitely-not-a-real-key",
            temperature=0.0,
        )
        err = result.tracker.results[0].error or ""
        status = extract_http_status(err)
        # Some servers (llama.cpp default) accept no auth — skip if they
        # returned 200; otherwise must be 401/403.
        if result.tracker.successful_requests == 1:
            pytest.skip("Server does not enforce bearer auth")
        assert status in (401, 403), f"Expected 401/403, got {status}"

    @pytest.mark.asyncio
    async def test_malformed_body_is_4xx(self, chat_completions_url: str):
        """Send raw garbage — server must return 4xx, not 5xx/hang."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                chat_completions_url,
                data="this is not json at all",
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                assert 400 <= resp.status < 500, (
                    f"Expected 4xx for malformed body, got {resp.status}"
                )
