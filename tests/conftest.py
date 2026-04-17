"""
Shared test fixtures and helpers.

- ``live_server_url`` fixture: skips integration tests when no reachable
  OpenAI-compatible backend is up. Controlled by env var
  ``LLM_TEST_SERVER_URL`` (default http://localhost:8080).
- Helpers for building mock OpenAI response payloads and SSE bodies used by
  the server-centric test suite (test_contract, test_streaming_protocol,
  test_determinism, test_param_honoring, test_concurrency_isolation,
  test_negative, test_differential, test_stat_regression).
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urlparse

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
OPENAI_SCHEMA_DIR = REPO_ROOT / "configs" / "openai_schemas"


# ── Server reachability ──────────────────────────────────────────────────────


def _probe(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def live_server_url() -> str:
    """
    Return the base URL of a reachable OpenAI-compatible server, or skip.

    Honored env var: ``LLM_TEST_SERVER_URL``. Default: http://localhost:8080
    (llama.cpp's default). Tests that require a live server should depend on
    this fixture and use ``chat_completions_url`` / ``completions_url``.
    """
    url = os.environ.get("LLM_TEST_SERVER_URL", "http://localhost:8080")
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not _probe(host, port):
        pytest.skip(f"No LLM inference server reachable at {url}")
    return url.rstrip("/")


@pytest.fixture(scope="session")
def chat_completions_url(live_server_url: str) -> str:
    return f"{live_server_url}/v1/chat/completions"


@pytest.fixture(scope="session")
def completions_url(live_server_url: str) -> str:
    return f"{live_server_url}/v1/completions"


@pytest.fixture(scope="session")
def model_name() -> str:
    """Model name to send in requests; overridable via ``LLM_TEST_MODEL``."""
    return os.environ.get("LLM_TEST_MODEL", "default")


# ── OpenAI schema loaders ────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def chat_completion_schema() -> dict:
    return json.loads((OPENAI_SCHEMA_DIR / "chat_completion.json").read_text())


@pytest.fixture(scope="session")
def chat_completion_chunk_schema() -> dict:
    return json.loads((OPENAI_SCHEMA_DIR / "chat_completion_chunk.json").read_text())


# ── Mock payload helpers ─────────────────────────────────────────────────────


def make_chat_completion(
    content: str,
    *,
    completion_id: str = "chatcmpl-mock-1",
    model: str = "mock-model",
    finish_reason: str = "stop",
    prompt_tokens: int = 8,
    completion_tokens: Optional[int] = None,
) -> dict:
    """Build a syntactically-valid non-streaming chat completion response."""
    if completion_tokens is None:
        completion_tokens = max(1, len(content.split()))
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_sse_body(
    tokens: Iterable[str],
    *,
    completion_id: str = "chatcmpl-mock-1",
    model: str = "mock-model",
    finish_reason: str = "stop",
    include_role: bool = True,
    include_usage: bool = True,
    prompt_tokens: int = 8,
) -> str:
    """
    Build an SSE body as an OpenAI-compatible chat stream would emit it.

    Sequence:
        1. (optional) role-only delta chunk
        2. one content-delta chunk per token
        3. final chunk with empty delta + finish_reason
        4. (optional) usage chunk
        5. ``data: [DONE]``
    """
    tokens = list(tokens)
    chunks: List[dict] = []
    if include_role:
        chunks.append({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        })
    for tok in tokens:
        chunks.append({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
        })
    chunks.append({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    })
    if include_usage:
        chunks.append({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(tokens),
                "total_tokens": prompt_tokens + len(tokens),
            },
        })
    body_lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    body_lines.append("data: [DONE]\n\n")
    return "".join(body_lines)
