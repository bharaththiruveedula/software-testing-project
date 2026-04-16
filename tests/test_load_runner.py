"""
Unit tests for the Async Load Runner Module (load_runner.py).

Uses aioresponses to mock HTTP responses without hitting a real server.
"""

import json

import pytest
import pytest_asyncio
from aioresponses import aioresponses

from load_runner import (
    LoadTestResult,
    _build_chat_payload,
    _build_completions_payload,
    _extract_generated_text,
    _extract_token_usage,
    _strip_markdown_fences,
    run_load_test,
)


# ── Test URL ──────────────────────────────────────────────────────────────────

TEST_URL = "http://localhost:8000/v1/completions"
TEST_CHAT_URL = "http://localhost:8000/v1/chat/completions"

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "confidence"],
}


# ── Helper: build a mock OpenAI response ──────────────────────────────────────

def _mock_completion_response(text: str) -> dict:
    return {
        "id": "cmpl-test",
        "object": "text_completion",
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
    }


def _mock_chat_response(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


# ── Tests: _extract_generated_text ────────────────────────────────────────────

class TestExtractGeneratedText:
    """Test extraction of generated text from OpenAI-style responses."""

    def test_extract_from_completions(self):
        response = json.dumps(_mock_completion_response("Hello world"))
        assert _extract_generated_text(response) == "Hello world"

    def test_extract_from_chat_completions(self):
        response = json.dumps(_mock_chat_response("Hello world"))
        assert _extract_generated_text(response) == "Hello world"

    def test_fallback_on_plain_text(self):
        assert _extract_generated_text("plain text response") == "plain text response"

    def test_fallback_on_malformed_json(self):
        response = json.dumps({"unexpected": "format"})
        assert _extract_generated_text(response) == response

    def test_fallback_on_empty_choices(self):
        response = json.dumps({"choices": []})
        assert _extract_generated_text(response) == response

    def test_strips_markdown_code_fences_from_chat(self):
        """Gemini wraps JSON in markdown fences — verify they are stripped."""
        fenced = '```json\n{"answer": "hello", "confidence": 0.9}\n```'
        response = json.dumps(_mock_chat_response(fenced))
        extracted = _extract_generated_text(response)
        assert extracted == '{"answer": "hello", "confidence": 0.9}'

    def test_strips_markdown_code_fences_from_completions(self):
        fenced = '```json\n{"answer": "hi"}\n```'
        response = json.dumps(_mock_completion_response(fenced))
        extracted = _extract_generated_text(response)
        assert extracted == '{"answer": "hi"}'


# ── Tests: _strip_markdown_fences ─────────────────────────────────────────────

class TestStripMarkdownFences:
    """Test markdown code fence stripping."""

    def test_json_fenced(self):
        text = '```json\n{"key": "value"}\n```'
        assert _strip_markdown_fences(text) == '{"key": "value"}'

    def test_plain_fenced(self):
        text = '```\n{"key": "value"}\n```'
        assert _strip_markdown_fences(text) == '{"key": "value"}'

    def test_no_fences(self):
        text = '{"key": "value"}'
        assert _strip_markdown_fences(text) == '{"key": "value"}'

    def test_plain_text_no_fences(self):
        text = "plain text"
        assert _strip_markdown_fences(text) == "plain text"

    def test_whitespace_handling(self):
        text = '  ```json\n  {"key": "value"}  \n```  '
        result = _strip_markdown_fences(text)
        assert '"key"' in result


# ── Tests: _extract_token_usage ───────────────────────────────────────────────

class TestExtractTokenUsage:
    """Test extraction of token usage from API responses."""

    def test_extract_valid_usage(self):
        response = json.dumps({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 25,
                "total_tokens": 35
            }
        })
        p, c = _extract_token_usage(response)
        assert p == 10
        assert c == 25

    def test_extract_missing_usage(self):
        response = json.dumps({"choices": []})
        p, c = _extract_token_usage(response)
        assert p == 0
        assert c == 0

    def test_extract_malformed_json(self):
        response = "invalid json"
        p, c = _extract_token_usage(response)
        assert p == 0
        assert c == 0


# ── Tests: Payload Builders ───────────────────────────────────────────────────

class TestPayloadBuilders:
    """Test the request payload builder functions."""

    def test_completions_payload(self):
        payload = _build_completions_payload("test prompt", "gpt-4", 0.5)
        assert payload["model"] == "gpt-4"
        assert payload["prompt"] == "test prompt"
        assert payload["temperature"] == 0.5
        assert "max_tokens" in payload

    def test_chat_payload(self):
        payload = _build_chat_payload("test prompt", "gpt-4", 0.5)
        assert payload["model"] == "gpt-4"
        assert payload["messages"][0]["content"] == "test prompt"
        assert payload["temperature"] == 0.5
        assert "max_tokens" in payload


# ── Tests: run_load_test ──────────────────────────────────────────────────────

class TestRunLoadTest:
    """Integration-style tests for the load runner using mocked HTTP."""

    @pytest.mark.asyncio
    async def test_successful_requests(self):
        """All requests succeed and return valid schema-compliant content."""
        valid_content = json.dumps({"answer": "hello", "confidence": 0.9})
        mock_response = _mock_completion_response(valid_content)

        with aioresponses() as m:
            # Register the mock for multiple calls
            for _ in range(5):
                m.post(TEST_URL, payload=mock_response)

            result = await run_load_test(
                url=TEST_URL,
                num_users=5,
                prompt="test",
                schema=SIMPLE_SCHEMA,
                timeout_s=10.0,
            )

        assert isinstance(result, LoadTestResult)
        assert result.tracker.total_requests == 5
        assert result.tracker.successful_requests == 5
        assert result.tracker.failed_requests == 0
        assert all(v.is_valid for v in result.validation_results)

    @pytest.mark.asyncio
    async def test_schema_validation_failures(self):
        """Requests succeed HTTP-wise but return schema-invalid content."""
        invalid_content = json.dumps({"wrong_field": "value"})
        mock_response = _mock_completion_response(invalid_content)

        with aioresponses() as m:
            for _ in range(3):
                m.post(TEST_URL, payload=mock_response)

            result = await run_load_test(
                url=TEST_URL,
                num_users=3,
                prompt="test",
                schema=SIMPLE_SCHEMA,
            )

        assert result.tracker.successful_requests == 3
        # All validation should fail
        assert all(not v.is_valid for v in result.validation_results)

    @pytest.mark.asyncio
    async def test_http_error_responses(self):
        """Server returns HTTP error status codes."""
        with aioresponses() as m:
            for _ in range(3):
                m.post(TEST_URL, status=500, body="Internal Server Error")

            result = await run_load_test(
                url=TEST_URL,
                num_users=3,
                prompt="test",
                schema=SIMPLE_SCHEMA,
            )

        assert result.tracker.successful_requests == 0
        assert result.tracker.failed_requests == 3

    @pytest.mark.asyncio
    async def test_connection_errors(self):
        """Requests fail due to connection errors."""
        with aioresponses() as m:
            for _ in range(2):
                m.post(TEST_URL, exception=ConnectionError("Connection refused"))

            result = await run_load_test(
                url=TEST_URL,
                num_users=2,
                prompt="test",
            )

        assert result.tracker.failed_requests == 2

    @pytest.mark.asyncio
    async def test_without_schema(self):
        """Test load run without schema validation."""
        mock_response = _mock_completion_response("any text response")

        with aioresponses() as m:
            for _ in range(3):
                m.post(TEST_URL, payload=mock_response)

            result = await run_load_test(
                url=TEST_URL,
                num_users=3,
                prompt="test",
                schema=None,
            )

        assert result.tracker.successful_requests == 3
        # Without schema, all should be marked valid
        assert all(v.is_valid for v in result.validation_results)

    @pytest.mark.asyncio
    async def test_chat_endpoint_detection(self):
        """Verify that chat/completions URL triggers chat payload format."""
        valid_content = json.dumps({"answer": "hello", "confidence": 0.9})
        mock_response = _mock_chat_response(valid_content)

        with aioresponses() as m:
            m.post(TEST_CHAT_URL, payload=mock_response)

            result = await run_load_test(
                url=TEST_CHAT_URL,
                num_users=1,
                prompt="test",
                schema=SIMPLE_SCHEMA,
            )

        assert result.tracker.successful_requests == 1

    @pytest.mark.asyncio
    async def test_mixed_results(self):
        """Mix of successful and failed requests."""
        valid_content = json.dumps({"answer": "ok", "confidence": 0.8})
        mock_response = _mock_completion_response(valid_content)

        with aioresponses() as m:
            m.post(TEST_URL, payload=mock_response)
            m.post(TEST_URL, status=500, body="Error")
            m.post(TEST_URL, payload=mock_response)

            result = await run_load_test(
                url=TEST_URL,
                num_users=3,
                prompt="test",
                schema=SIMPLE_SCHEMA,
            )

        assert result.tracker.total_requests == 3
        assert result.tracker.successful_requests == 2
        assert result.tracker.failed_requests == 1

    @pytest.mark.asyncio
    async def test_latency_is_positive(self):
        """Ensure recorded latencies are positive numbers."""
        mock_response = _mock_completion_response("test")

        with aioresponses() as m:
            m.post(TEST_URL, payload=mock_response)

            result = await run_load_test(
                url=TEST_URL,
                num_users=1,
                prompt="test",
            )

        for r in result.tracker.results:
            assert r.latency_ms > 0

    @pytest.mark.asyncio
    async def test_streaming_response_is_preserved(self):
        """Ensure stream-mode reconstructed payload is not overwritten."""
        sse_body = (
            'data: {"choices":[{"delta":{"content":"hello stream"}}]}\n\n'
            "data: [DONE]\n\n"
        )

        with aioresponses() as m:
            m.post(TEST_CHAT_URL, status=200, body=sse_body)

            result = await run_load_test(
                url=TEST_CHAT_URL,
                num_users=1,
                prompt="test",
                schema=None,
                stream=True,
            )

        assert result.tracker.successful_requests == 1
        assert result.raw_responses[0] is not None
        assert "hello stream" in result.raw_responses[0]
        assert result.tracker.results[0].ttft_ms is not None
