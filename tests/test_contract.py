"""
Test #1 — Contract conformance.

Methodology: Contract testing / schema conformance.
Precedent: llama.cpp ``tools/server/tests/unit/test_openai_schema.py`` uses
OpenAPI-derived JSON schemas to assert every response shape is correct.

Assertion: For every response, the OpenAI *envelope* (id, object, choices,
message role/content, finish_reason) conforms to the
``configs/openai_schemas/chat_completion.json`` schema.

Note this tests the **protocol wrapper** — not the generated content's
semantic correctness. Semantic quality is out of scope per the proposal.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
from aioresponses import aioresponses

from load_runner import run_load_test
from validator import validate_response

from tests.conftest import make_chat_completion


# ── Assertion: the schema catches known-bad shapes ───────────────────────────


class TestSchemaRejectsViolations:
    """If the schema itself can't flag violations, the test layer is useless."""

    def test_accepts_well_formed_response(self, chat_completion_schema: dict):
        body = make_chat_completion("hello")
        r = validate_response(json.dumps(body), chat_completion_schema)
        assert r.is_valid, r.error_message

    def test_rejects_wrong_object_tag(self, chat_completion_schema: dict):
        body = make_chat_completion("hello")
        body["object"] = "not.chat.completion"
        r = validate_response(json.dumps(body), chat_completion_schema)
        assert not r.is_valid

    def test_rejects_missing_finish_reason(self, chat_completion_schema: dict):
        body = make_chat_completion("hello")
        del body["choices"][0]["finish_reason"]
        r = validate_response(json.dumps(body), chat_completion_schema)
        assert not r.is_valid

    def test_rejects_bad_id_prefix(self, chat_completion_schema: dict):
        body = make_chat_completion("hello", completion_id="weird-id-123")
        r = validate_response(json.dumps(body), chat_completion_schema)
        assert not r.is_valid

    def test_rejects_non_array_choices(self, chat_completion_schema: dict):
        body: Dict[str, Any] = make_chat_completion("hello")
        body["choices"] = {"oops": "not an array"}
        r = validate_response(json.dumps(body), chat_completion_schema)
        assert not r.is_valid

    def test_rejects_empty_choices_array(self, chat_completion_schema: dict):
        body = make_chat_completion("hello")
        body["choices"] = []
        r = validate_response(json.dumps(body), chat_completion_schema)
        assert not r.is_valid

    def test_rejects_bad_role(self, chat_completion_schema: dict):
        body = make_chat_completion("hello")
        body["choices"][0]["message"]["role"] = "definitely-not-a-role"
        r = validate_response(json.dumps(body), chat_completion_schema)
        assert not r.is_valid


# ── Against the client: every response validates ────────────────────────────


class TestRunnerResponsesConform:
    """Exercising run_load_test against a conformant mock should yield
    schema-clean responses on every request (representative of a healthy
    server)."""

    @pytest.mark.asyncio
    async def test_conformant_server_passes_contract(
        self,
        chat_completion_schema: dict,
    ):
        url = "http://localhost:0/v1/chat/completions"
        mock = make_chat_completion("A deterministic greeting.")
        with aioresponses() as m:
            for _ in range(10):
                m.post(url, payload=mock)
            result = await run_load_test(
                url=url,
                num_users=10,
                prompt="hi",
                schema=None,
                temperature=0.0,
            )

        for raw in result.raw_responses:
            assert raw is not None
            r = validate_response(raw, chat_completion_schema)
            assert r.is_valid, r.error_message


# ── Integration: real server envelope conformance ───────────────────────────


@pytest.mark.integration
class TestContractIntegration:

    @pytest.mark.asyncio
    async def test_live_response_envelope_is_conformant(
        self,
        chat_completions_url: str,
        chat_completion_schema: dict,
        model_name: str,
    ):
        result = await run_load_test(
            url=chat_completions_url,
            num_users=3,
            prompt="Say hi.",
            schema=None,
            temperature=0.0,
            model=model_name,
        )
        for raw in result.raw_responses:
            assert raw is not None
            r = validate_response(raw, chat_completion_schema)
            assert r.is_valid, f"Envelope violation: {r.error_message}"
