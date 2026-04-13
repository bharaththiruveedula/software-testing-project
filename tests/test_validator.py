"""
Unit tests for the Structural Validation Module (validator.py).
"""

import json
import os
import tempfile

import pytest

from validator import ValidationResult, load_schema, validate_response


# ── Sample Schemas ────────────────────────────────────────────────────────────

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["answer", "confidence"],
    "additionalProperties": False,
}

FLEXIBLE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
}


# ── Tests: validate_response ─────────────────────────────────────────────────

class TestValidateResponseValidJSON:
    """Test validation of well-formed JSON that matches the schema."""

    def test_valid_response(self):
        response = json.dumps({"answer": "Hello world", "confidence": 0.95})
        result = validate_response(response, SIMPLE_SCHEMA)
        assert result.is_valid is True
        assert result.error_message is None

    def test_valid_response_min_confidence(self):
        response = json.dumps({"answer": "test", "confidence": 0.0})
        result = validate_response(response, SIMPLE_SCHEMA)
        assert result.is_valid is True

    def test_valid_response_max_confidence(self):
        response = json.dumps({"answer": "test", "confidence": 1.0})
        result = validate_response(response, SIMPLE_SCHEMA)
        assert result.is_valid is True


class TestValidateResponseInvalidJSON:
    """Test validation when the response is not valid JSON."""

    def test_plain_text(self):
        result = validate_response("This is not JSON", SIMPLE_SCHEMA)
        assert result.is_valid is False
        assert "Payload failed to parse as JSON" in result.error_message

    def test_empty_string(self):
        result = validate_response("", SIMPLE_SCHEMA)
        assert result.is_valid is False
        assert "Payload failed to parse as JSON" in result.error_message

    def test_partial_json(self):
        result = validate_response('{"answer": "hello"', SIMPLE_SCHEMA)
        assert result.is_valid is False
        assert "Payload failed to parse as JSON" in result.error_message

    def test_html_response(self):
        result = validate_response("<html><body>Error</body></html>", SIMPLE_SCHEMA)
        assert result.is_valid is False
        assert "Payload failed to parse as JSON" in result.error_message


class TestValidateResponseSchemaViolations:
    """Test validation when JSON is valid but doesn't match the schema."""

    def test_missing_required_field(self):
        response = json.dumps({"answer": "hello"})
        result = validate_response(response, SIMPLE_SCHEMA)
        assert result.is_valid is False
        assert "Schema validation failed" in result.error_message

    def test_wrong_type_confidence(self):
        response = json.dumps({"answer": "hello", "confidence": "high"})
        result = validate_response(response, SIMPLE_SCHEMA)
        assert result.is_valid is False
        assert "Schema validation failed" in result.error_message

    def test_confidence_out_of_range(self):
        response = json.dumps({"answer": "hello", "confidence": 1.5})
        result = validate_response(response, SIMPLE_SCHEMA)
        assert result.is_valid is False
        assert "Schema validation failed" in result.error_message

    def test_additional_properties_not_allowed(self):
        response = json.dumps(
            {"answer": "hello", "confidence": 0.5, "extra_field": "oops"}
        )
        result = validate_response(response, SIMPLE_SCHEMA)
        assert result.is_valid is False
        assert "Schema validation failed" in result.error_message

    def test_wrong_root_type(self):
        response = json.dumps([1, 2, 3])
        result = validate_response(response, SIMPLE_SCHEMA)
        assert result.is_valid is False
        assert "Schema validation failed" in result.error_message

    def test_null_value(self):
        response = json.dumps({"answer": None, "confidence": 0.5})
        result = validate_response(response, SIMPLE_SCHEMA)
        assert result.is_valid is False
        assert "Schema validation failed" in result.error_message


class TestValidateResponseFlexibleSchema:
    """Test with a more flexible schema."""

    def test_additional_properties_allowed(self):
        response = json.dumps({"answer": "hello", "extra": "allowed"})
        result = validate_response(response, FLEXIBLE_SCHEMA)
        assert result.is_valid is True

    def test_nested_objects_allowed(self):
        response = json.dumps({"answer": "hello", "details": {"key": "value"}})
        result = validate_response(response, FLEXIBLE_SCHEMA)
        assert result.is_valid is True


# ── Tests: load_schema ────────────────────────────────────────────────────────

class TestLoadSchema:
    """Test schema loading from file."""

    def test_load_valid_schema(self, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(SIMPLE_SCHEMA))
        loaded = load_schema(str(schema_file))
        assert loaded == SIMPLE_SCHEMA

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_schema("/nonexistent/path/schema.json")

    def test_load_invalid_json_file(self, tmp_path):
        schema_file = tmp_path / "bad.json"
        schema_file.write_text("not valid json {{{")
        with pytest.raises(json.JSONDecodeError):
            load_schema(str(schema_file))


# ── Tests: ValidationResult ──────────────────────────────────────────────────

class TestValidationResult:
    """Test the ValidationResult dataclass."""

    def test_valid_result(self):
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.error_message is None

    def test_invalid_result_with_message(self):
        result = ValidationResult(is_valid=False, error_message="test error")
        assert result.is_valid is False
        assert result.error_message == "test error"
