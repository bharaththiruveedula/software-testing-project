"""
Structural Validation Module

Validates that the inference server's response text conforms to a given
JSON schema using the jsonschema library.
"""

import json
from dataclasses import dataclass
from typing import Optional

import jsonschema


@dataclass
class ValidationResult:
    """Result of validating a single response against a JSON schema."""
    is_valid: bool
    error_message: Optional[str] = None


def validate_response(response_text: str, schema: dict) -> ValidationResult:
    """
    Parse the response text as JSON and validate it against the given schema.

    Args:
        response_text: Raw text from the inference server's response.
        schema: A JSON Schema dictionary to validate against.

    Returns:
        A ValidationResult indicating whether the response is valid.
    """
    # Step 1: Attempt to parse as JSON
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Payload failed to parse as JSON: {e}",
        )

    # Step 2: Validate against the schema
    try:
        jsonschema.validate(instance=parsed, schema=schema)
    except jsonschema.ValidationError as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Schema validation failed: {e.message}",
        )
    except jsonschema.SchemaError as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Invalid schema: {e.message}",
        )

    return ValidationResult(is_valid=True)


def load_schema(schema_path: str) -> dict:
    """
    Load a JSON schema from a file path.

    Args:
        schema_path: Path to the JSON schema file.

    Returns:
        Parsed JSON schema dictionary.

    Raises:
        FileNotFoundError: If the schema file does not exist.
        json.JSONDecodeError: If the schema file is not valid JSON.
    """
    with open(schema_path, "r") as f:
        return json.load(f)
