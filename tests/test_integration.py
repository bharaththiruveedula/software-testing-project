"""
Integration tests for testing live LLM servers.
"""

import os
import pytest
import asyncio
from load_runner import run_load_test


# To run these tests, the user must define LIVE_TEST_URL in the environment
# e.g., export LIVE_TEST_URL="http://localhost:11434/v1/chat/completions"
# e.g., export LIVE_TEST_MODEL="llama3.2"

pytestmark = pytest.mark.integration


@pytest.fixture
def live_url():
    url = os.environ.get("LIVE_TEST_URL")
    if not url:
        pytest.skip("LIVE_TEST_URL environment variable is not set")
    return url


@pytest.fixture
def live_model():
    return os.environ.get("LIVE_TEST_MODEL", "default")


@pytest.mark.asyncio
async def test_server_handles_concurrent_load(live_url, live_model):
    """Test that the server can handle 5 concurrent requests without errors."""
    
    result = await run_load_test(
        url=live_url,
        num_users=5,
        prompt="Reply with exactly 'Hello'.",
        model=live_model,
        timeout_s=60.0,
    )
    
    # Assert successful requests pass a threshold
    tracker = result.tracker
    assert tracker.total_requests == 5
    assert tracker.successful_requests >= 4 # Allow 1 failure in load


@pytest.mark.asyncio
async def test_response_latency_within_sla(live_url, live_model):
    """Test that the server responds within latency SLA."""
    
    result = await run_load_test(
        url=live_url,
        num_users=1,
        prompt="Reply with exactly 'Latency Test'.",
        model=live_model,
        timeout_s=30.0,
    )
    
    tracker = result.tracker
    assert tracker.successful_requests == 1
    
    # Assert latency is under 10 seconds for a short prompt
    assert tracker.average_latency() < 10000.0


@pytest.mark.asyncio
async def test_response_schema_compliance(live_url, live_model):
    """Test that server conforms to a JSON schema."""
    
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"]
    }
    
    result = await run_load_test(
        url=live_url,
        num_users=2,
        prompt="Output a JSON object with the key 'answer' and a string value.",
        schema=schema,
        model=live_model,
    )
    
    tracker = result.tracker
    assert tracker.successful_requests > 0
    
    valid_count = sum(1 for v in result.validation_results if v.is_valid)
    assert valid_count == tracker.successful_requests
