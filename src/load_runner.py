"""
Async Load Runner Module

Uses asyncio and aiohttp to send concurrent HTTP requests to an
OpenAI-compatible inference server, measuring latency and TTFT
(Time-To-First-Token) via streaming.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp

from metrics import PerformanceTracker
from validator import ValidationResult, validate_response


@dataclass
class LoadTestResult:
    """Aggregate result of a complete load test run."""
    tracker: PerformanceTracker
    validation_results: List[ValidationResult]
    raw_responses: List[Optional[str]]


async def _send_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    request_id: int,
    tracker: PerformanceTracker,
    timeout_s: float,
    schema: Optional[dict],
    api_key: Optional[str] = None,
) -> tuple:
    """
    Send a single async HTTP request and record metrics.

    Returns:
        A tuple of (response_text or None, ValidationResult or None).
    """
    start_time = time.perf_counter()
    ttft_ms = None
    tpot_ms = None
    response_text = None
    validation_result = None

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with session.post(
            url,
            json=payload,
            timeout=timeout,
            headers=headers,
        ) as response:
            # If streaming is supported, measure TTFT from the first chunk
            chunks = []
            if payload.get("stream", False):
                import json
                
                first_chunk = True
                most_recent_timestamp = start_time
                itl_ms_list = []
                generated_text = ""
                final_usage = None
                
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    
                    try:
                        decoded = chunk_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                        
                    if decoded.startswith(":"): # ping
                        continue
                    
                    if decoded.startswith("data:"):
                        chunk_str = decoded.removeprefix("data:").strip()
                        if chunk_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(chunk_str)
                            timestamp = time.perf_counter()
                            
                            # Parse token data
                            if "choices" in data and data["choices"]:
                                choice = data["choices"][0]
                                content = ""
                                if "delta" in choice and "content" in choice["delta"]:
                                    content = choice["delta"]["content"]
                                elif "text" in choice:
                                    content = choice["text"]
                                    
                                if content:
                                    generated_text += str(content)
                                    
                                if first_chunk:
                                    ttft_ms = (timestamp - start_time) * 1000
                                    first_chunk = False
                                else:
                                    itl_ms_list.append((timestamp - most_recent_timestamp) * 1000)
                                    
                                most_recent_timestamp = timestamp
                            
                            if "usage" in data and data["usage"]:
                                final_usage = data["usage"]
                        except json.JSONDecodeError:
                            pass
                
                if itl_ms_list:
                    tpot_ms = sum(itl_ms_list) / len(itl_ms_list)
                    
                # Reconstruct a faux OpenAI JSON response internally for the validator
                faux_payload = {
                    "choices": [{"message": {"content": generated_text}}],
                    "usage": final_usage or {}
                }
                response_text = json.dumps(faux_payload)
            else:
                first_chunk = True
                async for chunk in response.content.iter_any():
                    if first_chunk:
                        ttft_ms = (time.perf_counter() - start_time) * 1000
                        first_chunk = False
                    chunks.append(chunk)
                response_text = b"".join(chunks).decode("utf-8", errors="replace")

            response_text = b"".join(chunks).decode("utf-8", errors="replace")
            latency_ms = (time.perf_counter() - start_time) * 1000

            if response.status >= 400:
                tracker.record(
                    request_id=request_id,
                    latency_ms=latency_ms,
                    ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms,
                    success=False,
                    error=f"HTTP {response.status}: {response_text[:200]}",
                )
                validation_result = ValidationResult(
                    is_valid=False,
                    error_message=f"HTTP error {response.status}",
                )
            else:
                # Parse token usage from response if available
                prompt_tokens, completion_tokens = _extract_token_usage(
                    response_text
                )

                tracker.record(
                    request_id=request_id,
                    latency_ms=latency_ms,
                    ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms,
                    success=True,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

                # Validate response if schema is provided
                if schema is not None:
                    # Try to extract the generated text from OpenAI-style response
                    extracted_text = _extract_generated_text(response_text)
                    validation_result = validate_response(extracted_text, schema)
                else:
                    validation_result = ValidationResult(is_valid=True)

    except asyncio.TimeoutError:
        latency_ms = (time.perf_counter() - start_time) * 1000
        tracker.record(
            request_id=request_id,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            success=False,
            error=f"Request timed out after {timeout_s}s",
        )
        validation_result = ValidationResult(
            is_valid=False,
            error_message=f"Request timed out after {timeout_s}s",
        )

    except aiohttp.ClientError as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        tracker.record(
            request_id=request_id,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            success=False,
            error=str(e),
        )
        validation_result = ValidationResult(
            is_valid=False,
            error_message=f"Connection error: {e}",
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        tracker.record(
            request_id=request_id,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            success=False,
            error=str(e),
        )
        validation_result = ValidationResult(
            is_valid=False,
            error_message=f"Unexpected error: {e}",
        )

    return response_text, validation_result


def _extract_token_usage(response_text: str) -> tuple:
    """
    Extract token usage from an OpenAI-compatible response.

    Returns:
        A tuple of (prompt_tokens, completion_tokens). Defaults to (0, 0)
        if usage data is not available.
    """
    import json

    try:
        data = json.loads(response_text)
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        return prompt_tokens, completion_tokens
    except (json.JSONDecodeError, AttributeError):
        return 0, 0


def _strip_markdown_fences(text: str) -> str:
    """
    Strip markdown code fences from text if present.

    Many LLMs wrap JSON in ```json ... ``` blocks even when told not to.
    This strips those fences so the raw JSON can be validated.
    """
    import re

    stripped = text.strip()
    # Match ```json ... ``` or ``` ... ```
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def _extract_generated_text(response_text: str) -> str:
    """
    Attempt to extract the generated text from an OpenAI-compatible response.

    For /v1/completions, the text lives in choices[0].text.
    For /v1/chat/completions, the text lives in choices[0].message.content.

    If extraction fails, return the raw response text for direct validation.
    """
    import json

    try:
        data = json.loads(response_text)
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            # Chat completions format
            if "message" in choice and "content" in choice["message"]:
                return _strip_markdown_fences(choice["message"]["content"])
            # Completions format
            if "text" in choice:
                return _strip_markdown_fences(choice["text"])
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass

    # Fall back to raw text
    return response_text


def _build_completions_payload(
    prompt: str,
    model: str,
    temperature: float,
) -> dict:
    """Build a request payload for the /v1/completions endpoint."""
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": temperature,
        "stream": False,
    }


def _build_chat_payload(
    prompt: str,
    model: str,
    temperature: float,
) -> dict:
    """Build a request payload for the /v1/chat/completions endpoint."""
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 512,
        "temperature": temperature,
        "stream": False,
    }


async def _staggered_request(
    delay_s: float,
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    request_id: int,
    tracker: "PerformanceTracker",
    timeout_s: float,
    schema: Optional[dict],
    api_key: Optional[str] = None,
) -> tuple:
    """Wait for the stagger delay, then send the request."""
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    return await _send_request(
        session=session,
        url=url,
        payload=payload,
        request_id=request_id,
        tracker=tracker,
        timeout_s=timeout_s,
        schema=schema,
        api_key=api_key,
    )


async def run_load_test(
    url: str,
    num_users: int,
    prompt: str,
    schema: Optional[dict] = None,
    timeout_s: float = 30.0,
    model: str = "default",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    stagger_ms: float = 0.0,
    stream: bool = False,
) -> LoadTestResult:
    """
    Execute a concurrent load test against the given inference server URL.

    Args:
        url: Full URL of the inference endpoint (e.g., http://localhost:8000/v1/completions).
        num_users: Number of concurrent requests to simulate.
        prompt: The prompt text to send with each request.
        schema: Optional JSON schema to validate responses against.
        timeout_s: Timeout per request in seconds.
        model: Model name to specify in the request payload.
        temperature: Sampling temperature for the model.
        api_key: Optional API key for authentication (Bearer token).
        stagger_ms: Delay in milliseconds between launching each request
                    (0 = all at once, useful for rate-limited APIs).
        stream: Whether to request streaming responses from the API (for TPOT tracking).

    Returns:
        A LoadTestResult containing performance metrics and validation results.
    """
    tracker = PerformanceTracker()

    # Detect endpoint type based on URL
    if "chat/completions" in url:
        payload = _build_chat_payload(prompt, model, temperature)
    else:
        payload = _build_completions_payload(prompt, model, temperature)
        
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

    stagger_s = stagger_ms / 1000.0

    async with aiohttp.ClientSession() as session:
        tasks = [
            _staggered_request(
                delay_s=i * stagger_s,
                session=session,
                url=url,
                payload=payload,
                request_id=i + 1,
                tracker=tracker,
                timeout_s=timeout_s,
                schema=schema,
                api_key=api_key,
            )
            for i in range(num_users)
        ]

        results = await asyncio.gather(*tasks)

    raw_responses = [r[0] for r in results]
    validation_results = [r[1] for r in results if r[1] is not None]

    return LoadTestResult(
        tracker=tracker,
        validation_results=validation_results,
        raw_responses=raw_responses,
    )
