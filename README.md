# Python-Based LLM Inference Load and Validation Suite

**Team Members:** Bharath Thiruveedula, Kaitlin Arnold, Ronald Joseph Marcelin Franklin

A CLI tool that stress-tests an OpenAI-compatible inference server by sending concurrent async HTTP requests, measuring performance metrics (TTFT, latency), and validating response structure against JSON schemas.

## Features

- **Concurrent Load Testing** — Uses Python's `asyncio` and `aiohttp` to simulate multiple users sending requests simultaneously
- **TTFT Measurement** — Measures Time-To-First-Token via streamed response chunks
- **Latency Tracking** — Records per-request latency with aggregate statistics (avg, min, max)
- **JSON Schema Validation** — Validates each response against a user-defined JSON schema using `jsonschema`
- **Formatted CLI Reports** — Prints color-coded performance and validation reports to the terminal
- **Pass/Fail Determination** — Exits with appropriate status code based on configurable pass rate threshold

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast dependency management.

```bash
# Install all dependencies (including dev/test)
uv sync --dev
```

## Usage

```bash
uv run python3 llm_tester.py <URL> [OPTIONS]
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `url` | Target inference server URL | *(required)* |
| `--users`, `-u` | Number of concurrent users | 10 |
| `--schema`, `-s` | Path to JSON schema file | None |
| `--timeout` | Request timeout in seconds | 30 |
| `--model` | Model name for the request | "default" |
| `--temperature` | Sampling temperature | 0.7 |
| `--prompt` | Prompt text to send | *(built-in default)* |
| `--threshold` | Minimum pass rate (0.0–1.0) | 0.95 |

### Examples

```bash
# Basic load test with 20 concurrent users
uv run python3 llm_tester.py http://localhost:8000/v1/completions --users 20

# With schema validation
uv run python3 llm_tester.py http://localhost:8000/v1/completions \
    --users 20 --schema expected_format.json

# Chat completions endpoint with custom settings
uv run python3 llm_tester.py http://localhost:8000/v1/chat/completions \
    --users 10 --model "llama-3" --temperature 0.5 --schema expected_format.json
```

### Example Output

```
============================================================
  LLM Inference Load & Validation Suite
============================================================
  Target:       http://localhost:8000/v1/completions
  Concurrency:  20 users
  Model:        default
  Temperature:  0.7
  Timeout:      30.0s
  Schema:       expected_format.json
  Threshold:    95%
============================================================

Running asynchronous load test with 20 concurrent users...

──────────────────────────────────────────────────────
  Performance Metrics
──────────────────────────────────────────────────────
  Total Requests:              20
  Successful Requests:         20
  Failed Requests:             0
  Average Latency:             510ms
  Max Latency:                 535ms
  Min Latency:                 490ms
  Average TTFT:                45ms
  Max TTFT:                    78ms

──────────────────────────────────────────────────────
  Structural Validation
──────────────────────────────────────────────────────
  Valid JSON Responses:        18/20 (90%)
  Failed Validations:          2
    #1: Payload failed to parse as JSON at temp 1.8
    #2: Schema validation failed: 'confidence' is a required property

──────────────────────────────────────────────────────
  Final Status
──────────────────────────────────────────────────────
   ✗ FAILED 
  Target structure pass rate (95%) not met — actual: 90%
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
software-testing-project/
├── llm_tester.py              # Main CLI entry point
├── load_runner.py             # Async load runner (asyncio + aiohttp)
├── metrics.py                 # Performance tracking module
├── validator.py               # JSON schema validation module
├── reporter.py                # Formatted terminal report printer
├── expected_format.json       # Example JSON schema
├── pyproject.toml             # Project config & dependencies (uv)
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py        # Tests for metrics module
│   ├── test_validator.py      # Tests for validator module
│   ├── test_load_runner.py    # Tests for load runner (mocked HTTP)
│   └── test_reporter.py       # Tests for reporter
└── README.md
```

## Architecture

The tool is organized into four core modules:

1. **`metrics.py`** — `PerformanceTracker` class that collects per-request timing data and computes aggregate statistics
2. **`validator.py`** — Validates response text by parsing as JSON and checking against a `jsonschema` schema
3. **`load_runner.py`** — Async HTTP engine that sends concurrent requests, measures TTFT via streaming, and orchestrates validation
4. **`reporter.py`** — Formats and prints a polished, ANSI-colored terminal report with pass/fail determination

The main entry point (`llm_tester.py`) ties them together via `argparse` CLI.
