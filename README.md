# Python-Based LLM Inference Load and Validation Suite

**Team Members:** Bharath Thiruveedula, Kaitlin Arnold, Ronald Joseph Marcelin Franklin

A CLI tool that stress-tests OpenAI-compatible inference servers (vLLM, llama.cpp, Ollama) by sending concurrent async HTTP requests, measuring performance metrics (**TTFT, ITL, TPOT**, Latency), and validating response structure against JSON schemas.

## Features

- **Concurrent Load Testing** — Uses Python's `asyncio` and `aiohttp` to simulate 100+ concurrent requests.
- **Ramp-Up Testing** — Safely probe server limits (`--ramp`) simulating step-wise concurrent load.
- **Context Scaling (NIAH)** — Needle In A Haystack mapping measuring exponential degradation mapping large context sequences (`--niah`).
- **Streaming Metrics** — Native integration with Server-Sent Events to measure **Time-Per-Output-Token (TPOT)** mirroring Official vLLM framework (`--stream`).
- **YAML Configuration** — Create pipelines using `evaluation_scenarios.yaml`.
- **Graphical HTML Reports** — Automatically builds Chart.js rich HTML summaries representing Time Series matrices.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast dependency management and execution isolation.

```bash
# Install all dependencies
uv sync --dev
```

## Quick Start
All core modules reside in `src/`.

### 1. Basic Concurrency Load Test
```bash
uv run python3 src/llm_tester.py http://localhost:8000/v1/completions \
    --users 50 \
    --stream \
    --report reports/my_load_test.html
```

### 2. Context Scaling (Needle In A Haystack)
Map out Memory KV Cache drop-offs automatically injecting a target-needle over varied context blocks:
```bash
uv run python3 src/llm_tester.py http://localhost:8000/v1/chat/completions \
    --niah \
    --niah-lengths "1000,2000,3000,4000" \
    --stream \
    --report reports/niah_scale.html
```

### 3. CI/CD Automated Pipelines
Execute a full array of cross-validated parameters via declarative YAML blocks:
```bash
uv run python3 src/llm_tester.py --config configs/evaluation_scenarios.yaml --report reports/comparison.html
```

## Adding New YAML Test Cases
The system reads YAML scenarios globally. To add your own server:
1. Open `configs/evaluation_scenarios.yaml` 
2. Append a new `- name: "llama.cpp RTX 5070"` block specifying:
    * `url`: The generate endpoint.
    * `users`: Peak concurrency target.
    * `stagger`: Milliseconds offset between threads.
    * `threshold`: JSON schema valid return rate expectations (e.g. 0.95).

## Running Tests
This repository features an integration validation engine using Pytest configured for `src/`.
```bash
# Execute Pytest reporting HTML coverage
uv run pytest tests/ --cov=src/ --cov-report=html
```

## Repository Structure

```
software-testing-project/
├── src/                       # Main Code Hierarchy
│   ├── llm_tester.py          # Main CLI entry point
│   ├── load_runner.py         # Async HTTP Stream Extractor
│   ├── niah_runner.py         # Context Scaling Engine
│   ├── ramp_runner.py         # Incremental Load Tester
│   ├── metrics.py             # Performance trackers (TPOT, TTFT)
│   ├── html_reporter.py       # HTML graphical injection system
│   └── validator.py           # JSON Schema Validator
├── configs/                   # Execution environment configuration
│   ├── evaluation_scenarios.yaml 
│   ├── test_scenarios.yaml
│   └── expected_format.json
├── tests/                     # Unit Testing Framework
├── reports/                   # Persisted Analysis Results
└── README.md
```
