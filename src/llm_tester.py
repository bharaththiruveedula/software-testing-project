#!/usr/bin/env python3
"""
LLM Inference Load and Validation Suite

A CLI tool that stress-tests an OpenAI-compatible inference server by
sending concurrent async HTTP requests, measuring performance metrics
(TTFT, latency), and validating response structure against JSON schemas.

Usage:
    python3 llm_tester.py <url> --users 20 --schema expected_format.json

Example:
    python3 llm_tester.py http://localhost:8000/v1/completions \\
        --users 20 --schema expected_format.json
"""

import argparse
import asyncio
import os
import sys

from html_reporter import generate_html_report, generate_comparison_report
from load_runner import run_load_test
from ramp_runner import run_ramp_test
from consistency import run_consistency_test
from config_loader import load_config
from niah_runner import run_niah_test
from comparator import analyze_results
from reporter import print_report
from validator import load_schema


DEFAULT_PROMPT = (
    "Please respond with a JSON object containing two fields: "
    '"answer" (a string with your response) and '
    '"confidence" (a number between 0 and 1 indicating your confidence). '
    "Respond ONLY with valid JSON, no additional text."
)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="llm_tester",
        description=(
            "Python-Based LLM Inference Load and Validation Suite — "
            "stress-test an OpenAI-compatible inference server with "
            "concurrent requests, measure TTFT & latency, and validate "
            "response structure."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 llm_tester.py http://localhost:8000/v1/completions --users 20\n"
            "  python3 llm_tester.py http://localhost:8000/v1/chat/completions "
            "--users 10 --schema expected_format.json\n"
        ),
    )

    parser.add_argument(
        "url",
        nargs="?",
        type=str,
        default=None,
        help="Target inference server URL (e.g., http://localhost:8000/v1/completions)",
    )
    parser.add_argument(
        "--users",
        "-u",
        type=int,
        default=10,
        help="Number of concurrent users to simulate (default: 10)",
    )
    parser.add_argument(
        "--schema",
        "-s",
        type=str,
        default=None,
        help="Path to a JSON schema file for response validation",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help='Model name to specify in the request payload (default: "default")',
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the model (default: 0.7)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt text to send with each request",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Minimum pass rate for structural validation (0.0–1.0, default: 0.95)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key for authentication (Bearer token). "
            "If not provided, falls back to GEMINI_API_KEY or OPENAI_API_KEY env vars."
        ),
    )
    parser.add_argument(
        "--stagger",
        type=float,
        default=0,
        help=(
            "Delay in milliseconds between launching each request. "
            "Useful for rate-limited APIs like Gemini free tier (default: 0 = all at once)"
        ),
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save an HTML report with charts (e.g., report.html)",
    )
    parser.add_argument(
        "--ramp",
        action="store_true",
        help="Enable ramp-up testing mode to find the server's breaking point",
    )
    parser.add_argument(
        "--ramp-levels",
        type=str,
        default="1,2,5,10,20",
        help="Comma-separated list of concurrency levels for ramp test (default: 1,2,5,10,20)",
    )
    parser.add_argument(
        "--consistency",
        type=int,
        default=0,
        metavar="N",
        help="Run N sequential requests to measure reliability and consistency.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file containing test scenarios.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming evaluation (Server-Sent Events) to harvest TPOT.",
    )
    parser.add_argument(
        "--niah",
        action="store_true",
        help="Enable Needle-In-A-Haystack context scaling evaluation.",
    )
    parser.add_argument(
        "--niah-lengths",
        type=str,
        default="1000,2000,3000,4000",
        help="Comma-separated block sizes for NIAH context lengths.",
    )

    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    if not args.url and not args.config:
        print("Error: You must provide either a URL argument or a --config file.", file=sys.stderr)
        return 1

    # Load schema if provided
    schema = None
    if args.schema:
        try:
            schema = load_schema(args.schema)
        except FileNotFoundError:
            print(f"Error: Schema file not found: {args.schema}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error loading schema: {e}", file=sys.stderr)
            return 1

    # Resolve API key: CLI flag > GEMINI_API_KEY > OPENAI_API_KEY
    api_key = (
        args.api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )

    # Banner
    api_key_display = f"***{api_key[-4:]}" if api_key else "None (no auth)"

    if args.config:
        try:
            scenarios = load_config(args.config)
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            return 1
            
        print(f"\n{'='*60}")
        print(f"  LLM Inference Load & Validation Suite — Config Mode")
        print(f"{'='*60}")
        print(f"  Config File:  {args.config}")
        print(f"  Scenarios:    {len(scenarios)}")
        print(f"{'='*60}")
        
        all_passed = True
        scenario_results = []
        for i, s in enumerate(scenarios, 1):
            s_schema = None
            if s.schema:
                try:
                    s_schema = load_schema(s.schema)
                except Exception as e:
                    print(f"Error loading schema {s.schema} for scenario {s.name}: {e}")
                    s_schema = None
                    
            print(f"\n[{i}/{len(scenarios)}] Running Scenario: {s.name}")
            print(f"  Model: {s.model} | Concurrency: {s.users} | Target: {s.url}")
            
            result = asyncio.run(
                run_load_test(
                    url=s.url,
                    num_users=s.users,
                    prompt=s.prompt,
                    schema=s_schema,
                    timeout_s=s.timeout,
                    model=s.model,
                    temperature=s.temperature,
                    api_key=api_key,
                    stagger_ms=s.stagger,
                )
            )
            scenario_results.append((s, result))
            passed = print_report(result, pass_rate_threshold=s.threshold)
            if not passed:
                all_passed = False

        if args.report:
            comp_metrics = analyze_results(scenario_results)
            generate_comparison_report(comp_metrics, output_path=args.report)
            print(f"Generated comparison report at {args.report}")
                
        return 0 if all_passed else 1

    # Banner for single run mode
    print(f"\n{'='*60}")
    print(f"  LLM Inference Load & Validation Suite")
    print(f"{'='*60}")
    print(f"  Target:       {args.url}")
    print(f"  Concurrency:  {args.users} users")
    print(f"  Model:        {args.model}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Timeout:      {args.timeout}s")
    print(f"  Schema:       {args.schema or 'None (skip validation)'}")
    print(f"  Threshold:    {args.threshold * 100:.0f}%")
    print(f"  API Key:      {api_key_display}")
    stagger_display = f"{args.stagger:.0f}ms" if args.stagger > 0 else "None (all at once)"
    print(f"  Stagger:      {stagger_display}")
    print(f"{'='*60}")

    if args.ramp:
        levels = [int(x.strip()) for x in args.ramp_levels.split(",")]
        result = asyncio.run(
            run_ramp_test(
                url=args.url,
                levels=levels,
                prompt=args.prompt,
                schema=schema,
                timeout_s=args.timeout,
                model=args.model,
                temperature=args.temperature,
                api_key=api_key,
                stagger_ms=args.stagger,
                pass_rate_threshold=args.threshold,
            )
        )
        passed = result.breaking_point is None

        # Just use the last result for HTML report for now if ramp is used
        # Note: A dedicated ramp report would be better, but this avoids crashes
        final_load_result = result.results[-1] if result.results else None
        
        if args.report and final_load_result:
            generate_html_report(
                result=final_load_result,
                output_path=args.report,
                pass_rate_threshold=args.threshold,
                ramp_result=result,
            )
        return 0 if passed else 1

    if args.consistency > 0:
        result = asyncio.run(
            run_consistency_test(
                url=args.url,
                iterations=args.consistency,
                prompt=args.prompt,
                schema=schema,
                timeout_s=args.timeout,
                model=args.model,
                temperature=args.temperature,
                api_key=api_key,
                pass_rate_threshold=args.threshold,
            )
        )
        return 0 if result.is_consistent else 1
        
    if args.niah:
        lengths = [int(x.strip()) for x in args.niah_lengths.split(",")]
        print(f"\n{'='*60}")
        print(f"  NIAH Context Scaling Test (- {len(lengths)} checkpoints)")
        print(f"{'='*60}")
        niah_obj = asyncio.run(
            run_niah_test(
                url=args.url,
                lengths=lengths,
                timeout_s=args.timeout,
                model=args.model,
                temperature=args.temperature,
                api_key=api_key,
                stream=args.stream,
            )
        )
        # Delegate HTML reporting...
        if args.report:
            from html_reporter import generate_niah_html_report
            generate_niah_html_report(niah_obj, output_path=args.report)
            
        return 0 if niah_obj.passed_all else 1

    print(f"\nRunning asynchronous load test with {args.users} concurrent users...")

    # Run the load test
    result = asyncio.run(
        run_load_test(
            url=args.url,
            num_users=args.users,
            prompt=args.prompt,
            schema=schema,
            timeout_s=args.timeout,
            model=args.model,
            temperature=args.temperature,
            api_key=api_key,
            stagger_ms=args.stagger,
            stream=args.stream,
        )
    )

    # Print report and determine exit code
    passed = print_report(result, pass_rate_threshold=args.threshold)

    # Generate HTML report if requested
    if args.report:
        generate_html_report(
            result=result,
            output_path=args.report,
            pass_rate_threshold=args.threshold,
        )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
