"""
Ramp-Up Testing Module

Increases concurrent users gradually over multiple stages to find the server's
breaking point.
"""

import sys
import asyncio
from dataclasses import dataclass
from typing import List, Optional

from load_runner import LoadTestResult, run_load_test
from reporter import _header, _color, Colors, print_report


@dataclass
class RampTestResult:
    """Aggregate result of a ramp-up load test."""
    levels: List[int]
    results: List[LoadTestResult]
    breaking_point: Optional[int]


async def run_ramp_test(
    url: str,
    levels: List[int],
    prompt: str,
    schema: Optional[dict] = None,
    timeout_s: float = 30.0,
    model: str = "default",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    stagger_ms: float = 0.0,
    pass_rate_threshold: float = 0.95,
) -> RampTestResult:
    """
    Execute a ramp-up test incrementally increasing users to find breaking point.

    Args:
        url, prompt, schema, timeout_s, model, temperature, api_key, stagger_ms: Same as run_load_test
        levels: List of concurrency levels to test (e.g., [1, 5, 10, 20]).
        pass_rate_threshold: Threshold below which a level is considered failed.
    
    Returns:
        A RampTestResult containing results for all levels and the identified breaking point.
    """
    all_results = []
    breaking_point = None

    print(_header("Ramp-Up Testing Mode"))
    print(f"  Levels to test: {levels}")
    print(f"  Failure threshold: {(1 - pass_rate_threshold) * 100:.0f}% error rate\n")

    for idx, users in enumerate(levels):
        print(f"--- Stage {idx + 1}: {users} Users ---")
        
        result = await run_load_test(
            url=url,
            num_users=users,
            prompt=prompt,
            schema=schema,
            timeout_s=timeout_s,
            model=model,
            temperature=temperature,
            api_key=api_key,
            stagger_ms=stagger_ms,
        )
        
        all_results.append(result)
        
        valid_count = sum(1 for v in result.validation_results if v.is_valid)
        total_validated = len(result.validation_results)
        pass_rate = valid_count / total_validated if total_validated > 0 else 0.0
        
        tracker = result.tracker
        is_success = tracker.successful_requests > 0 and pass_rate >= pass_rate_threshold

        avg_lat = tracker.average_latency()
        print(f"  Passed: {'Yes' if is_success else 'No'} | "
              f"Valid JSON: {valid_count}/{total_validated} | "
              f"Avg Latency: {avg_lat:.0f}ms\n")

        if not is_success and breaking_point is None:
            # We found the breaking point
            breaking_point = users
            print(_color(f"  [!] Breaking point detected at {users} concurrent users.", Colors.RED))
            break
            
        if idx < len(levels) - 1:
            # Small cooldown between stages
            await asyncio.sleep(2.0)

    if breaking_point is None:
        print(_color(f"  [✓] Server survived all ramp stages up to {levels[-1]} users.", Colors.GREEN))

    return RampTestResult(
        levels=levels[:len(all_results)],
        results=all_results,
        breaking_point=breaking_point,
    )
