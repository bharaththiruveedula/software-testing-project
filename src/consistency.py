"""
Consistency Testing Module

Sends the same prompt multiple times sequentially to measure reliability
and output stability.
"""

import math
import statistics
import asyncio
from dataclasses import dataclass
from typing import List, Optional

from load_runner import run_load_test, LoadTestResult
from reporter import _header, _color, Colors


@dataclass
class ConsistencyResult:
    """Result of a consistency test run."""
    iterations: int
    latencies: List[float]
    pass_rate: float
    avg_latency: float
    latency_variance: float
    latency_cv: float  # Coefficient of Variation (stddev/mean)
    is_consistent: bool


async def run_consistency_test(
    url: str,
    iterations: int,
    prompt: str,
    schema: Optional[dict] = None,
    timeout_s: float = 30.0,
    model: str = "default",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    pass_rate_threshold: float = 0.95,
) -> ConsistencyResult:
    """
    Run N sequential requests to measure reliability.
    """
    latencies = []
    valid_count = 0

    print(_header("Consistency Testing Mode"))
    print(f"  Iterations: {iterations}")
    print(f"  Target pass rate: {pass_rate_threshold * 100:.0f}%\n")

    for i in range(iterations):
        print(f"  [{i+1}/{iterations}] Requesting...", end="", flush=True)
        # Using load runner with 1 user
        result = await run_load_test(
            url=url,
            num_users=1,
            prompt=prompt,
            schema=schema,
            timeout_s=timeout_s,
            model=model,
            temperature=temperature,
            api_key=api_key,
        )

        res = result.tracker.results[0] if result.tracker.results else None
        if res and res.success:
            latencies.append(res.latency_ms)
            
            # Check validation
            val = result.validation_results[0] if result.validation_results else None
            is_valid = val.is_valid if val else False
            if is_valid:
                valid_count += 1
                print(_color(f" ✓ {res.latency_ms:.0f}ms", Colors.GREEN))
            else:
                print(_color(f" ✗ Invalid JSON ({res.latency_ms:.0f}ms)", Colors.RED))
        else:
            print(_color(" ✗ Failed", Colors.RED))

        # Small cooldown
        await asyncio.sleep(1.0)

    pass_rate = valid_count / iterations if iterations > 0 else 0.0
    
    if latencies:
        avg_lat = statistics.mean(latencies)
        variance = statistics.variance(latencies) if len(latencies) > 1 else 0.0
        stddev = math.sqrt(variance)
        cv = (stddev / avg_lat) if avg_lat > 0 else 0.0
    else:
        avg_lat = 0.0
        variance = 0.0
        cv = 0.0

    is_consistent = pass_rate >= pass_rate_threshold

    print(_header("Consistency Report"))
    color_pass = Colors.GREEN if is_consistent else Colors.RED
    print(f"  Reliability Score:  {_color(f'{pass_rate*100:.0f}%', color_pass)}")
    print(f"  Avg Latency:        {avg_lat:.0f}ms")
    print(f"  Latency StdDev:     {math.sqrt(variance):.0f}ms")
    # Low CV means highly consistent latency. Usually < 0.2 is good.
    cv_color = Colors.GREEN if cv < 0.2 else (Colors.YELLOW if cv < 0.5 else Colors.RED)
    print(f"  Latency CV:         {_color(f'{cv:.2f}', cv_color)}")
    print(f"  Overall Status:     {_color('CONSISTENT', Colors.GREEN) if is_consistent else _color('INCONSISTENT', Colors.RED)}\n")

    return ConsistencyResult(
        iterations=iterations,
        latencies=latencies,
        pass_rate=pass_rate,
        avg_latency=avg_lat,
        latency_variance=variance,
        latency_cv=cv,
        is_consistent=is_consistent,
    )
