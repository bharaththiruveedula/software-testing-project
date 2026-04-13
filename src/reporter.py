"""
Terminal Report Printer Module

Formats and prints a polished, colored performance and validation report
to the terminal.
"""

import sys
from typing import Optional

from metrics import PerformanceTracker
from load_runner import LoadTestResult


# ── ANSI Color Codes ──────────────────────────────────────────────────────────

class Colors:
    """ANSI escape codes for terminal colors."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"


def _color(text: str, color: str) -> str:
    """Wrap text in the given ANSI color."""
    return f"{color}{text}{Colors.RESET}"


def _header(title: str) -> str:
    """Create a formatted section header."""
    line = "─" * 50
    return f"\n{_color(line, Colors.DIM)}\n{_color(f'  {title}', Colors.BOLD + Colors.CYAN)}\n{_color(line, Colors.DIM)}"


def _status_badge(passed: bool) -> str:
    """Create a colored PASSED/FAILED badge."""
    if passed:
        return _color(" ✓ PASSED ", Colors.BOLD + Colors.BG_GREEN + Colors.WHITE)
    else:
        return _color(" ✗ FAILED ", Colors.BOLD + Colors.BG_RED + Colors.WHITE)


def print_report(
    result: LoadTestResult,
    pass_rate_threshold: float = 0.95,
) -> bool:
    """
    Print a formatted load test report to the terminal.

    Args:
        result: The LoadTestResult from the load test run.
        pass_rate_threshold: Minimum fraction of valid responses to pass (0.0–1.0).

    Returns:
        True if the test passed, False otherwise.
    """
    tracker = result.tracker
    summary = tracker.summary()

    total = summary["total_requests"]
    successful = summary["successful_requests"]
    failed = summary["failed_requests"]

    # ── Performance Metrics ───────────────────────────────────────────

    print(_header("Performance Metrics"))
    print(f"  {'Total Requests:':<28} {_color(str(total), Colors.WHITE + Colors.BOLD)}")
    print(f"  {'Successful Requests:':<28} {_color(str(successful), Colors.GREEN)}")
    if failed > 0:
        print(f"  {'Failed Requests:':<28} {_color(str(failed), Colors.RED)}")
    else:
        print(f"  {'Failed Requests:':<28} {_color(str(failed), Colors.GREEN)}")
    print(f"  {'Average Latency:':<28} {_color(f'{summary['average_latency_ms']:.0f}ms', Colors.YELLOW)}")
    print(f"  {'Max Latency:':<28} {_color(f'{summary['max_latency_ms']:.0f}ms', Colors.YELLOW)}")
    print(f"  {'Min Latency:':<28} {_color(f'{summary['min_latency_ms']:.0f}ms', Colors.YELLOW)}")
    print(f"  {'Average TTFT:':<28} {_color(f'{summary['average_ttft_ms']:.0f}ms', Colors.MAGENTA)}")
    print(f"  {'Max TTFT:':<28} {_color(f'{summary['max_ttft_ms']:.0f}ms', Colors.MAGENTA)}")

    # Token throughput (only show if token data is available)
    if summary.get("total_completion_tokens", 0) > 0:
        print(f"  {'Prompt Tokens:':<28} {_color(str(summary['total_prompt_tokens']), Colors.BLUE)}")
        print(f"  {'Completion Tokens:':<28} {_color(str(summary['total_completion_tokens']), Colors.BLUE)}")
        print(f"  {'Throughput:':<28} {_color(f'{summary['tokens_per_second']:.1f} tok/s', Colors.GREEN + Colors.BOLD)}")

    # ── Structural Validation ─────────────────────────────────────────

    print(_header("Structural Validation"))

    valid_count = sum(1 for v in result.validation_results if v.is_valid)
    total_validated = len(result.validation_results)

    if total_validated > 0:
        pass_rate = valid_count / total_validated
        pass_pct = pass_rate * 100
    else:
        pass_rate = 0.0
        pass_pct = 0.0

    pct_color = Colors.GREEN if pass_rate >= pass_rate_threshold else Colors.RED
    print(
        f"  {'Valid JSON Responses:':<28} "
        f"{_color(f'{valid_count}/{total_validated}', pct_color)} "
        f"({_color(f'{pass_pct:.0f}%', pct_color)})"
    )

    # List failed validations
    failed_validations = [v for v in result.validation_results if not v.is_valid]
    if failed_validations:
        print(
            f"  {'Failed Validations:':<28} "
            f"{_color(str(len(failed_validations)), Colors.RED)}"
        )
        for i, fv in enumerate(failed_validations, 1):
            msg = fv.error_message or "Unknown error"
            # Truncate long error messages
            if len(msg) > 100:
                msg = msg[:97] + "..."
            print(f"    {_color(f'#{i}:', Colors.DIM)} {_color(msg, Colors.RED)}")
    else:
        print(f"  {'Failed Validations:':<28} {_color('0', Colors.GREEN)}")

    # ── Final Status ──────────────────────────────────────────────────

    print(_header("Final Status"))

    passed = pass_rate >= pass_rate_threshold and total_validated > 0
    # Also fail if there were no successful requests at all
    if successful == 0:
        passed = False

    print(f"  {_status_badge(passed)}")

    if not passed:
        if successful == 0:
            print(
                f"  {_color('All requests failed. Check server connectivity.', Colors.RED)}"
            )
        elif pass_rate < pass_rate_threshold:
            print(
                f"  {_color(f'Target structure pass rate ({pass_rate_threshold*100:.0f}%) not met — actual: {pass_pct:.0f}%', Colors.RED)}"
            )

    print()  # blank line at end
    return passed
