import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional

from load_runner import run_load_test, LoadTestResult
from validator import ValidationResult


@dataclass
class NIAHResultLevel:
    context_length: int
    load_result: LoadTestResult
    passed: bool


@dataclass
class NIAHTestResult:
    levels: List[NIAHResultLevel]

    @property
    def passed_all(self) -> bool:
        return all(level.passed for level in self.levels)


def _generate_haystack(target_tokens: int) -> str:
    """
    Generate a pseudo-random block of text that mimics 'target_tokens' length.
    We estimate 1 token ≈ 4 characters of English text.
    """
    chunk = (
        "The infrastructure of modern artificial intelligence relies heavily upon "
        "large-scale transformer networks. By understanding context limits, we "
        "can determine if the model accurately retrieves facts embedded deep within. "
    )
    chunk_tokens = len(chunk) // 4
    repeats = (target_tokens // chunk_tokens) + 1
    return (chunk * repeats)[: target_tokens * 4]


async def run_niah_test(
    url: str,
    lengths: List[int],
    timeout_s: float,
    model: str,
    temperature: float,
    api_key: Optional[str],
    stream: bool,
) -> dict:
    """
    Execute the Needle In A Haystack context expansion test.
    """
    results_obj = NIAHTestResult(levels=[])
    
    needle_fact = "The secure vault passcode is ZETA-9-ALPHA."
    system_instruction = "Retrieve the passcode."

    for length in lengths:
        print(f"\n[NIAH] Testing Context Length: {length} tokens...")
        
        # Build prompt
        haystack_prefix = _generate_haystack(int(length * 0.5))
        haystack_suffix = _generate_haystack(int(length * 0.5))
        
        prompt = (
            f"Please read the following text:\n\n{haystack_prefix}\n"
            f"{needle_fact}\n{haystack_suffix}\n\n"
            f"Question: {system_instruction}"
        )
        
        result: LoadTestResult = await run_load_test(
            url=url,
            num_users=1,
            prompt=prompt,
            schema=None,
            timeout_s=timeout_s,
            model=model,
            temperature=temperature,
            api_key=api_key,
            stagger_ms=0,
            stream=stream,
        )
        
        # Verify the needle was extracted accurately
        passed = False
        raw = result.raw_responses[0]
        if raw and "ZETA-9-ALPHA" in raw:
            passed = True
            
        # We manually overwrite validation for report purposes
        result.validation_results = [
            ValidationResult(is_valid=passed, error_message=None if passed else "Failed to retrieve needle")
        ]
        
        results_obj.levels.append(
            NIAHResultLevel(context_length=length, load_result=result, passed=passed)
        )
        
        status = "PASSED" if passed else "FAILED"
        print(f"       Status: {status}")
        summary = result.tracker.summary()
        print(f"       TTFT:   {summary['average_ttft_ms']}ms")
        print(f"       TPOT:   {summary.get('average_tpot_ms', 0.0)}ms")
        print(f"       Latency:{summary['average_latency_ms']}ms")

    return results_obj
