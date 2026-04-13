"""
Comparative Testing Module

Analyzes multiple test results and generates comparative reports.
"""

from dataclasses import dataclass
from typing import List, Tuple
from load_runner import LoadTestResult
from config_loader import TestScenario


@dataclass
class ComparisonMetric:
    scenario_name: str
    target: str
    users: int
    throughput: float
    avg_latency: float
    pass_rate: float
    passed: bool


def analyze_results(results: List[Tuple[TestScenario, LoadTestResult]]) -> List[ComparisonMetric]:
    """
    Extract key comparative metrics from a list of scenario results.
    """
    metrics = []
    
    for scenario, result in results:
        tracker = result.tracker
        
        valid_count = sum(1 for v in result.validation_results if v.is_valid)
        total_validated = len(result.validation_results)
        pass_rate = valid_count / total_validated if total_validated > 0 else 0.0
        
        passed = pass_rate >= scenario.threshold and tracker.successful_requests > 0
        
        metrics.append(
            ComparisonMetric(
                scenario_name=scenario.name,
                target=scenario.model, # Using model as target identity
                users=scenario.users,
                throughput=tracker.tokens_per_second(),
                avg_latency=tracker.average_latency(),
                pass_rate=pass_rate,
                passed=passed,
            )
        )
        
    return metrics
