"""
Unit tests for the Comparative Testing Module.
"""

from comparator import analyze_results, ComparisonMetric
from config_loader import TestScenario
from load_runner import LoadTestResult
from metrics import PerformanceTracker
from validator import ValidationResult


def test_analyze_results():
    """Test generating comparison metrics from scenario results."""
    
    # Scenario 1 (Passing)
    s1 = TestScenario(name="Test 1", url="http://", model="gpt-3", users=5, threshold=0.9)
    t1 = PerformanceTracker()
    t1.record(1, 100.0, 20.0, None, True, prompt_tokens=10, completion_tokens=100) # 1 sec throughput
    r1 = LoadTestResult(tracker=t1, validation_results=[ValidationResult(is_valid=True)], raw_responses=[])
    
    # Scenario 2 (Failing due to pass rate)
    s2 = TestScenario(name="Test 2", url="http://", model="gpt-4", users=5, threshold=0.9)
    t2 = PerformanceTracker()
    t2.record(1, 200.0, 30.0, None, True, prompt_tokens=10, completion_tokens=50) # 0.25 sec throughput
    r2 = LoadTestResult(
        tracker=t2,
        validation_results=[ValidationResult(is_valid=False, error_message="err")],
        raw_responses=[]
    )
    
    metrics = analyze_results([(s1, r1), (s2, r2)])
    
    assert len(metrics) == 2
    
    # Check Scenario 1
    m1 = metrics[0]
    assert m1.scenario_name == "Test 1"
    assert m1.target == "gpt-3"
    assert m1.users == 5
    assert m1.throughput == 1000.0 # 100 tokens / 0.1s
    assert m1.avg_latency == 100.0
    assert m1.pass_rate == 1.0
    assert m1.passed is True
    
    # Check Scenario 2
    m2 = metrics[1]
    assert m2.scenario_name == "Test 2"
    assert m2.target == "gpt-4"
    assert m2.pass_rate == 0.0
    assert m2.passed is False
    assert m2.avg_latency == 200.0
    assert m2.throughput == 250.0 # 50 tokens / 0.2s
