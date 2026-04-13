"""
Unit tests for Ramp-Up Testing Module.
"""

import pytest
from ramp_runner import run_ramp_test


@pytest.mark.asyncio
async def test_ramp_test_success(mocker):
    """Test that ramp test runs through all levels and reports success."""
    
    # Mock run_load_test to always return a successful result
    mock_result = mocker.MagicMock()
    mock_result.tracker.successful_requests = 10
    mock_result.tracker.average_latency.return_value = 150.0
    mock_result.validation_results = [mocker.MagicMock() for _ in range(10)]
    for v in mock_result.validation_results:
        v.is_valid = True

    mocker.patch("ramp_runner.run_load_test", return_value=mock_result)

    result = await run_ramp_test(
        url="http://fake",
        levels=[1, 2, 3],
        prompt="test",
        schema=None,
    )

    assert result.breaking_point is None
    assert len(result.results) == 3
    assert result.levels == [1, 2, 3]


@pytest.mark.asyncio
async def test_ramp_test_breaking_point(mocker):
    """Test that ramp test stops at breaking point if failure threshold reached."""
    
    # Mock to succeed on first call, fail on second
    mock_success = mocker.MagicMock()
    mock_success.tracker.successful_requests = 10
    mock_success.tracker.average_latency.return_value = 100.0
    mock_success.validation_results = [mocker.MagicMock(is_valid=True) for _ in range(10)]

    mock_fail = mocker.MagicMock()
    mock_fail.tracker.successful_requests = 10
    mock_fail.tracker.average_latency.return_value = 5000.0
    # 5/10 valid = 50% pass rate, which is below 95% threshold
    mock_fail.validation_results = [mocker.MagicMock(is_valid=i<5) for i in range(10)]

    mocker.patch("ramp_runner.run_load_test", side_effect=[mock_success, mock_fail])

    result = await run_ramp_test(
        url="http://fake",
        levels=[1, 5, 10, 20],
        prompt="test",
        schema=None,
        pass_rate_threshold=0.95,
    )

    # Breaking point should be at level '5'
    assert result.breaking_point == 5
    # Should have stopped after the second level
    assert len(result.results) == 2
    assert result.levels == [1, 5]
