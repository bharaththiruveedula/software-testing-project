"""
Unit tests for Consistency Testing Module.
"""

import pytest
from consistency import run_consistency_test


@pytest.mark.asyncio
async def test_run_consistency_test(mocker):
    """Test consistency test successfully computes variance and CV."""
    
    mock_result = mocker.MagicMock()
    mock_result.tracker.results = [mocker.MagicMock(success=True, latency_ms=100.0)]
    mock_result.validation_results = [mocker.MagicMock(is_valid=True)]
    
    mocker.patch("consistency.run_load_test", return_value=mock_result)

    result = await run_consistency_test(
        url="http://fake",
        iterations=3,
        prompt="test",
    )

    assert result.iterations == 3
    assert result.pass_rate == 1.0
    assert result.avg_latency == 100.0
    assert result.latency_variance == 0.0
    assert result.latency_cv == 0.0
    assert result.is_consistent is True


@pytest.mark.asyncio
async def test_run_consistency_test_variance(mocker):
    """Test variance and CV computation with different latencies."""
    
    mock_res_1 = mocker.MagicMock()
    mock_res_1.tracker.results = [mocker.MagicMock(success=True, latency_ms=100.0)]
    mock_res_1.validation_results = [mocker.MagicMock(is_valid=True)]

    mock_res_2 = mocker.MagicMock()
    mock_res_2.tracker.results = [mocker.MagicMock(success=True, latency_ms=200.0)]
    mock_res_2.validation_results = [mocker.MagicMock(is_valid=True)]

    mocker.patch("consistency.run_load_test", side_effect=[mock_res_1, mock_res_2])

    result = await run_consistency_test(
        url="http://fake",
        iterations=2,
        prompt="test",
    )

    assert result.iterations == 2
    assert result.avg_latency == 150.0
    # Variance of [100, 200] is 5000: ((100-150)^2 + (200-150)^2) / (2-1)
    assert result.latency_variance == 5000.0
    # CV is sqrt(5000) / 150 ~= 70.7 / 150 ~= 0.471
    assert result.latency_cv == pytest.approx(0.4714, rel=1e-3)
