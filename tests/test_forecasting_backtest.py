from datetime import datetime, timedelta, timezone

import pytest

from services.forecasting.backtest import BacktestCase, evaluate_cases, run_backtest
from services.forecasting.base import ForecastAdapter, ForecastResult
from services.forecasting.price_series import PriceObservation


class FixedAdapter(ForecastAdapter):
    model_id = "test/fixed"

    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.requests = []

    def forecast(self, request):
        self.requests.append(request)
        return ForecastResult(
            model_id=self.model_id,
            symbol=request.symbol,
            generated_at=datetime.now(timezone.utc),
            horizon=request.horizon,
            median=self.values[: request.horizon],
        )


def test_evaluate_cases_metrics():
    cases = [
        BacktestCase(datetime(2026, 1, 1, tzinfo=timezone.utc), [101, 99], [100, 100]),
    ]
    metrics = evaluate_cases(cases, baseline_closes=[100])
    assert metrics.cases == 1
    assert metrics.horizons == 2
    assert metrics.mae == pytest.approx(1.0)
    assert metrics.rmse == pytest.approx(1.0)
    assert metrics.mape == pytest.approx((1 / 101 + 1 / 99) / 2 * 100)
    assert metrics.directional_accuracy == pytest.approx(0.5)


def test_run_backtest_never_passes_future_observations_to_adapter():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    observations = [
        PriceObservation(start + timedelta(days=i), float(100 + i))
        for i in range(10)
    ]
    adapter = FixedAdapter([110, 111])

    cases, metrics = run_backtest(
        adapter,
        "TEST",
        observations,
        context_length=5,
        horizon=2,
        stride=2,
        max_cases=2,
    )

    assert len(cases) == 2
    assert metrics.cases == 2
    assert [len(request.values) for request in adapter.requests] == [5, 5]
    assert adapter.requests[0].timestamps[-1] == observations[4].timestamp
    assert adapter.requests[1].timestamps[-1] == observations[6].timestamp
    assert cases[0].actual == [105.0, 106.0]
    assert cases[1].actual == [107.0, 108.0]


def test_run_backtest_requires_enough_observations():
    observations = [
        PriceObservation(datetime(2026, 1, 1, tzinfo=timezone.utc), 100.0),
        PriceObservation(datetime(2026, 1, 2, tzinfo=timezone.utc), 101.0),
    ]
    with pytest.raises(ValueError, match="at least 7 observations"):
        run_backtest(
            FixedAdapter([1, 2]),
            "TEST",
            observations,
            context_length=5,
            horizon=2,
        )
