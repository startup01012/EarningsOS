from datetime import datetime, timedelta, timezone

import pytest

from services.forecasting.backtest import BacktestCase, REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS_HIGH_VOLATILITY, classify_regime, evaluate_cases, run_backtest
from services.forecasting.base import ForecastAdapter, ForecastResult
from services.forecasting.benchmarks import baseline_predictions, bootstrap_mean_ci, drift_predictions, evaluate_predictions, evaluate_return_predictions, moving_average_predictions, naive_last_close_predictions, previous_return_predictions
from services.forecasting.price_series import PriceObservation


class FixedAdapter(ForecastAdapter):
    model_id = "test/fixed"

    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.requests = []

    def forecast(self, request):
        self.requests.append(request)
        return ForecastResult(model_id=self.model_id, symbol=request.symbol, generated_at=datetime.now(timezone.utc), horizon=request.horizon, median=self.values[: request.horizon])


def test_evaluate_cases_metrics():
    cases = [BacktestCase(datetime(2026, 1, 1, tzinfo=timezone.utc), [101, 99], [100, 100], 100)]
    metrics = evaluate_cases(cases)
    assert metrics.cases == 1
    assert metrics.horizons == 2
    assert metrics.mae == pytest.approx(1.0)
    assert metrics.rmse == pytest.approx(1.0)
    assert metrics.mape == pytest.approx((1 / 101 + 1 / 99) / 2 * 100)
    assert metrics.smape == pytest.approx((2 / 201 + 2 / 199) / 2 * 100)
    assert metrics.directional_accuracy == pytest.approx(0.0)


def test_naive_last_close_predictions_use_cutoff_close():
    cases = [BacktestCase(datetime(2026, 1, 1, tzinfo=timezone.utc), [101, 99], [100, 100], 100)]
    predictions = naive_last_close_predictions(cases)
    assert predictions == [[100, 100]]
    metrics = evaluate_predictions(cases, predictions)
    assert metrics.mae == pytest.approx(1.0)


def test_return_metrics_use_selected_horizon_and_direction():
    cases = [BacktestCase(datetime(2026, 1, 1, tzinfo=timezone.utc), [102, 110], [101, 105], 100), BacktestCase(datetime(2026, 1, 2, tzinfo=timezone.utc), [98, 90], [99, 95], 100)]
    metrics = evaluate_return_predictions(cases, [case.predicted for case in cases], horizon=2)
    assert metrics.cases == 2
    assert metrics.horizon == 2
    assert metrics.return_mae == pytest.approx(0.05)
    assert metrics.return_rmse == pytest.approx(0.05)
    assert metrics.directional_accuracy == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.information_coefficient == pytest.approx(1.0)


def test_return_metrics_reject_invalid_horizon():
    cases = [BacktestCase(datetime(2026, 1, 1, tzinfo=timezone.utc), [101, 102], [100, 100], 100)]
    with pytest.raises(ValueError, match="horizon must be between"):
        evaluate_return_predictions(cases, [case.predicted for case in cases], horizon=3)


def test_run_backtest_never_passes_future_observations_to_adapter():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    observations = [PriceObservation(start + timedelta(days=i), float(100 + i)) for i in range(10)]
    adapter = FixedAdapter([110, 111])
    cases, metrics = run_backtest(adapter, "TEST", observations, context_length=5, horizon=2, stride=2, max_cases=2)
    assert len(cases) == 2
    assert metrics.cases == 2
    assert [len(request.values) for request in adapter.requests] == [5, 5]
    assert adapter.requests[0].timestamps[-1] == observations[4].timestamp
    assert adapter.requests[1].timestamps[-1] == observations[6].timestamp
    assert cases[0].cutoff_close == 104.0
    assert cases[0].actual == [105.0, 106.0]
    assert cases[0].context_closes == [100.0, 101.0, 102.0, 103.0, 104.0]
    assert cases[0].regime == REGIME_SIDEWAYS_HIGH_VOLATILITY
    assert cases[1].cutoff_close == 106.0
    assert cases[1].actual == [107.0, 108.0]


def test_run_backtest_selects_historical_window():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    observations = [PriceObservation(start + timedelta(days=i), float(100 + i)) for i in range(16)]
    adapter = FixedAdapter([110, 111])
    cases, metrics = run_backtest(adapter, "TEST", observations, context_length=5, horizon=2, stride=2, start_case=2, max_cases=2)
    assert len(cases) == 2
    assert metrics.cases == 2
    assert adapter.requests[0].timestamps[-1] == observations[8].timestamp
    assert adapter.requests[1].timestamps[-1] == observations[10].timestamp
    assert cases[0].cutoff_close == 108.0
    assert cases[0].actual == [109.0, 110.0]
    assert cases[1].cutoff_close == 110.0
    assert cases[1].actual == [111.0, 112.0]


def test_run_backtest_rejects_invalid_start_case():
    observations = [PriceObservation(datetime(2026, 1, 1, tzinfo=timezone.utc), 100.0), PriceObservation(datetime(2026, 1, 2, tzinfo=timezone.utc), 101.0)]
    with pytest.raises(ValueError, match="start_case must be >= 0"):
        run_backtest(FixedAdapter([1, 2]), "TEST", observations, context_length=2, horizon=1, start_case=-1)


def test_run_backtest_requires_enough_observations():
    observations = [PriceObservation(datetime(2026, 1, 1, tzinfo=timezone.utc), 100.0), PriceObservation(datetime(2026, 1, 2, tzinfo=timezone.utc), 101.0)]
    with pytest.raises(ValueError, match="at least 7 observations"):
        run_backtest(FixedAdapter([1, 2]), "TEST", observations, context_length=5, horizon=2)


def test_regime_classifier_uses_context_only():
    bull = [100.0 + i * 0.8 for i in range(40)]
    bear = [140.0 - i * 0.8 for i in range(40)]
    sideways = [100.0, 101.0, 100.5, 100.8, 100.2] * 8
    assert classify_regime(bull) == REGIME_BULL
    assert classify_regime(bear) == REGIME_BEAR
    assert classify_regime(sideways) == REGIME_SIDEWAYS_HIGH_VOLATILITY


def test_stronger_baselines_are_causal():
    case = BacktestCase(datetime(2026, 1, 1, tzinfo=timezone.utc), [110, 111, 112], [0, 0, 0], 100, context_closes=[90, 95, 100])
    previous = previous_return_predictions([case])
    drift = drift_predictions([case])
    moving_average = moving_average_predictions([case], window=2)
    all_baselines = baseline_predictions([case])
    last_return = 100 / 95 - 1
    assert previous[0] == pytest.approx([100 * (1 + last_return), 100 * (1 + last_return) ** 2, 100 * (1 + last_return) ** 3])
    assert drift[0] == pytest.approx([105.0, 110.0, 115.0])
    assert moving_average[0] == pytest.approx([97.5, 97.5, 97.5])
    assert set(all_baselines) == {"last_close", "previous_return", "drift", "moving_average_20"}


def test_bootstrap_mean_ci_is_deterministic_and_reports_variability():
    first = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], seed=7, resamples=500)
    second = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], seed=7, resamples=500)
    assert first == second
    assert first.n == 4
    assert first.mean == pytest.approx(2.5)
    assert first.std is not None and first.std > 0
    assert first.ci_low is not None and first.ci_high is not None
    assert first.ci_low <= first.mean <= first.ci_high
