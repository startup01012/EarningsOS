from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import sqrt

from .base import ForecastAdapter
from .price_series import PriceObservation, build_forecast_request


@dataclass(frozen=True)
class BacktestCase:
    cutoff_timestamp: datetime
    actual: list[float]
    predicted: list[float]


@dataclass(frozen=True)
class BacktestMetrics:
    cases: int
    horizons: int
    mae: float
    rmse: float
    mape: float | None
    directional_accuracy: float | None


def evaluate_cases(cases: list[BacktestCase], *, baseline_closes: list[float]) -> BacktestMetrics:
    if not cases:
        raise ValueError("at least one backtest case is required")
    if len(baseline_closes) != len(cases):
        raise ValueError("baseline_closes must have one value per case")

    errors: list[float] = []
    squared_errors: list[float] = []
    percentage_errors: list[float] = []
    direction_hits = 0
    direction_total = 0
    horizons = len(cases[0].actual)

    for case, baseline in zip(cases, baseline_closes):
        if len(case.actual) != len(case.predicted):
            raise ValueError("actual and predicted lengths must match")
        if len(case.actual) != horizons:
            raise ValueError("all backtest cases must use the same horizon")
        for actual, predicted in zip(case.actual, case.predicted):
            error = predicted - actual
            errors.append(abs(error))
            squared_errors.append(error * error)
            if actual != 0:
                percentage_errors.append(abs(error) / abs(actual))
            direction_hits += int((predicted > baseline) == (actual > baseline))
            direction_total += 1

    return BacktestMetrics(
        cases=len(cases),
        horizons=horizons,
        mae=sum(errors) / len(errors),
        rmse=sqrt(sum(squared_errors) / len(squared_errors)),
        mape=(sum(percentage_errors) / len(percentage_errors) * 100) if percentage_errors else None,
        directional_accuracy=(direction_hits / direction_total) if direction_total else None,
    )


def run_backtest(
    adapter: ForecastAdapter,
    symbol: str,
    observations: list[PriceObservation],
    *,
    context_length: int = 256,
    horizon: int = 5,
    stride: int = 5,
    max_cases: int | None = 5,
) -> tuple[list[BacktestCase], BacktestMetrics]:
    """Run a strictly causal rolling-origin backtest.

    Each forecast only sees observations at or before its cutoff. The subsequent
    ``horizon`` observations are held out as ground truth. ``stride`` controls
    how far the rolling origin advances between cases.
    """
    if context_length < 2:
        raise ValueError("context_length must be >= 2")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be >= 1 when provided")

    ordered = sorted(observations, key=lambda item: item.timestamp)
    minimum = context_length + horizon
    if len(ordered) < minimum:
        raise ValueError(
            f"at least {minimum} observations are required, got {len(ordered)}"
        )

    cases: list[BacktestCase] = []
    baselines: list[float] = []
    cutoff_end = context_length

    while cutoff_end + horizon <= len(ordered):
        context = ordered[cutoff_end - context_length : cutoff_end]
        future = ordered[cutoff_end : cutoff_end + horizon]
        request = build_forecast_request(symbol, context, horizon)
        result = adapter.forecast(request)
        cases.append(
            BacktestCase(
                cutoff_timestamp=context[-1].timestamp,
                actual=[item.close for item in future],
                predicted=result.median,
            )
        )
        baselines.append(context[-1].close)

        if max_cases is not None and len(cases) >= max_cases:
            break
        cutoff_end += stride

    return cases, evaluate_cases(cases, baseline_closes=baselines)
