from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .backtest import BacktestCase


@dataclass(frozen=True)
class BenchmarkMetrics:
    cases: int
    horizons: int
    mae: float
    rmse: float
    mape: float | None
    smape: float | None


def evaluate_predictions(cases: list[BacktestCase], predictions: list[list[float]]) -> BenchmarkMetrics:
    if not cases:
        raise ValueError("at least one backtest case is required")
    if len(predictions) != len(cases):
        raise ValueError("predictions must have one row per case")

    errors: list[float] = []
    squared_errors: list[float] = []
    percentage_errors: list[float] = []
    smape_errors: list[float] = []
    horizons = len(cases[0].actual)

    for case, predicted in zip(cases, predictions):
        if len(case.actual) != len(predicted):
            raise ValueError("actual and predicted lengths must match")
        if len(case.actual) != horizons:
            raise ValueError("all backtest cases must use the same horizon")
        for actual, forecast in zip(case.actual, predicted):
            error = forecast - actual
            errors.append(abs(error))
            squared_errors.append(error * error)
            if actual != 0:
                percentage_errors.append(abs(error) / abs(actual))
            denominator = abs(actual) + abs(forecast)
            if denominator != 0:
                smape_errors.append(2 * abs(error) / denominator)

    return BenchmarkMetrics(
        cases=len(cases),
        horizons=horizons,
        mae=sum(errors) / len(errors),
        rmse=sqrt(sum(squared_errors) / len(squared_errors)),
        mape=(sum(percentage_errors) / len(percentage_errors) * 100) if percentage_errors else None,
        smape=(sum(smape_errors) / len(smape_errors) * 100) if smape_errors else None,
    )


def naive_last_close_predictions(cases: list[BacktestCase]) -> list[list[float]]:
    """Forecast every future point at the cutoff close (last-value baseline)."""
    return [[case.actual[0] * 0 + case.predicted[0] * 0 + 0.0 for _ in case.actual] for case in cases]
