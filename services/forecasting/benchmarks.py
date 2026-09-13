from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import random

from .backtest import BacktestCase


BASELINE_LAST_CLOSE = "last_close"
BASELINE_PREVIOUS_RETURN = "previous_return"
BASELINE_DRIFT = "drift"
BASELINE_MOVING_AVERAGE_20 = "moving_average_20"


@dataclass(frozen=True)
class BenchmarkMetrics:
    cases: int
    horizons: int
    mae: float
    rmse: float
    mape: float | None
    smape: float | None


@dataclass(frozen=True)
class ReturnBenchmarkMetrics:
    cases: int
    horizon: int
    return_mae: float
    return_rmse: float
    directional_accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    information_coefficient: float | None


@dataclass(frozen=True)
class BootstrapSummary:
    n: int
    mean: float
    std: float | None
    ci_low: float | None
    ci_high: float | None


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


def _pearson(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) < 2 or len(values_x) != len(values_y):
        return None
    mean_x = sum(values_x) / len(values_x)
    mean_y = sum(values_y) / len(values_y)
    centered_x = [value - mean_x for value in values_x]
    centered_y = [value - mean_y for value in values_y]
    numerator = sum(x * y for x, y in zip(centered_x, centered_y))
    denominator = sqrt(sum(x * x for x in centered_x) * sum(y * y for y in centered_y))
    return numerator / denominator if denominator else None


def evaluate_return_predictions(
    cases: list[BacktestCase],
    predictions: list[list[float]],
    horizon: int | None = None,
) -> ReturnBenchmarkMetrics:
    """Evaluate final-horizon returns and up/down classification from causal forecasts."""
    if not cases:
        raise ValueError("at least one backtest case is required")
    if len(predictions) != len(cases):
        raise ValueError("predictions must have one row per case")

    case_horizon = len(cases[0].actual)
    selected_horizon = case_horizon if horizon is None else horizon
    if selected_horizon < 1 or selected_horizon > case_horizon:
        raise ValueError("horizon must be between 1 and the case horizon")

    actual_returns: list[float] = []
    predicted_returns: list[float] = []
    actual_directions: list[int] = []
    predicted_directions: list[int] = []

    for case, predicted in zip(cases, predictions):
        if len(case.actual) != len(predicted):
            raise ValueError("actual and predicted lengths must match")
        if len(case.actual) != case_horizon:
            raise ValueError("all backtest cases must use the same horizon")
        if case.cutoff_close == 0:
            raise ValueError("cutoff_close must be non-zero for return evaluation")

        actual_return = case.actual[selected_horizon - 1] / case.cutoff_close - 1.0
        predicted_return = predicted[selected_horizon - 1] / case.cutoff_close - 1.0
        actual_returns.append(actual_return)
        predicted_returns.append(predicted_return)
        actual_directions.append(1 if actual_return > 0 else -1 if actual_return < 0 else 0)
        predicted_directions.append(1 if predicted_return > 0 else -1 if predicted_return < 0 else 0)

    return_errors = [abs(a - p) for a, p in zip(actual_returns, predicted_returns)]
    squared_return_errors = [(a - p) ** 2 for a, p in zip(actual_returns, predicted_returns)]

    classified = [
        (actual, predicted)
        for actual, predicted in zip(actual_directions, predicted_directions)
        if actual != 0
    ]
    if classified:
        directional_accuracy = sum(actual == predicted for actual, predicted in classified) / len(classified)
        true_positive = sum(actual == 1 and predicted == 1 for actual, predicted in classified)
        false_positive = sum(actual == -1 and predicted == 1 for actual, predicted in classified)
        false_negative = sum(actual == 1 and predicted == -1 for actual, predicted in classified)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    else:
        directional_accuracy = precision = recall = f1 = None

    return ReturnBenchmarkMetrics(
        cases=len(cases),
        horizon=selected_horizon,
        return_mae=sum(return_errors) / len(return_errors),
        return_rmse=sqrt(sum(squared_return_errors) / len(squared_return_errors)),
        directional_accuracy=directional_accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        information_coefficient=_pearson(actual_returns, predicted_returns),
    )


def _context_or_fail(case: BacktestCase) -> list[float]:
    if not case.context_closes:
        raise ValueError("backtest case is missing context_closes required by this baseline")
    if len(case.context_closes) < 2:
        raise ValueError("at least two context closes are required for this baseline")
    return case.context_closes


def naive_last_close_predictions(cases: list[BacktestCase]) -> list[list[float]]:
    """Forecast every future point at the cutoff close (last-value baseline)."""
    return [[case.cutoff_close for _ in case.actual] for case in cases]


def previous_return_predictions(cases: list[BacktestCase]) -> list[list[float]]:
    """Repeat the most recent one-session return and compound it forward."""
    predictions: list[list[float]] = []
    for case in cases:
        context = _context_or_fail(case)
        previous = context[-2]
        last = context[-1]
        if previous == 0:
            predictions.append([last for _ in case.actual])
            continue
        last_return = last / previous - 1.0
        row = [last * ((1.0 + last_return) ** step) for step in range(1, len(case.actual) + 1)]
        predictions.append(row)
    return predictions


def drift_predictions(cases: list[BacktestCase]) -> list[list[float]]:
    """Project the historical average absolute price change forward."""
    predictions: list[list[float]] = []
    for case in cases:
        context = _context_or_fail(case)
        increments = [context[index] - context[index - 1] for index in range(1, len(context))]
        drift = sum(increments) / len(increments)
        predictions.append([context[-1] + drift * step for step in range(1, len(case.actual) + 1)])
    return predictions


def moving_average_predictions(cases: list[BacktestCase], window: int = 20) -> list[list[float]]:
    """Use the trailing simple moving average as a flat future-price baseline."""
    if window < 1:
        raise ValueError("moving-average window must be >= 1")
    predictions: list[list[float]] = []
    for case in cases:
        context = _context_or_fail(case)
        effective_window = min(window, len(context))
        average = sum(context[-effective_window:]) / effective_window
        predictions.append([average for _ in case.actual])
    return predictions


def baseline_predictions(cases: list[BacktestCase]) -> dict[str, list[list[float]]]:
    return {
        BASELINE_LAST_CLOSE: naive_last_close_predictions(cases),
        BASELINE_PREVIOUS_RETURN: previous_return_predictions(cases),
        BASELINE_DRIFT: drift_predictions(cases),
        BASELINE_MOVING_AVERAGE_20: moving_average_predictions(cases, window=20),
    }


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int = 20260913,
    resamples: int = 2000,
) -> BootstrapSummary:
    """Return mean, sample std and deterministic percentile bootstrap 95% CI."""
    if not values:
        return BootstrapSummary(0, float("nan"), None, None, None)
    if len(values) == 1:
        return BootstrapSummary(1, values[0], 0.0, values[0], values[0])
    if resamples < 100:
        raise ValueError("resamples must be >= 100")

    mean = sum(values) / len(values)
    sample_std = sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in values:
            total += values[rng.randrange(len(values))]
        bootstrap_means.append(total / len(values))
    bootstrap_means.sort()
    low_index = int(0.025 * (resamples - 1))
    high_index = int(0.975 * (resamples - 1))
    return BootstrapSummary(
        n=len(values),
        mean=mean,
        std=sample_std,
        ci_low=bootstrap_means[low_index],
        ci_high=bootstrap_means[high_index],
    )
