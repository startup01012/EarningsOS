from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
import random

from .backtest import BacktestCase


@dataclass(frozen=True)
class PairedComparison:
    model: str
    baseline: str
    horizon: int
    regime: str
    cases: int
    model_loss_mean: float
    baseline_loss_mean: float
    mean_loss_diff: float
    median_loss_diff: float
    std_loss_diff: float | None
    ci_low: float | None
    ci_high: float | None
    model_win_rate: float
    tie_rate: float
    baseline_win_rate: float
    dm_stat: float | None
    dm_pvalue: float | None
    model_directional_accuracy: float | None
    baseline_directional_accuracy: float | None
    directional_accuracy_diff: float | None


@dataclass(frozen=True)
class PairedCaseResult:
    symbol: str
    case_index: int
    model: str
    baseline: str
    cutoff_timestamp: object
    horizon: int
    regime: str
    model_loss: float
    baseline_loss: float
    loss_diff: float
    model_direction_correct: int | None
    baseline_direction_correct: int | None


def _validate(cases: list[BacktestCase], predictions: list[list[float]], name: str) -> None:
    if len(cases) != len(predictions):
        raise ValueError(f"{name} predictions must have one row per case")
    if not cases:
        raise ValueError("at least one backtest case is required")
    case_horizon = len(cases[0].actual)
    for case, prediction in zip(cases, predictions):
        if len(case.actual) != case_horizon or len(prediction) != case_horizon:
            raise ValueError("all paired cases and predictions must use the same horizon")
        if case.cutoff_close == 0:
            raise ValueError("cutoff_close must be non-zero for paired return comparison")


def _return_loss(case: BacktestCase, prediction: list[float], horizon: int) -> float:
    actual_return = case.actual[horizon - 1] / case.cutoff_close - 1.0
    predicted_return = prediction[horizon - 1] / case.cutoff_close - 1.0
    return abs(actual_return - predicted_return)


def _direction_correct(case: BacktestCase, prediction: list[float], horizon: int) -> int | None:
    actual_return = case.actual[horizon - 1] / case.cutoff_close - 1.0
    predicted_return = prediction[horizon - 1] / case.cutoff_close - 1.0
    if actual_return == 0:
        return None
    return int((actual_return > 0) == (predicted_return > 0))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _bootstrap_mean_ci(values: list[float], *, seed: int, resamples: int = 4000) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    if len(values) == 1:
        return values[0], values[0], 0.0
    mean = sum(values) / len(values)
    std = sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    rng = random.Random(seed)
    means = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(resamples)]
    means.sort()
    return means[int(0.025 * (resamples - 1))], means[int(0.975 * (resamples - 1))], std


def _normal_two_sided_pvalue(z: float) -> float:
    return 1.0 - erf(abs(z) / sqrt(2.0))


def dm_style_hac(loss_diffs: list[float], max_lag: int) -> tuple[float | None, float | None]:
    """DM-style normal statistic using Newey-West HAC variance."""
    n = len(loss_diffs)
    if n < 3:
        return None, None
    mean = sum(loss_diffs) / n
    centered = [value - mean for value in loss_diffs]
    gamma0 = sum(value * value for value in centered) / n
    lag = min(max(0, max_lag), n - 1)
    long_run = gamma0
    for k in range(1, lag + 1):
        gamma = sum(centered[t] * centered[t - k] for t in range(k, n)) / n
        long_run += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    if long_run <= 0:
        return None, None
    statistic = mean / sqrt(long_run / n)
    return statistic, _normal_two_sided_pvalue(statistic)


def compare_paired(
    cases: list[BacktestCase],
    model_predictions: list[list[float]],
    baseline_predictions: list[list[float]],
    *,
    model: str,
    baseline: str,
    horizon: int,
    regime: str = "all",
    symbol: str = "",
    dependence_lag: int | None = None,
) -> tuple[PairedComparison, list[PairedCaseResult]]:
    """Compare model and baseline on identical causal cutoffs."""
    _validate(cases, model_predictions, "model")
    _validate(cases, baseline_predictions, "baseline")
    if horizon < 1 or horizon > len(cases[0].actual):
        raise ValueError("horizon must be between 1 and the case horizon")

    selected = [
        (index, case, model_pred, baseline_pred)
        for index, (case, model_pred, baseline_pred) in enumerate(zip(cases, model_predictions, baseline_predictions))
        if regime == "all" or case.regime == regime
    ]
    if not selected:
        raise ValueError("no cases match the requested regime")

    model_losses: list[float] = []
    baseline_losses: list[float] = []
    diffs: list[float] = []
    model_hits: list[int] = []
    baseline_hits: list[int] = []
    paired_rows: list[PairedCaseResult] = []

    for index, case, model_pred, baseline_pred in selected:
        model_loss = _return_loss(case, model_pred, horizon)
        baseline_loss = _return_loss(case, baseline_pred, horizon)
        diff = model_loss - baseline_loss
        model_correct = _direction_correct(case, model_pred, horizon)
        baseline_correct = _direction_correct(case, baseline_pred, horizon)
        model_losses.append(model_loss)
        baseline_losses.append(baseline_loss)
        diffs.append(diff)
        if model_correct is not None:
            model_hits.append(model_correct)
        if baseline_correct is not None:
            baseline_hits.append(baseline_correct)
        paired_rows.append(PairedCaseResult(symbol, index, model, baseline, case.cutoff_timestamp, horizon, case.regime, model_loss, baseline_loss, diff, model_correct, baseline_correct))

    ci_low, ci_high, std = _bootstrap_mean_ci(diffs, seed=20260913 + horizon + len(model) * 17 + len(baseline))
    wins = sum(diff < 0 for diff in diffs)
    ties = sum(diff == 0 for diff in diffs)
    n = len(diffs)
    lag = dependence_lag if dependence_lag is not None else max(0, horizon - 1)
    dm_stat, dm_pvalue = dm_style_hac(diffs, lag)
    model_da = sum(model_hits) / len(model_hits) if model_hits else None
    baseline_da = sum(baseline_hits) / len(baseline_hits) if baseline_hits else None

    return (
        PairedComparison(
            model, baseline, horizon, regime, n,
            sum(model_losses) / n, sum(baseline_losses) / n,
            sum(diffs) / n, _median(diffs), std, ci_low, ci_high,
            wins / n, ties / n, (n - wins - ties) / n,
            dm_stat, dm_pvalue, model_da, baseline_da,
            (model_da - baseline_da) if model_da is not None and baseline_da is not None else None,
        ),
        paired_rows,
    )


def dedupe_paired_cases(rows: list[PairedCaseResult], *, include_regime: bool = True) -> list[PairedCaseResult]:
    """Deduplicate persisted paired rows created by overlapping report views.

    The canonical identity is symbol + cutoff + model + baseline + horizon.
    Regime is metadata of that cutoff, not a second forecast observation.
    """
    seen: set[tuple[object, ...]] = set()
    result: list[PairedCaseResult] = []
    for row in rows:
        key = (row.symbol, row.cutoff_timestamp, row.model, row.baseline, row.horizon)
        if include_regime:
            key = (*key, row.regime)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def strongest_baseline(summaries: list[PairedComparison], *, horizon: int) -> PairedComparison | None:
    """Return the baseline with the lowest mean return MAE at a horizon.

    This is the only baseline that may determine the production gate. Other
    baselines remain diagnostic and are never substituted merely because a
    model happens to beat them.
    """
    candidates = [item for item in summaries if item.regime == "all" and item.horizon == horizon]
    return min(candidates, key=lambda item: item.baseline_loss_mean) if candidates else None
