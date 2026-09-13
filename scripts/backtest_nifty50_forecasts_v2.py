from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from apps.api.db.session import SessionLocal
from services.forecasting.backtest import (
    BacktestCase,
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_SIDEWAYS_HIGH_VOLATILITY,
    classify_regime,
    evaluate_cases,
    run_backtest,
)
from services.forecasting.benchmarks import (
    BASELINE_DRIFT,
    BASELINE_LAST_CLOSE,
    BASELINE_MOVING_AVERAGE_20,
    BASELINE_PREVIOUS_RETURN,
    baseline_predictions,
    bootstrap_mean_ci,
    evaluate_predictions,
    evaluate_return_predictions,
)
from services.forecasting.base import ForecastRequest
from services.forecasting.chronos import Chronos2Adapter
from services.forecasting.kronos import KronosSmallAdapter
from services.forecasting.kronos_loader import load_ohlcv_observations
from services.forecasting.price_loader import load_close_observations


DEFAULT_HORIZONS = (1, 5, 10)
DEFAULT_REGIMES = ("all", REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS_HIGH_VOLATILITY)
DEFAULT_MODELS = ("chronos", "kronos")


@dataclass(frozen=True)
class EvaluationRow:
    symbol: str
    model: str
    horizon: int
    regime: str
    cases: int
    price_mae: float
    price_rmse: float
    return_mae: float
    return_rmse: float
    directional_accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    information_coefficient: float | None


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not values or any(item < 1 for item in values):
        raise ValueError("horizons must contain positive integers")
    return values


def _parse_symbols(value: str) -> list[str]:
    symbols = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required")
    return list(dict.fromkeys(symbols))


def _load_nifty50_symbols(path: str) -> list[str]:
    frame = pd.read_csv(path)
    if "Symbol" not in frame.columns:
        raise ValueError(f"{path} must contain a Symbol column")
    symbols = [str(value).strip().upper() for value in frame["Symbol"].dropna() if str(value).strip()]
    symbols = list(dict.fromkeys(symbols))
    if len(symbols) != 50:
        raise ValueError(f"expected exactly 50 NIFTY 50 symbols in {path}, found {len(symbols)}")
    return symbols


def _subset_by_regime(cases: list[BacktestCase], regime: str) -> list[BacktestCase]:
    if regime == "all":
        return cases
    return [case for case in cases if case.regime == regime]


def _rows_for_predictions(
    symbol: str,
    model: str,
    cases: list[BacktestCase],
    predictions: list[list[float]],
    horizons: tuple[int, ...],
) -> list[EvaluationRow]:
    rows: list[EvaluationRow] = []
    for horizon in horizons:
        for regime in DEFAULT_REGIMES:
            subset_indices = [index for index, case in enumerate(cases) if regime == "all" or case.regime == regime]
            if not subset_indices:
                continue
            subset_cases = [cases[index] for index in subset_indices]
            subset_predictions = [predictions[index] for index in subset_indices]
            return_metrics = evaluate_return_predictions(subset_cases, subset_predictions, horizon=horizon)
            price_cases = [
                BacktestCase(
                    cutoff_timestamp=case.cutoff_timestamp,
                    actual=case.actual[:horizon],
                    predicted=subset_predictions[row_index][:horizon],
                    cutoff_close=case.cutoff_close,
                    context_closes=case.context_closes,
                    regime=case.regime,
                )
                for row_index, case in enumerate(subset_cases)
            ]
            price_metrics = evaluate_cases(price_cases)
            rows.append(
                EvaluationRow(
                    symbol=symbol,
                    model=model,
                    horizon=horizon,
                    regime=regime,
                    cases=len(subset_cases),
                    price_mae=price_metrics.mae,
                    price_rmse=price_metrics.rmse,
                    return_mae=return_metrics.return_mae,
                    return_rmse=return_metrics.return_rmse,
                    directional_accuracy=return_metrics.directional_accuracy,
                    precision=return_metrics.precision,
                    recall=return_metrics.recall,
                    f1=return_metrics.f1,
                    information_coefficient=return_metrics.information_coefficient,
                )
            )
    return rows


def _run_chronos(symbol: str, args: argparse.Namespace, adapter: Chronos2Adapter, db) -> tuple[list[BacktestCase], list[float]]:
    observations = load_close_observations(db, symbol, interval=args.interval, source=args.source, limit=args.history_limit)
    cases, _ = run_backtest(
        adapter,
        symbol,
        observations,
        context_length=args.context_length,
        horizon=max(args.horizons),
        stride=args.stride,
        start_case=args.start_case,
        max_cases=args.max_cases,
    )
    return cases, [case.predicted for case in cases]


def _run_kronos(symbol: str, args: argparse.Namespace, adapter: KronosSmallAdapter, db) -> tuple[list[BacktestCase], list[list[float]]]:
    observations = load_ohlcv_observations(db, symbol, interval=args.interval, source=args.source, limit=args.history_limit)
    horizon = max(args.horizons)
    minimum = args.context_length + horizon
    if len(observations) < minimum:
        raise ValueError(f"{symbol}: {len(observations)} OHLCV observations; need {minimum}")

    cases: list[BacktestCase] = []
    cutoff_end = args.context_length + args.start_case * args.stride
    while cutoff_end + horizon <= len(observations) and len(cases) < args.max_cases:
        context = observations[cutoff_end - args.context_length : cutoff_end]
        future = observations[cutoff_end : cutoff_end + horizon]
        context_closes = [item.close for item in context]
        request = ForecastRequest(
            symbol=symbol,
            values=context_closes,
            timestamps=[item.timestamp for item in context],
            horizon=horizon,
            opens=[item.open for item in context],
            highs=[item.high for item in context],
            lows=[item.low for item in context],
            volumes=[item.volume for item in context],
            amounts=[item.amount for item in context],
        )
        forecast = adapter.forecast(request)
        cases.append(
            BacktestCase(
                cutoff_timestamp=context[-1].timestamp,
                actual=[item.close for item in future],
                predicted=forecast.median,
                cutoff_close=context[-1].close,
                context_closes=context_closes,
                regime=classify_regime(context_closes),
            )
        )
        cutoff_end += args.stride
    return cases, [case.predicted for case in cases]


def _write_csv(rows: list[EvaluationRow], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EvaluationRow.__dataclass_fields__)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def _aggregate(rows: list[EvaluationRow], *, metric: str, label: str) -> None:
    print(f"\n=== {label} ===")
    models = sorted({row.model for row in rows})
    for model in models:
        for horizon in DEFAULT_HORIZONS:
            for regime in ("all", REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS_HIGH_VOLATILITY):
                values = [getattr(row, metric) for row in rows if row.model == model and row.horizon == horizon and row.regime == regime]
                values = [float(value) for value in values if value is not None]
                if not values:
                    continue
                summary = bootstrap_mean_ci(values, seed=20260913 + horizon)
                print(
                    f"{model:22s} h={horizon:2d} {regime:24s} n={summary.n:2d} "
                    f"mean={summary.mean:.4f} std={summary.std:.4f} "
                    f"95%CI=[{summary.ci_low:.4f}, {summary.ci_high:.4f}]"
                )


def _print_model_comparison(rows: list[EvaluationRow]) -> None:
    baselines = {BASELINE_LAST_CLOSE, BASELINE_PREVIOUS_RETURN, BASELINE_DRIFT, BASELINE_MOVING_AVERAGE_20}
    print("\n=== Model vs stronger baselines ===")
    for horizon in DEFAULT_HORIZONS:
        print(f"\nHorizon {horizon}d")
        baseline_rows = [r for r in rows if r.model in baselines and r.horizon == horizon and r.regime == "all"]
        model_rows = [r for r in rows if r.model not in baselines and r.horizon == horizon and r.regime == "all"]
        baseline_means = {}
        for baseline in sorted(baselines):
            values = [r.return_mae for r in baseline_rows if r.model == baseline]
            if values:
                baseline_means[baseline] = sum(values) / len(values)
        if not baseline_means:
            continue
        best_baseline, best_value = min(baseline_means.items(), key=lambda item: item[1])
        print(f"  Best baseline: {best_baseline} return MAE={best_value * 100:.4f}%")
        for model in sorted({r.model for r in model_rows}):
            values = [r.return_mae for r in model_rows if r.model == model]
            if not values:
                continue
            summary = bootstrap_mean_ci(values, seed=5000 + horizon)
            delta = summary.mean - best_value
            print(
                f"  {model}: return MAE={summary.mean * 100:.4f}% "
                f"delta={delta * 100:+.4f}pp 95%CI=[{summary.ci_low * 100:.4f}%, {summary.ci_high * 100:.4f}%]"
            )


def _print_decision(rows: list[EvaluationRow]) -> None:
    print("\n=== PRODUCTION DECISION GATE ===")
    print("Rule: a pretrained model is a candidate only if it beats the best baseline on return MAE at all three horizons and its 95% bootstrap CI does not overlap the baseline mean; direction and F1 are reported as supporting evidence.")
    baselines = {BASELINE_LAST_CLOSE, BASELINE_PREVIOUS_RETURN, BASELINE_DRIFT, BASELINE_MOVING_AVERAGE_20}
    model_names = sorted({r.model for r in rows if r.model not in baselines})
    candidates: list[str] = []
    for model in model_names:
        passes = True
        for horizon in DEFAULT_HORIZONS:
            baseline_values = [r.return_mae for r in rows if r.model in baselines and r.horizon == horizon and r.regime == "all"]
            if not baseline_values:
                passes = False
                break
            best_baseline = min(baseline_values)
            values = [r.return_mae for r in rows if r.model == model and r.horizon == horizon and r.regime == "all"]
            if not values:
                passes = False
                break
            summary = bootstrap_mean_ci(values, seed=9000 + horizon)
            if summary.ci_low is None or summary.ci_low >= best_baseline:
                passes = False
        if passes:
            candidates.append(model)
    if candidates:
        print(f"Candidate models: {', '.join(candidates)}")
    else:
        print("No pretrained model passes the production gate yet.")
        print("Decision: keep pretrained forecasters in research/benchmarking; do not promote Chronos-2 or Kronos-small to production from this dataset alone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanded causal NIFTY 50 benchmark for pretrained forecasters")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols; defaults to all 50 NIFTY 50 symbols")
    parser.add_argument("--nifty50-csv", default="data/reference/nifty50.csv")
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--horizons", default="1,5,10")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--history-limit", type=int, default=1000)
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--kronos-seed", type=int, default=101)
    parser.add_argument("--models", default="chronos,kronos", help="Comma-separated: chronos,kronos")
    parser.add_argument("--output", default="data/processed/forecast_benchmark_v2.csv")
    args = parser.parse_args()

    args.horizons = _parse_csv_ints(args.horizons)
    if args.horizons != DEFAULT_HORIZONS:
        raise ValueError("this benchmark protocol requires horizons exactly 1,5,10")
    args.models = tuple(item.strip().lower() for item in args.models.split(",") if item.strip())
    if not args.models or any(item not in DEFAULT_MODELS for item in args.models):
        raise ValueError("models must contain only chronos and/or kronos")
    if args.context_length < 2 or args.stride < 1 or args.max_cases < 1 or args.start_case < 0:
        raise ValueError("context-length >= 2, stride >= 1, max-cases >= 1, start-case >= 0 are required")
    if args.history_limit < args.context_length + max(args.horizons):
        raise ValueError("history-limit is too small for context-length + max horizon")

    symbols = _parse_symbols(args.symbols) if args.symbols else _load_nifty50_symbols(args.nifty50_csv)
    print("=== NIFTY 50 EXPANDED CAUSAL FORECAST BENCHMARK ===")
    print(f"Symbols: {len(symbols)}")
    print(f"Horizons: {args.horizons} trading sessions")
    print(f"Context: {args.context_length}, stride: {args.stride}, max cases/stock: {args.max_cases}")
    print(f"Models: {', '.join(args.models)}")
    print("Baselines: last_close, previous_return, drift, moving_average_20")
    print("Regimes: bull, bear, sideways/high-volatility; regime is assigned from context only")

    db = SessionLocal()
    rows: list[EvaluationRow] = []
    chronos = Chronos2Adapter(device_map=args.device_map) if "chronos" in args.models else None
    kronos = KronosSmallAdapter(device=args.device_map, seed=args.kronos_seed) if "kronos" in args.models else None
    try:
        for index, symbol in enumerate(symbols, start=1):
            print(f"[{index:02d}/{len(symbols)}] {symbol}")
            try:
                if chronos is not None:
                    cases, predictions = _run_chronos(symbol, args, chronos, db)
                    rows.extend(_rows_for_predictions(symbol, "Chronos-2", cases, predictions, args.horizons))
                    baseline_map = baseline_predictions(cases)
                    for baseline_name, baseline_prediction in baseline_map.items():
                        rows.extend(_rows_for_predictions(symbol, baseline_name, cases, baseline_prediction, args.horizons))
                    print(f"  Chronos-2: {len(cases)} causal cases")
                if kronos is not None:
                    cases, predictions = _run_kronos(symbol, args, kronos, db)
                    rows.extend(_rows_for_predictions(symbol, "Kronos-small", cases, predictions, args.horizons))
                    print(f"  Kronos-small: {len(cases)} causal cases")
            except Exception as exc:
                print(f"  ERROR: {exc}")
    finally:
        db.close()

    if not rows:
        raise RuntimeError("No benchmark rows were produced")
    _write_csv(rows, args.output)
    print(f"\nWrote {len(rows)} benchmark rows to {args.output}")
    _aggregate(rows, metric="return_mae", label="Return MAE variability across stocks")
    _aggregate(rows, metric="directional_accuracy", label="Directional accuracy variability across stocks")
    _print_model_comparison(rows)
    _print_decision(rows)


if __name__ == "__main__":
    main()
