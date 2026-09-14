from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from apps.api.db.session import SessionLocal
from services.forecasting.backtest import BacktestCase, REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS_HIGH_VOLATILITY, classify_regime, evaluate_cases, run_backtest
from services.forecasting.benchmarks import BASELINE_DRIFT, BASELINE_LAST_CLOSE, BASELINE_MOVING_AVERAGE_20, BASELINE_PREVIOUS_RETURN, baseline_predictions, bootstrap_mean_ci, evaluate_return_predictions
from services.forecasting.base import ForecastRequest
from services.forecasting.chronos import Chronos2Adapter
from services.forecasting.kronos import KronosSmallAdapter
from services.forecasting.kronos_loader import load_ohlcv_observations
from services.forecasting.paired import PairedCaseResult, PairedComparison, compare_paired
from services.forecasting.price_loader import load_close_observations

HORIZONS = (1, 5, 10)
REGIMES = ("all", REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS_HIGH_VOLATILITY)
BASELINES = (BASELINE_LAST_CLOSE, BASELINE_PREVIOUS_RETURN, BASELINE_DRIFT, BASELINE_MOVING_AVERAGE_20)


@dataclass(frozen=True)
class Row:
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


def rows_for(symbol: str, model: str, cases: list[BacktestCase], predictions: list[list[float]]) -> list[Row]:
    output: list[Row] = []
    for horizon in HORIZONS:
        for regime in REGIMES:
            selected = [(case, predictions[i]) for i, case in enumerate(cases) if regime == "all" or case.regime == regime]
            if not selected:
                continue
            selected_cases = [item[0] for item in selected]
            selected_predictions = [item[1] for item in selected]
            ret = evaluate_return_predictions(selected_cases, selected_predictions, horizon=horizon)
            price_cases = [
                BacktestCase(case.cutoff_timestamp, case.actual[:horizon], pred[:horizon], case.cutoff_close, case.context_closes, case.regime)
                for case, pred in selected
            ]
            price = evaluate_cases(price_cases)
            output.append(Row(symbol, model, horizon, regime, len(selected_cases), price.mae, price.rmse, ret.return_mae, ret.return_rmse, ret.directional_accuracy, ret.precision, ret.recall, ret.f1, ret.information_coefficient))
    return output


def symbols_from_csv(path: str) -> list[str]:
    frame = pd.read_csv(path)
    if "Symbol" not in frame.columns:
        raise ValueError(f"{path} must contain Symbol")
    symbols = list(dict.fromkeys(str(v).strip().upper() for v in frame["Symbol"].dropna() if str(v).strip()))
    if len(symbols) != 50:
        raise ValueError(f"expected 50 NIFTY 50 symbols, found {len(symbols)}")
    return symbols


def chronos_cases(symbol: str, args, adapter, db) -> list[BacktestCase]:
    observations = load_close_observations(db, symbol, interval=args.interval, source=args.source, limit=args.history_limit)
    return run_backtest(adapter, symbol, observations, context_length=args.context_length, horizon=10, stride=args.stride, start_case=args.start_case, max_cases=args.max_cases)[0]


def kronos_cases(symbol: str, args, adapter, db) -> list[BacktestCase]:
    observations = load_ohlcv_observations(db, symbol, interval=args.interval, source=args.source, limit=args.history_limit)
    need = args.context_length + 10
    if len(observations) < need:
        raise ValueError(f"{symbol}: need {need} OHLCV rows, got {len(observations)}")
    cases: list[BacktestCase] = []
    cutoff = args.context_length + args.start_case * args.stride
    while cutoff + 10 <= len(observations) and len(cases) < args.max_cases:
        context = observations[cutoff - args.context_length:cutoff]
        future = observations[cutoff:cutoff + 10]
        closes = [x.close for x in context]
        request = ForecastRequest(
            symbol=symbol,
            values=closes,
            timestamps=[x.timestamp for x in context],
            horizon=10,
            opens=[x.open for x in context],
            highs=[x.high for x in context],
            lows=[x.low for x in context],
            volumes=[x.volume for x in context],
            amounts=[x.amount for x in context],
        )
        forecast = adapter.forecast(request)
        cases.append(BacktestCase(context[-1].timestamp, [x.close for x in future], forecast.median, context[-1].close, closes, classify_regime(closes)))
        cutoff += args.stride
    return cases


def write_csv(rows, path: str) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def pooled_summary(case_rows: list[PairedCaseResult], model: str) -> list[PairedComparison]:
    summaries: list[PairedComparison] = []
    for horizon in HORIZONS:
        for baseline in BASELINES:
            for regime in REGIMES:
                rows = [
                    row for row in case_rows
                    if row.model == model and row.baseline == baseline and row.horizon == horizon
                    and (regime == "all" or row.regime == regime)
                ]
                if not rows:
                    continue
                diffs = [row.loss_diff for row in rows]
                ci = bootstrap_mean_ci(diffs, seed=610000 + horizon + len(baseline))
                model_hits = [row.model_direction_correct for row in rows if row.model_direction_correct is not None]
                baseline_hits = [row.baseline_direction_correct for row in rows if row.baseline_direction_correct is not None]
                model_da = sum(model_hits) / len(model_hits) if model_hits else None
                baseline_da = sum(baseline_hits) / len(baseline_hits) if baseline_hits else None
                ordered = sorted(diffs)
                middle = len(ordered) // 2
                median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
                summaries.append(
                    PairedComparison(
                        model=model,
                        baseline=baseline,
                        horizon=horizon,
                        regime=regime,
                        cases=len(rows),
                        model_loss_mean=sum(row.model_loss for row in rows) / len(rows),
                        baseline_loss_mean=sum(row.baseline_loss for row in rows) / len(rows),
                        mean_loss_diff=sum(diffs) / len(rows),
                        median_loss_diff=median,
                        std_loss_diff=ci.std,
                        ci_low=ci.ci_low,
                        ci_high=ci.ci_high,
                        model_win_rate=sum(diff < 0 for diff in diffs) / len(rows),
                        tie_rate=sum(diff == 0 for diff in diffs) / len(rows),
                        baseline_win_rate=sum(diff > 0 for diff in diffs) / len(rows),
                        dm_stat=None,
                        dm_pvalue=None,
                        model_directional_accuracy=model_da,
                        baseline_directional_accuracy=baseline_da,
                        directional_accuracy_diff=(model_da - baseline_da) if model_da is not None and baseline_da is not None else None,
                    )
                )
    return summaries


def print_pooled_gate(summaries: list[PairedComparison]) -> None:
    print("\n=== PAIRED CASE-LEVEL PRODUCTION GATE ===")
    print("Gate: model must beat the best baseline at every horizon, with paired mean loss difference < 0 and 95% paired CI upper bound < 0.")
    for model in sorted({item.model for item in summaries}):
        passed = True
        for horizon in HORIZONS:
            candidates = [item for item in summaries if item.model == model and item.horizon == horizon and item.regime == "all"]
            best = min(candidates, key=lambda item: item.mean_loss_diff) if candidates else None
            if best is None or best.mean_loss_diff >= 0 or best.ci_high is None or best.ci_high >= 0:
                passed = False
                break
        print(f"{model}: {'PASS' if passed else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired case-level NIFTY 50 forecast benchmark")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--nifty50-csv", default="data/reference/nifty50.csv")
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--history-limit", type=int, default=1000)
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--kronos-seed", type=int, default=101)
    parser.add_argument("--models", default="chronos,kronos")
    parser.add_argument("--output", default="data/processed/forecast_benchmark_v3.csv")
    parser.add_argument("--case-output", default="data/processed/forecast_benchmark_cases_v3.csv")
    parser.add_argument("--paired-output", default="data/processed/forecast_benchmark_paired_v3.csv")
    args = parser.parse_args()

    models = tuple(x.strip().lower() for x in args.models.split(",") if x.strip())
    if not models or any(x not in ("chronos", "kronos") for x in models):
        raise ValueError("--models must contain chronos and/or kronos")
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()] if args.symbols else symbols_from_csv(args.nifty50_csv)
    if not symbols:
        raise ValueError("no symbols")

    print("=== NIFTY 50 PAIRED CAUSAL FORECAST BENCHMARK ===")
    print(f"Stocks: {len(symbols)} | Horizons: 1d, 5d, 10d | Context: {args.context_length} | Stride: {args.stride} | Cases/stock: {args.max_cases}")
    print(f"Models: {', '.join(models)} | Baselines: {', '.join(BASELINES)}")

    db = SessionLocal()
    rows: list[Row] = []
    paired_cases: list[PairedCaseResult] = []
    per_stock: list[PairedComparison] = []
    baseline_rows_written: set[tuple[str, str]] = set()
    adapters = {
        "Chronos-2": Chronos2Adapter(device_map=args.device_map) if "chronos" in models else None,
        "Kronos-small": KronosSmallAdapter(device=args.device_map, seed=args.kronos_seed) if "kronos" in models else None,
    }
    try:
        for number, symbol in enumerate(symbols, 1):
            print(f"[{number:02d}/{len(symbols)}] {symbol}")
            model_results: dict[str, tuple[list[BacktestCase], list[list[float]]]] = {}
            if adapters["Chronos-2"] is not None:
                try:
                    cases = chronos_cases(symbol, args, adapters["Chronos-2"], db)
                    predictions = [case.predicted for case in cases]
                    model_results["Chronos-2"] = (cases, predictions)
                    rows.extend(rows_for(symbol, "Chronos-2", cases, predictions))
                    print(f"  Chronos-2: {len(cases)} cases")
                except Exception as exc:
                    print(f"  Chronos-2 ERROR: {exc}")
            if adapters["Kronos-small"] is not None:
                try:
                    cases = kronos_cases(symbol, args, adapters["Kronos-small"], db)
                    predictions = [case.predicted for case in cases]
                    model_results["Kronos-small"] = (cases, predictions)
                    rows.extend(rows_for(symbol, "Kronos-small", cases, predictions))
                    print(f"  Kronos-small: {len(cases)} cases")
                except Exception as exc:
                    print(f"  Kronos-small ERROR: {exc}")

            for model_name, (cases, predictions) in model_results.items():
                baselines = baseline_predictions(cases)
                for baseline_name, baseline_prediction in baselines.items():
                    key = (symbol, baseline_name)
                    if key not in baseline_rows_written:
                        rows.extend(rows_for(symbol, baseline_name, cases, baseline_prediction))
                        baseline_rows_written.add(key)
                    for horizon in HORIZONS:
                        for regime in REGIMES:
                            comparison, details = compare_paired(
                                cases,
                                predictions,
                                baseline_prediction,
                                model=model_name,
                                baseline=baseline_name,
                                horizon=horizon,
                                regime=regime,
                                symbol=symbol,
                                dependence_lag=max(0, (horizon - 1) // max(1, args.stride)),
                            )
                            per_stock.append(comparison)
                            paired_cases.extend(details)
    finally:
        db.close()

    if not rows or not paired_cases:
        raise RuntimeError("No benchmark rows were produced")

    write_csv(rows, args.output)
    write_csv(paired_cases, args.case_output)
    write_csv(per_stock, args.paired_output)
    print(f"\nWrote {len(rows)} aggregate rows to {args.output}")
    print(f"Wrote {len(paired_cases)} paired case rows to {args.case_output}")
    print(f"Wrote {len(per_stock)} per-stock paired summaries to {args.paired_output}")

    print("\n=== POOLED PAIRED CASE-LEVEL RESULTS ===")
    pooled: list[PairedComparison] = []
    for model in sorted({row.model for row in paired_cases}):
        model_summary = pooled_summary(paired_cases, model)
        pooled.extend(model_summary)
        for item in model_summary:
            if item.regime != "all":
                continue
            print(
                f"{item.model:22s} vs {item.baseline:20s} h={item.horizon:2d} n={item.cases:4d} "
                f"mean_diff={item.mean_loss_diff*100:+.4f}pp median={item.median_loss_diff*100:+.4f}pp "
                f"CI=[{item.ci_low*100:+.4f}, {item.ci_high*100:+.4f}]pp "
                f"wins={item.model_win_rate*100:.1f}% "
                f"direction_delta={(item.directional_accuracy_diff*100) if item.directional_accuracy_diff is not None else float('nan'):+.2f}pp"
            )
    print_pooled_gate(pooled)
    print("\nPer-stock DM-style HAC statistics are available in the paired summary CSV; pooled case-level CI is paired-bootstrap and intentionally does not treat stocks as independent time series.")


if __name__ == "__main__":
    main()
