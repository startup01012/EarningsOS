from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from apps.api.db.session import SessionLocal
from services.forecasting.backtest import BacktestCase, classify_regime, run_backtest
from services.forecasting.base import ForecastRequest
from services.forecasting.benchmarks import baseline_predictions
from services.forecasting.chronos import Chronos2Adapter
from services.forecasting.fincast import FinCastAdapter
from services.forecasting.kronos import KronosSmallAdapter
from services.forecasting.kronos_loader import load_ohlcv_observations
from services.forecasting.paired import compare_paired
from services.forecasting.price_loader import load_close_observations
from services.forecasting.timesfm import TimesFM25Adapter

DEFAULT_SYMBOLS = ("ADANIENT", "AXISBANK", "HDFCBANK", "INFY", "RELIANCE")
DEFAULT_MODELS = ("chronos", "timesfm", "kronos", "fincast")
HORIZONS = (1, 5, 10)
BASELINES = ("last_close", "previous_return", "drift", "moving_average_20")


def _close_cases(symbol: str, adapter, args, db) -> list[BacktestCase]:
    observations = load_close_observations(
        db,
        symbol,
        interval=args.interval,
        source=args.source,
        limit=args.history_limit,
    )
    return run_backtest(
        adapter,
        symbol,
        observations,
        context_length=args.context_length,
        horizon=max(HORIZONS),
        stride=args.stride,
        start_case=args.start_case,
        max_cases=args.max_cases,
    )[0]


def _kronos_cases(symbol: str, adapter: KronosSmallAdapter, args, db) -> list[BacktestCase]:
    observations = load_ohlcv_observations(
        db,
        symbol,
        interval=args.interval,
        source=args.source,
        limit=args.history_limit,
    )
    ordered = sorted(observations, key=lambda item: item.timestamp)
    horizon = max(HORIZONS)
    cutoff_end = args.context_length + args.start_case * args.stride
    if cutoff_end + horizon > len(ordered):
        raise ValueError(
            f"start_case requires {cutoff_end + horizon} observations, got {len(ordered)}"
        )

    cases: list[BacktestCase] = []
    while cutoff_end + horizon <= len(ordered):
        context = ordered[cutoff_end - args.context_length : cutoff_end]
        future = ordered[cutoff_end : cutoff_end + horizon]
        request = ForecastRequest(
            symbol=symbol,
            values=[item.close for item in context],
            timestamps=[item.timestamp for item in context],
            horizon=horizon,
            opens=[item.open for item in context],
            highs=[item.high for item in context],
            lows=[item.low for item in context],
            volumes=[item.volume for item in context],
            amounts=[item.amount for item in context],
        )
        result = adapter.forecast(request)
        context_closes = [item.close for item in context]
        cases.append(
            BacktestCase(
                cutoff_timestamp=context[-1].timestamp,
                actual=[item.close for item in future],
                predicted=result.median,
                cutoff_close=context[-1].close,
                context_closes=context_closes,
                regime=classify_regime(context_closes),
            )
        )
        if len(cases) >= args.max_cases:
            break
        cutoff_end += args.stride
    return cases


def build_adapter(name: str, args):
    if name == "chronos":
        return Chronos2Adapter(device_map=args.device)
    if name == "timesfm":
        return TimesFM25Adapter(device=args.device)
    if name == "kronos":
        return KronosSmallAdapter(device=args.device, max_context=args.context_length, seed=args.seed)
    if name == "fincast":
        return FinCastAdapter(
            model_path=args.fincast_model_path,
            fincast_root=args.fincast_root,
            device=args.device,
        )
    raise ValueError(f"unsupported model: {name}")


def _paired_rows(symbol: str, model_name: str, cases: list[BacktestCase]):
    predictions = [case.predicted for case in cases]
    baselines = baseline_predictions(cases)
    rows = []
    for baseline_name in BASELINES:
        for horizon in HORIZONS:
            comparison, _ = compare_paired(
                cases,
                predictions,
                baselines[baseline_name],
                model=model_name,
                baseline=baseline_name,
                horizon=horizon,
                regime="all",
                symbol=symbol,
                dependence_lag=0,
            )
            rows.append(comparison)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast pretrained-only forecasting model funnel")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=15)
    parser.add_argument("--history-limit", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--fincast-root", default=None)
    parser.add_argument("--fincast-model-path", default=None)
    parser.add_argument("--output", default="data/processed/pretrained_model_screen.csv")
    args = parser.parse_args()

    symbols = tuple(x.strip().upper() for x in args.symbols.split(",") if x.strip())
    models = tuple(x.strip().lower() for x in args.models.split(",") if x.strip())
    unknown = set(models) - set(DEFAULT_MODELS)
    if unknown:
        raise ValueError(f"unsupported models: {sorted(unknown)}")
    if args.max_cases < 1 or args.context_length < 32 or args.stride < 1:
        raise ValueError("max-cases >= 1, context-length >= 32 and stride >= 1 are required")

    print("=== FAST PRETRAINED MODEL SCREEN ===")
    print(f"Stocks: {symbols}")
    print(f"Models: {models}")
    print(f"Cases/stock/model: {args.max_cases} | Horizons: {HORIZONS} | Context: {args.context_length}")
    print("No training or fine-tuning is performed.")

    db = SessionLocal()
    output_rows: list[dict[str, object]] = []
    try:
        for model_name in models:
            print(f"\n--- {model_name} ---")
            adapter = build_adapter(model_name, args)
            model_started = time.perf_counter()
            all_rows = []
            stock_count = 0
            try:
                for symbol in symbols:
                    started = time.perf_counter()
                    if model_name == "kronos":
                        cases = _kronos_cases(symbol, adapter, args, db)
                    else:
                        cases = _close_cases(symbol, adapter, args, db)
                    all_rows.extend(_paired_rows(symbol, model_name, cases))
                    stock_count += 1
                    print(f"  {symbol}: {len(cases)} cases ({time.perf_counter() - started:.1f}s)")

                for row in all_rows:
                    output_rows.append(
                        {
                            "model": row.model,
                            "baseline": row.baseline,
                            "horizon": row.horizon,
                            "cases": row.cases,
                            "model_loss_mean": row.model_loss_mean,
                            "baseline_loss_mean": row.baseline_loss_mean,
                            "mean_loss_diff": row.mean_loss_diff,
                            "ci_low": row.ci_low,
                            "ci_high": row.ci_high,
                            "model_win_rate": row.model_win_rate,
                            "stocks": stock_count,
                            "runtime_seconds": round(time.perf_counter() - model_started, 3),
                            "status": "ok",
                        }
                    )
            except Exception as exc:
                elapsed = round(time.perf_counter() - model_started, 3)
                print(f"  SKIPPED: {type(exc).__name__}: {exc}")
                output_rows.append(
                    {
                        "model": model_name,
                        "baseline": "",
                        "horizon": "",
                        "cases": 0,
                        "model_loss_mean": "",
                        "baseline_loss_mean": "",
                        "mean_loss_diff": "",
                        "ci_low": "",
                        "ci_high": "",
                        "model_win_rate": "",
                        "stocks": stock_count,
                        "runtime_seconds": elapsed,
                        "status": f"error: {type(exc).__name__}: {exc}",
                    }
                )

    finally:
        db.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0]) if output_rows else ["model", "status"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nWrote {len(output_rows)} rows to {output}")
    print("\n=== SCREENING DECISION ===")
    for model in models:
        good = [
            row for row in output_rows
            if row["model"] == model and row["status"] == "ok" and row["baseline"] == "last_close"
        ]
        passed = all(
            float(row["mean_loss_diff"]) < 0
            and float(row["ci_high"]) < 0
            and float(row["model_win_rate"]) > 0.5
            for row in good
        ) and len(good) == len(HORIZONS)
        print(f"{model}: {'PASS' if passed else 'FAIL/REVIEW'}")


if __name__ == "__main__":
    main()
