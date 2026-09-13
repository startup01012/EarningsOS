from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from apps.api.db.session import SessionLocal
from services.forecasting.backtest import BacktestCase, evaluate_cases, run_backtest
from services.forecasting.benchmarks import evaluate_predictions, naive_last_close_predictions
from services.forecasting.base import ForecastRequest
from services.forecasting.chronos import Chronos2Adapter
from services.forecasting.kronos import KronosSmallAdapter
from services.forecasting.kronos_loader import load_ohlcv_observations
from services.forecasting.price_loader import load_close_observations


DEFAULT_SYMBOLS = (
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "AXISBANK", "BHARTIARTL", "ITC", "LT", "SBIN",
)


@dataclass(frozen=True)
class Result:
    symbol: str
    model: str
    cases: int
    mae: float
    rmse: float
    mape: float | None
    smape: float | None
    directional_accuracy: float | None
    naive_mae: float
    naive_rmse: float
    naive_mape: float | None
    naive_smape: float | None


def _parse_symbols(value: str) -> list[str]:
    symbols = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required")
    return list(dict.fromkeys(symbols))


def _load_default_symbols(path: str) -> list[str]:
    csv_path = Path(path)
    if not csv_path.exists():
        return list(DEFAULT_SYMBOLS)
    frame = pd.read_csv(csv_path)
    if "Symbol" not in frame.columns:
        return list(DEFAULT_SYMBOLS)
    symbols = [str(value).strip().upper() for value in frame["Symbol"].dropna()]
    symbols = list(dict.fromkeys(symbols))
    return symbols[:10] if symbols else list(DEFAULT_SYMBOLS)


def _print_metric_block(result: Result) -> None:
    mape_text = f"{result.mape:.4f}%" if result.mape is not None else "N/A"
    print(
        f"  {result.model}: cases={result.cases} "
        f"MAE={result.mae:.4f} RMSE={result.rmse:.4f} MAPE={mape_text}"
    )
    if result.mape is not None and result.naive_mape is not None:
        mape_change = f"{result.mape - result.naive_mape:+.4f}pp"
    else:
        mape_change = "N/A"
    print(
        f"    vs naive: MAE {result.mae - result.naive_mae:+.4f}, "
        f"RMSE {result.rmse - result.naive_rmse:+.4f}, MAPE {mape_change}"
    )


def _summarize(results: list[Result]) -> None:
    print("\n=== Per-stock results ===")
    for symbol in sorted({result.symbol for result in results}):
        print(f"\n{symbol}")
        for result in [r for r in results if r.symbol == symbol]:
            _print_metric_block(result)

    print("\n=== Aggregate model comparison ===")
    for model in sorted({result.model for result in results}):
        rows = [r for r in results if r.model == model]
        if not rows:
            continue
        mae_changes = [r.mae - r.naive_mae for r in rows]
        rmse_changes = [r.rmse - r.naive_rmse for r in rows]
        print(f"\n{model}")
        print(f"  Stocks evaluated: {len(rows)}")
        print(f"  Mean MAE: {sum(r.mae for r in rows) / len(rows):.4f}")
        print(f"  Mean RMSE: {sum(r.rmse for r in rows) / len(rows):.4f}")
        print(f"  Mean MAE change vs naive: {sum(mae_changes) / len(rows):+.4f}")
        print(f"  Mean RMSE change vs naive: {sum(rmse_changes) / len(rows):+.4f}")
        print(f"  Stocks beating naive on MAE: {sum(value < 0 for value in mae_changes)}/{len(rows)}")
        print(f"  Stocks beating naive on RMSE: {sum(value < 0 for value in rmse_changes)}/{len(rows)}")


def _run_chronos(symbol: str, args: argparse.Namespace, adapter: Chronos2Adapter, db) -> Result:
    observations = load_close_observations(
        db, symbol, interval=args.interval, source=args.source, limit=args.history_limit
    )
    cases, metrics = run_backtest(
        adapter, symbol, observations,
        context_length=args.context_length,
        horizon=args.horizon,
        stride=args.stride,
        start_case=args.start_case,
        max_cases=args.max_cases,
    )
    naive_metrics = evaluate_predictions(cases, naive_last_close_predictions(cases))
    return Result(
        symbol, "Chronos-2", metrics.cases, metrics.mae, metrics.rmse,
        metrics.mape, metrics.smape, metrics.directional_accuracy,
        naive_metrics.mae, naive_metrics.rmse, naive_metrics.mape, naive_metrics.smape,
    )


def _run_kronos(symbol: str, args: argparse.Namespace, adapter: KronosSmallAdapter, db) -> Result:
    observations = load_ohlcv_observations(
        db, symbol, interval=args.interval, source=args.source, limit=args.history_limit
    )
    minimum = args.context_length + args.horizon
    if len(observations) < minimum:
        raise ValueError(f"{symbol}: {len(observations)} OHLCV observations; need {minimum}")

    cases: list[BacktestCase] = []
    cutoff_end = args.context_length + args.start_case * args.stride
    while cutoff_end + args.horizon <= len(observations) and len(cases) < args.max_cases:
        context = observations[cutoff_end - args.context_length : cutoff_end]
        future = observations[cutoff_end : cutoff_end + args.horizon]
        request = ForecastRequest(
            symbol=symbol,
            values=[item.close for item in context],
            timestamps=[item.timestamp for item in context],
            horizon=args.horizon,
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
            )
        )
        cutoff_end += args.stride

    metrics = evaluate_cases(cases)
    naive_metrics = evaluate_predictions(cases, naive_last_close_predictions(cases))
    return Result(
        symbol, f"Kronos-small(seed={args.kronos_seed})", metrics.cases,
        metrics.mae, metrics.rmse, metrics.mape, metrics.smape,
        metrics.directional_accuracy, naive_metrics.mae, naive_metrics.rmse,
        naive_metrics.mape, naive_metrics.smape,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-stock causal benchmark for pretrained forecasters")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols; defaults to first 10 NIFTY 50 symbols")
    parser.add_argument("--nifty50-csv", default="data/reference/nifty50.csv")
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--start-case", type=int, default=21)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=1000)
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--kronos-seed", type=int, default=101)
    parser.add_argument("--skip-kronos", action="store_true")
    args = parser.parse_args()

    if args.context_length < 2 or args.horizon < 1 or args.stride < 1:
        raise ValueError("context-length must be >= 2; horizon and stride must be >= 1")
    if args.start_case < 0 or args.max_cases < 1:
        raise ValueError("start-case must be >= 0 and max-cases must be >= 1")
    if args.history_limit < args.context_length + args.horizon:
        raise ValueError("history-limit is too small for context-length + horizon")

    symbols = _parse_symbols(args.symbols) if args.symbols else _load_default_symbols(args.nifty50_csv)
    print("=== NIFTY 50 causal forecast benchmark ===")
    print(f"Symbols: {symbols}")
    print(f"Context: {args.context_length}, horizon: {args.horizon}, stride: {args.stride}")
    print(f"Start case: {args.start_case}, max cases: {args.max_cases}")
    print(f"History limit: {args.history_limit}, source: {args.source}")
    if not args.skip_kronos:
        print(f"Kronos seed: {args.kronos_seed}")

    db = SessionLocal()
    results: list[Result] = []
    chronos = Chronos2Adapter(device_map=args.device_map)
    # KronosSmallAdapter accepts a device argument, not Chronos' device_map.
    kronos = None if args.skip_kronos else KronosSmallAdapter(
        device=args.device_map, seed=args.kronos_seed
    )
    try:
        for index, symbol in enumerate(symbols, start=1):
            print(f"\n[{index}/{len(symbols)}] {symbol}")
            try:
                result = _run_chronos(symbol, args, chronos, db)
                results.append(result)
                print(f"  Chronos-2: MAE={result.mae:.4f} vs naive={result.naive_mae:.4f}")
            except Exception as exc:
                print(f"  Chronos-2 ERROR: {exc}")

            if kronos is not None:
                try:
                    result = _run_kronos(symbol, args, kronos, db)
                    results.append(result)
                    print(f"  Kronos-small: MAE={result.mae:.4f} vs naive={result.naive_mae:.4f}")
                except Exception as exc:
                    print(f"  Kronos-small ERROR: {exc}")
    finally:
        db.close()

    if not results:
        raise RuntimeError("No benchmark results were produced")
    _summarize(results)


if __name__ == "__main__":
    main()
