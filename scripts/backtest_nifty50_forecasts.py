from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from apps.api.db.session import SessionLocal
from services.forecasting.backtest import BacktestCase, evaluate_cases, run_backtest
from services.forecasting.benchmarks import evaluate_predictions, evaluate_return_predictions, naive_last_close_predictions
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
    horizon: int
    mae: float
    rmse: float
    mape: float | None
    smape: float | None
    directional_accuracy: float | None
    return_mae: float | None
    return_rmse: float | None
    return_directional_accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    information_coefficient: float | None
    naive_mae: float
    naive_rmse: float
    naive_mape: float | None
    naive_smape: float | None
    naive_return_mae: float | None
    naive_return_rmse: float | None


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


def _build_result(symbol: str, model: str, cases: list[BacktestCase], metrics) -> Result:
    if not cases:
        raise ValueError("benchmark produced no cases")
    horizon = len(cases[0].actual)
    predictions = [case.predicted for case in cases]
    naive_predictions = naive_last_close_predictions(cases)
    naive_metrics = evaluate_predictions(cases, naive_predictions)
    return_metrics = evaluate_return_predictions(cases, predictions, horizon=horizon)
    naive_return_metrics = evaluate_return_predictions(cases, naive_predictions, horizon=horizon)
    return Result(
        symbol=symbol,
        model=model,
        cases=metrics.cases,
        horizon=horizon,
        mae=metrics.mae,
        rmse=metrics.rmse,
        mape=metrics.mape,
        smape=metrics.smape,
        directional_accuracy=metrics.directional_accuracy,
        return_mae=return_metrics.return_mae,
        return_rmse=return_metrics.return_rmse,
        return_directional_accuracy=return_metrics.directional_accuracy,
        precision=return_metrics.precision,
        recall=return_metrics.recall,
        f1=return_metrics.f1,
        information_coefficient=return_metrics.information_coefficient,
        naive_mae=naive_metrics.mae,
        naive_rmse=naive_metrics.rmse,
        naive_mape=naive_metrics.mape,
        naive_smape=naive_metrics.smape,
        naive_return_mae=naive_return_metrics.return_mae,
        naive_return_rmse=naive_return_metrics.return_rmse,
    )


def _print_metric_block(result: Result) -> None:
    mape_text = f"{result.mape:.4f}%" if result.mape is not None else "N/A"
    return_mae_text = f"{result.return_mae * 100:.4f}%" if result.return_mae is not None else "N/A"
    return_rmse_text = f"{result.return_rmse * 100:.4f}%" if result.return_rmse is not None else "N/A"
    direction_text = f"{result.return_directional_accuracy * 100:.2f}%" if result.return_directional_accuracy is not None else "N/A"
    precision_text = f"{result.precision * 100:.2f}%" if result.precision is not None else "N/A"
    recall_text = f"{result.recall * 100:.2f}%" if result.recall is not None else "N/A"
    f1_text = f"{result.f1 * 100:.2f}%" if result.f1 is not None else "N/A"
    ic_text = f"{result.information_coefficient:.4f}" if result.information_coefficient is not None else "N/A"
    print(
        f"  {result.model}: cases={result.cases} MAE={result.mae:.4f} "
        f"RMSE={result.rmse:.4f} MAPE={mape_text}"
    )
    print(
        f"    return@{result.horizon}d: MAE={return_mae_text} RMSE={return_rmse_text} "
        f"direction={direction_text} precision={precision_text} recall={recall_text} "
        f"F1={f1_text} IC={ic_text}"
    )
    mape_change = f"{result.mape - result.naive_mape:+.4f}pp" if result.mape is not None and result.naive_mape is not None else "N/A"
    return_mae_change = (
        f"{(result.return_mae - result.naive_return_mae) * 100:+.4f}pp"
        if result.return_mae is not None and result.naive_return_mae is not None else "N/A"
    )
    print(
        f"    vs naive: MAE {result.mae - result.naive_mae:+.4f}, "
        f"RMSE {result.rmse - result.naive_rmse:+.4f}, MAPE {mape_change}, "
        f"return MAE {return_mae_change}"
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
        return_mae_changes = [r.return_mae - r.naive_return_mae for r in rows if r.return_mae is not None and r.naive_return_mae is not None]
        print(f"\n{model}")
        print(f"  Stocks evaluated: {len(rows)}")
        print(f"  Mean MAE: {sum(r.mae for r in rows) / len(rows):.4f}")
        print(f"  Mean RMSE: {sum(r.rmse for r in rows) / len(rows):.4f}")
        print(f"  Mean MAE change vs naive: {sum(mae_changes) / len(rows):+.4f}")
        print(f"  Mean RMSE change vs naive: {sum(rmse_changes) / len(rows):+.4f}")
        print(f"  Stocks beating naive on MAE: {sum(value < 0 for value in mae_changes)}/{len(rows)}")
        print(f"  Stocks beating naive on RMSE: {sum(value < 0 for value in rmse_changes)}/{len(rows)}")
        if return_mae_changes:
            print(f"  Mean return MAE change vs naive: {sum(return_mae_changes) / len(return_mae_changes) * 100:+.4f}pp")
            print(f"  Stocks beating naive on return MAE: {sum(value < 0 for value in return_mae_changes)}/{len(return_mae_changes)}")
        direction_values = [r.return_directional_accuracy for r in rows if r.return_directional_accuracy is not None]
        f1_values = [r.f1 for r in rows if r.f1 is not None]
        ic_values = [r.information_coefficient for r in rows if r.information_coefficient is not None]
        if direction_values:
            print(f"  Mean return directional accuracy: {sum(direction_values) / len(direction_values) * 100:.2f}%")
        if f1_values:
            print(f"  Mean return F1: {sum(f1_values) / len(f1_values) * 100:.2f}%")
        if ic_values:
            print(f"  Mean return IC: {sum(ic_values) / len(ic_values):.4f}")


def _run_chronos(symbol: str, args: argparse.Namespace, adapter: Chronos2Adapter, db) -> Result:
    observations = load_close_observations(db, symbol, interval=args.interval, source=args.source, limit=args.history_limit)
    cases, metrics = run_backtest(
        adapter, symbol, observations,
        context_length=args.context_length, horizon=args.horizon,
        stride=args.stride, start_case=args.start_case, max_cases=args.max_cases,
    )
    return _build_result(symbol, "Chronos-2", cases, metrics)


def _run_kronos(symbol: str, args: argparse.Namespace, adapter: KronosSmallAdapter, db) -> Result:
    observations = load_ohlcv_observations(db, symbol, interval=args.interval, source=args.source, limit=args.history_limit)
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
        cases.append(BacktestCase(
            cutoff_timestamp=context[-1].timestamp,
            actual=[item.close for item in future],
            predicted=forecast.median,
            cutoff_close=context[-1].close,
        ))
        cutoff_end += args.stride

    return _build_result(symbol, f"Kronos-small(seed={args.kronos_seed})", cases, evaluate_cases(cases))


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
    kronos = None if args.skip_kronos else KronosSmallAdapter(device=args.device_map, seed=args.kronos_seed)
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
