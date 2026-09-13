from __future__ import annotations

import argparse

from apps.api.db.session import SessionLocal
from services.forecasting.backtest import BacktestCase, evaluate_cases
from services.forecasting.benchmarks import evaluate_predictions, naive_last_close_predictions
from services.forecasting.base import ForecastRequest
from services.forecasting.kronos import KronosSmallAdapter
from services.forecasting.kronos_loader import load_ohlcv_observations


def build_request(symbol: str, context, horizon: int) -> ForecastRequest:
    return ForecastRequest(
        symbol=symbol,
        values=[row.close for row in context],
        timestamps=[row.timestamp for row in context],
        horizon=horizon,
        opens=[row.open for row in context],
        highs=[row.high for row in context],
        lows=[row.low for row in context],
        volumes=[row.volume for row in context],
        amounts=[row.amount for row in context],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a strictly causal rolling backtest with pretrained Kronos-small"
    )
    parser.add_argument("--symbol", default="RELIANCE")
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--start-case", type=int, default=21)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=1000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.context_length < 2:
        raise ValueError("--context-length must be >= 2")
    if args.horizon < 1:
        raise ValueError("--horizon must be >= 1")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.start_case < 0:
        raise ValueError("--start-case must be >= 0")
    if args.max_cases < 1:
        raise ValueError("--max-cases must be >= 1")
    if args.context_length > 512:
        raise ValueError("Kronos-small supports max_context <= 512")

    db = SessionLocal()
    try:
        observations = load_ohlcv_observations(
            db,
            args.symbol,
            interval=args.interval,
            source=args.source,
            limit=args.history_limit,
        )
    finally:
        db.close()

    minimum = args.context_length + args.horizon
    if len(observations) < minimum:
        raise ValueError(
            f"at least {minimum} observations are required, got {len(observations)}"
        )

    adapter = KronosSmallAdapter(device=args.device, max_context=512)
    cases: list[BacktestCase] = []
    cutoff_end = args.context_length + args.start_case * args.stride

    if cutoff_end + args.horizon > len(observations):
        raise ValueError(
            "start_case points beyond the available observations: "
            f"cutoff requires {cutoff_end + args.horizon} observations, got {len(observations)}"
        )

    while cutoff_end + args.horizon <= len(observations):
        context = observations[cutoff_end - args.context_length : cutoff_end]
        future = observations[cutoff_end : cutoff_end + args.horizon]
        request = build_request(args.symbol, context, args.horizon)
        result = adapter.forecast(request)
        cases.append(
            BacktestCase(
                cutoff_timestamp=context[-1].timestamp,
                actual=[row.close for row in future],
                predicted=result.median,
                cutoff_close=context[-1].close,
            )
        )
        if len(cases) >= args.max_cases:
            break
        cutoff_end += args.stride

    metrics = evaluate_cases(cases)
    naive_predictions = naive_last_close_predictions(cases)
    naive_metrics = evaluate_predictions(cases, naive_predictions)

    print(f"Model: {adapter.model_id}")
    print(f"Symbol: {args.symbol.strip().upper()}")
    print(f"Available observations: {len(observations)}")
    print(f"Start case: {args.start_case}")
    print(f"Cases: {metrics.cases}")
    print(f"Horizon: {metrics.horizons}")
    print(f"Context length: {args.context_length}")
    print("\n=== Kronos-small ===")
    print(f"MAE: {metrics.mae:.6f}")
    print(f"RMSE: {metrics.rmse:.6f}")
    print(f"MAPE: {metrics.mape:.6f}%" if metrics.mape is not None else "MAPE: N/A")
    print(f"sMAPE: {metrics.smape:.6f}%" if metrics.smape is not None else "sMAPE: N/A")
    print(
        f"Directional accuracy: {metrics.directional_accuracy:.2%}"
        if metrics.directional_accuracy is not None
        else "Directional accuracy: N/A"
    )
    print("\n=== Naive last-close baseline ===")
    print(f"MAE: {naive_metrics.mae:.6f}")
    print(f"RMSE: {naive_metrics.rmse:.6f}")
    print(f"MAPE: {naive_metrics.mape:.6f}%" if naive_metrics.mape is not None else "MAPE: N/A")
    print(f"sMAPE: {naive_metrics.smape:.6f}%" if naive_metrics.smape is not None else "sMAPE: N/A")
    print("Directional accuracy: N/A (constant forecast has no directional signal)")
    print("\n=== Kronos improvement vs naive ===")
    print(f"MAE change: {metrics.mae - naive_metrics.mae:+.6f} (negative is better)")
    print(f"RMSE change: {metrics.rmse - naive_metrics.rmse:+.6f} (negative is better)")
    if metrics.mape is not None and naive_metrics.mape is not None:
        print(f"MAPE change: {metrics.mape - naive_metrics.mape:+.6f} percentage points (negative is better)")
    if metrics.smape is not None and naive_metrics.smape is not None:
        print(f"sMAPE change: {metrics.smape - naive_metrics.smape:+.6f} percentage points (negative is better)")

    print("\nCases:")
    for case in cases:
        print(
            f"  cutoff={case.cutoff_timestamp.isoformat()} "
            f"cutoff_close={case.cutoff_close:.6f} "
            f"actual={case.actual} predicted={case.predicted}"
        )


if __name__ == "__main__":
    main()
