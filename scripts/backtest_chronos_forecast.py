from __future__ import annotations

import argparse

from apps.api.db.session import SessionLocal
from services.forecasting.backtest import BacktestCase, run_backtest, evaluate_cases
from services.forecasting.benchmarks import evaluate_predictions, naive_last_close_predictions
from services.forecasting.chronos import Chronos2Adapter
from services.forecasting.price_loader import load_close_observations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a strictly causal rolling backtest with pretrained Chronos-2"
    )
    parser.add_argument("--symbol", default="RELIANCE")
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=1000)
    parser.add_argument("--device-map", default="cpu")
    args = parser.parse_args()

    if args.start_case < 0:
        raise ValueError("--start-case must be >= 0")
    if args.max_cases < 1:
        raise ValueError("--max-cases must be >= 1")
    minimum = args.context_length + args.horizon
    if args.history_limit < minimum:
        raise ValueError(f"--history-limit must be >= {minimum}")

    db = SessionLocal()
    try:
        observations = load_close_observations(
            db,
            args.symbol,
            interval=args.interval,
            source=args.source,
            limit=args.history_limit,
        )
    finally:
        db.close()

    adapter = Chronos2Adapter(device_map=args.device_map)
    cases, metrics = run_backtest(
        adapter,
        args.symbol,
        observations,
        context_length=args.context_length,
        horizon=args.horizon,
        stride=args.stride,
        start_case=args.start_case,
        max_cases=args.max_cases,
    )

    naive_predictions = naive_last_close_predictions(cases)
    naive_metrics = evaluate_predictions(cases, naive_predictions)

    print(f"Model: {adapter.model_id}")
    print(f"Symbol: {args.symbol.strip().upper()}")
    print(f"Available observations: {len(observations)}")
    print(f"Start case: {args.start_case}")
    print(f"Cases: {metrics.cases}")
    print(f"Horizon: {metrics.horizons}")
    print(f"Context length: {args.context_length}")
    print("\n=== Chronos-2 ===")
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

    print("\n=== Chronos improvement vs naive ===")
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
