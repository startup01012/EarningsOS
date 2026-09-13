from __future__ import annotations

import argparse

from apps.api.db.session import SessionLocal
from services.forecasting.backtest import run_backtest
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
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--device-map", default="cpu")
    args = parser.parse_args()

    required = args.context_length + args.horizon + (args.max_cases - 1) * args.stride
    db = SessionLocal()
    try:
        observations = load_close_observations(
            db,
            args.symbol,
            interval=args.interval,
            source=args.source,
            limit=required,
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
        max_cases=args.max_cases,
    )

    print(f"Model: {adapter.model_id}")
    print(f"Symbol: {args.symbol.strip().upper()}")
    print(f"Cases: {metrics.cases}")
    print(f"Horizon: {metrics.horizons}")
    print(f"Context length: {args.context_length}")
    print(f"MAE: {metrics.mae:.6f}")
    print(f"RMSE: {metrics.rmse:.6f}")
    print(f"MAPE: {metrics.mape:.6f}%" if metrics.mape is not None else "MAPE: N/A")
    print(
        f"Directional accuracy: {metrics.directional_accuracy:.2%}"
        if metrics.directional_accuracy is not None
        else "Directional accuracy: N/A"
    )
    print("Cases:")
    for case in cases:
        print(
            f"  cutoff={case.cutoff_timestamp.isoformat()} "
            f"actual={case.actual} predicted={case.predicted}"
        )


if __name__ == "__main__":
    main()
