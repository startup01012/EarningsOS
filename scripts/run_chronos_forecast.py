from __future__ import annotations

import argparse

from apps.api.db.session import SessionLocal
from services.forecasting.chronos import Chronos2Adapter
from services.forecasting.price_loader import load_close_observations
from services.forecasting.price_series import build_forecast_request


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pretrained Amazon Chronos-2 on causal PostgreSQL price data"
    )
    parser.add_argument("--symbol", default="RELIANCE")
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--device-map",
        default="cpu",
        help="Chronos device map, e.g. cpu or cuda",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        observations = load_close_observations(
            db,
            args.symbol,
            interval=args.interval,
            source=args.source,
            limit=args.limit,
        )
    finally:
        db.close()

    request = build_forecast_request(args.symbol, observations, args.horizon)
    adapter = Chronos2Adapter(device_map=args.device_map)
    result = adapter.forecast(request)

    print(f"Model: {result.model_id}")
    print(f"Symbol: {result.symbol}")
    print(f"Observations: {len(request.values)}")
    print(f"Input last timestamp: {request.timestamps[-1].isoformat()}")
    print(f"Input last close: {request.values[-1]:.6f}")
    print(f"Forecast horizon: {result.horizon}")
    print(f"Generated at: {result.generated_at.isoformat()}")
    print("Forecast:")
    for step, (median, lower, upper) in enumerate(
        zip(result.median, result.lower or [], result.upper or []), start=1
    ):
        print(
            f"  t+{step}: median={median:.6f} "
            f"lower={lower:.6f} upper={upper:.6f}"
        )


if __name__ == "__main__":
    main()
