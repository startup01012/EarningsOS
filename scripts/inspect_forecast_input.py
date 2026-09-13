from __future__ import annotations

import argparse
from datetime import datetime

from apps.api.db.session import SessionLocal
from services.forecasting.price_loader import load_close_observations
from services.forecasting.price_series import build_forecast_request


def parse_as_of(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("--as-of must include a timezone offset")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a causal PostgreSQL price series before model inference"
    )
    parser.add_argument("--symbol", default="RELIANCE")
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--as-of", type=parse_as_of)
    args = parser.parse_args()

    db = SessionLocal()
    try:
        observations = load_close_observations(
            db,
            args.symbol,
            interval=args.interval,
            source=args.source,
            limit=args.limit,
            as_of=args.as_of,
        )
    finally:
        db.close()

    request = build_forecast_request(args.symbol, observations, args.horizon)
    print(f"Symbol: {request.symbol}")
    print(f"Observations: {len(request.values)}")
    print(f"First timestamp: {request.timestamps[0].isoformat()}")
    print(f"Last timestamp: {request.timestamps[-1].isoformat()}")
    print(f"Last close: {request.values[-1]:.6f}")
    print(f"Forecast horizon: {request.horizon}")
    if args.as_of is not None:
        print(f"As-of boundary: {args.as_of.isoformat()}")
    print("Forecast input validation: OK")


if __name__ == "__main__":
    main()
