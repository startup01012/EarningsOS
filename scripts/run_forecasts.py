from __future__ import annotations

import argparse
import os

from services.ml_worker.runner import run_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run validated pretrained EarningsOS forecasts as an offline batch job."
    )
    parser.add_argument(
        "--models",
        default=os.getenv("FORECAST_MODELS", "chronos"),
        help="Comma-separated model names: chronos,timesfm,kronos,fincast",
    )
    parser.add_argument(
        "--symbols",
        default=os.getenv(
            "FORECAST_SYMBOLS",
            "ADANIENT,AXISBANK,HDFCBANK,INFY,RELIANCE",
        ),
        help="Comma-separated NSE symbols",
    )
    parser.add_argument(
        "--horizons",
        default=os.getenv("FORECAST_HORIZONS", "1,5,10"),
        help="Comma-separated forecast horizons",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=int(os.getenv("FORECAST_CONTEXT", "256")),
    )
    args = parser.parse_args()

    models = [value.strip().lower() for value in args.models.split(",") if value.strip()]
    symbols = [value.strip().upper() for value in args.symbols.split(",") if value.strip()]
    horizons = [int(value) for value in args.horizons.split(",") if value.strip()]

    if not models or not symbols or not horizons:
        raise SystemExit("models, symbols and horizons must not be empty")

    print(
        f"Starting pretrained batch inference: models={models}, "
        f"symbols={len(symbols)}, horizons={horizons}, context={args.context}"
    )

    for model in models:
        if model == "fincast" and not os.getenv("FINCAST_MODEL_PATH"):
            print("Skipping fincast: FINCAST_MODEL_PATH is not configured")
            continue

        summary = run_model(
            model_name=model,
            symbols=symbols,
            horizons=horizons,
            context=args.context,
        )
        print(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
