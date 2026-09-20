from __future__ import annotations

import hashlib
import math
import os
from datetime import datetime, timedelta, timezone
from time import perf_counter

from sqlalchemy import select

from apps.api.db.models import Forecast, ModelRegistry, ModelRun, PriceBar, Stock
from apps.api.db.session import SessionLocal
from services.forecasting import (
    Chronos2Adapter,
    FinCastAdapter,
    ForecastAdapter,
    ForecastRequest,
    KronosSmallAdapter,
    TimesFM25Adapter,
)


MODEL_META = {
    "amazon/chronos-2": {
        "name": "Chronos-2",
        "version": "2",
        "provider": "Amazon",
        "source_url": "https://huggingface.co/amazon/chronos-2",
        "license": "Apache-2.0",
    },
    "google/timesfm-2.5-200m-pytorch": {
        "name": "TimesFM 2.5",
        "version": "2.5",
        "provider": "Google Research",
        "source_url": "https://huggingface.co/google/timesfm-2.5-200m-pytorch",
        "license": "Apache-2.0",
    },
    "NeoQuasar/Kronos-small": {
        "name": "Kronos-small",
        "version": "small",
        "provider": "NeoQuasar",
        "source_url": "https://huggingface.co/NeoQuasar/Kronos-small",
        "license": "MIT",
    },
    "Vincent05R/FinCast:v1": {
        "name": "FinCast v1",
        "version": "v1",
        "provider": "Vincent05R",
        "source_url": "https://huggingface.co/Vincent05R/FinCast",
        "license": "Apache-2.0",
    },
}


def _adapter(name: str) -> ForecastAdapter:
    if name == "chronos":
        return Chronos2Adapter(device_map=os.getenv("CHRONOS_DEVICE", "cpu"))
    if name == "timesfm":
        return TimesFM25Adapter(device=os.getenv("TIMESFM_DEVICE", "cpu"))
    if name == "kronos":
        return KronosSmallAdapter(
            device=os.getenv("KRONOS_DEVICE") or None,
            seed=int(os.getenv("KRONOS_SEED", "7")),
        )
    if name == "fincast":
        return FinCastAdapter(device=os.getenv("FINCAST_DEVICE", "cpu"))
    raise ValueError(f"unknown forecasting model: {name}")


def _model_key(name: str) -> str:
    return {
        "chronos": "amazon/chronos-2",
        "timesfm": "google/timesfm-2.5-200m-pytorch",
        "kronos": "NeoQuasar/Kronos-small",
        "fincast": "Vincent05R/FinCast:v1",
    }[name]


def _business_days_after(value: datetime, count: int) -> datetime:
    current = value
    remaining = count
    while remaining:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def _input_hash(request: ForecastRequest) -> str:
    payload = "|".join(
        [
            request.symbol,
            str(request.horizon),
            *(f"{ts.isoformat()}:{value:.8f}" for ts, value in zip(request.timestamps, request.values)),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_result(result, last_close: float) -> None:
    if len(result.median) != result.horizon:
        raise ValueError("forecast median length does not match horizon")

    series = result.median
    if any(not math.isfinite(value) or value <= 0 for value in series):
        raise ValueError("forecast contains non-finite or non-positive prices")

    if result.lower is not None and result.upper is not None:
        if len(result.lower) != result.horizon or len(result.upper) != result.horizon:
            raise ValueError("forecast interval length does not match horizon")
        for low, high, median in zip(result.lower, result.upper, series):
            if not all(math.isfinite(x) for x in (low, high, median)):
                raise ValueError("forecast interval contains non-finite values")
            if low > median or median > high:
                raise ValueError("forecast median is outside its prediction interval")

    if not math.isfinite(last_close) or last_close <= 0:
        raise ValueError("last close is invalid")


def _registry(db, model_key: str) -> ModelRegistry:
    row = db.scalar(select(ModelRegistry).where(ModelRegistry.model_key == model_key))
    meta = MODEL_META[model_key]
    if row is None:
        row = ModelRegistry(
            model_key=model_key,
            name=meta["name"],
            version=meta["version"],
            task="pretrained time-series forecasting",
            provider=meta["provider"],
            source_url=meta["source_url"],
            license=meta["license"],
            pretrained_only=True,
            active=True,
        )
        db.add(row)
        db.flush()
    return row


def _load_request(db, symbol: str, horizon: int, context: int = 256) -> ForecastRequest:
    stock = db.scalar(select(Stock).where(Stock.symbol == symbol))
    if stock is None:
        raise ValueError(f"unknown stock symbol: {symbol}")

    rows = list(
        db.execute(
            select(
                PriceBar.timestamp,
                PriceBar.open,
                PriceBar.high,
                PriceBar.low,
                PriceBar.close,
                PriceBar.volume,
                PriceBar.traded_value,
            )
            .where(
                PriceBar.stock_id == stock.id,
                PriceBar.interval == "1d",
                PriceBar.source == "yfinance",
            )
            .order_by(PriceBar.timestamp.desc())
            .limit(context)
        ).all()
    )
    rows.reverse()
    if len(rows) < 2:
        raise ValueError(f"insufficient yfinance price history for {symbol}")

    return ForecastRequest(
        symbol=symbol,
        values=[float(row.close) for row in rows],
        timestamps=[row.timestamp for row in rows],
        horizon=horizon,
        opens=[float(row.open) for row in rows],
        highs=[float(row.high) for row in rows],
        lows=[float(row.low) for row in rows],
        volumes=[float(row.volume or 0) for row in rows],
        amounts=[float(row.traded_value or 0) for row in rows],
    )


def run_model(
    *,
    model_name: str,
    symbols: list[str],
    horizons: list[int],
    context: int = 256,
) -> dict:
    adapter = _adapter(model_name)
    model_key = _model_key(model_name)
    db = SessionLocal()
    summary = {"model": model_name, "model_key": model_key, "success": 0, "failed": 0, "errors": []}

    try:
        registry = _registry(db, model_key)

        for symbol in symbols:
            for horizon in horizons:
                started = perf_counter()
                run = ModelRun(
                    model_id=registry.id,
                    started_at=datetime.now(timezone.utc),
                    status="running",
                    metadata_json={
                        "worker": "services.ml_worker.runner",
                        "pretrained_only": True,
                        "symbol": symbol,
                        "horizon": horizon,
                        "context": context,
                    },
                )
                db.add(run)
                db.flush()

                try:
                    request = _load_request(db, symbol, horizon, context)
                    run.input_hash = _input_hash(request)
                    result = adapter.forecast(request)
                    last_close = request.values[-1]
                    _validate_result(result, last_close)

                    generated_at = result.generated_at
                    for step, predicted in enumerate(result.median, start=1):
                        target_time = _business_days_after(request.timestamps[-1], step)
                        direction = (
                            "up"
                            if predicted > last_close
                            else "down"
                            if predicted < last_close
                            else "flat"
                        )
                        low = result.lower[step - 1] if result.lower else None
                        high = result.upper[step - 1] if result.upper else None

                        db.add(
                            Forecast(
                                model_run_id=run.id,
                                stock_id=db.scalar(
                                    select(Stock.id).where(Stock.symbol == symbol)
                                ),
                                generated_at=generated_at,
                                target_time=target_time,
                                horizon=f"h{horizon}",
                                predicted_value=predicted,
                                lower_bound=low,
                                upper_bound=high,
                                direction=direction,
                                confidence=None,
                            )
                        )

                    run.finished_at = datetime.now(timezone.utc)
                    run.status = "validated"
                    run.latency_ms = int((perf_counter() - started) * 1000)
                    db.commit()
                    summary["success"] += 1
                except Exception as exc:
                    db.rollback()
                    run = ModelRun(
                        model_id=registry.id,
                        started_at=datetime.now(timezone.utc),
                        finished_at=datetime.now(timezone.utc),
                        status="failed",
                        latency_ms=int((perf_counter() - started) * 1000),
                        metadata_json={
                            "worker": "services.ml_worker.runner",
                            "symbol": symbol,
                            "horizon": horizon,
                            "error": str(exc),
                        },
                    )
                    db.add(run)
                    db.commit()
                    summary["failed"] += 1
                    summary["errors"].append(
                        {"symbol": symbol, "horizon": horizon, "error": str(exc)}
                    )
    finally:
        db.close()

    return summary
