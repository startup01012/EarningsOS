from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from db.models import (
    EarningsEvent,
    Forecast,
    ModelRegistry,
    ModelRun,
    NewsArticle,
    PriceBar,
    SentimentScore,
    Stock,
)
from db.session import get_session

router = APIRouter(prefix="/api/v1", tags=["intelligence"])

MODEL_CATALOG = [
    {
        "key": "chronos-2",
        "name": "Chronos-2",
        "model_id": "amazon/chronos-2",
        "task": "probabilistic time-series forecasting",
        "status": "adapter_ready",
        "pretrained_only": True,
        "production_gate": "failed_current_screen",
    },
    {
        "key": "timesfm-2.5",
        "name": "TimesFM 2.5",
        "model_id": "google/timesfm-2.5-200m-pytorch",
        "task": "probabilistic time-series forecasting",
        "status": "adapter_ready",
        "pretrained_only": True,
        "production_gate": "failed_current_screen",
    },
    {
        "key": "kronos-small",
        "name": "Kronos-small",
        "model_id": "NeoQuasar/Kronos-small",
        "task": "financial OHLCV forecasting",
        "status": "adapter_ready",
        "pretrained_only": True,
        "production_gate": "screen_required",
    },
    {
        "key": "fincast-v1",
        "name": "FinCast v1",
        "model_id": "Vincent05R/FinCast:v1",
        "task": "financial time-series forecasting",
        "status": "adapter_ready",
        "pretrained_only": True,
        "production_gate": "checkpoint_required",
    },
]


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _float(value: Decimal | int | float | None) -> float | None:
    return None if value is None else float(value)


def _baseline(db, symbol: str, as_of: datetime | None) -> dict:
    if as_of is None:
        return {
            "available": False,
            "score": None,
            "label": None,
            "article_count": 0,
            "model": "ProsusAI/finbert",
            "boundary": None,
            "method": "result_announcement_datetime required; no current-time fallback",
        }

    rows = db.execute(
        select(NewsArticle, SentimentScore)
        .join(SentimentScore, SentimentScore.article_id == NewsArticle.id)
        .join(Stock, Stock.id == NewsArticle.stock_id)
        .where(
            Stock.symbol == symbol,
            NewsArticle.published_at < as_of,
            SentimentScore.model_name == "ProsusAI/finbert",
        )
        .order_by(NewsArticle.published_at.desc())
        .limit(500)
    ).all()

    if not rows:
        return {
            "available": False,
            "score": None,
            "label": None,
            "article_count": 0,
            "model": "ProsusAI/finbert",
            "boundary": _iso(as_of),
            "method": "published_at < boundary; no result-timestamp information",
        }

    positive = sum(float(score.positive_probability or 0) for _, score in rows) / len(rows)
    negative = sum(float(score.negative_probability or 0) for _, score in rows) / len(rows)
    neutral = sum(float(score.neutral_probability or 0) for _, score in rows) / len(rows)
    balance = positive - negative
    score = max(0.0, min(100.0, 50.0 + 50.0 * balance))
    label = "positive" if balance > 0.08 else "negative" if balance < -0.08 else "neutral"

    return {
        "available": True,
        "score": round(score, 2),
        "label": label,
        "article_count": len(rows),
        "positive_probability": round(positive, 4),
        "negative_probability": round(negative, 4),
        "neutral_probability": round(neutral, 4),
        "model": "ProsusAI/finbert",
        "boundary": _iso(as_of),
        "method": "mean FinBERT probabilities with published_at < boundary",
    }


def _latest_price(db, symbol: str) -> dict | None:
    row = db.execute(
        select(PriceBar)
        .join(Stock, Stock.id == PriceBar.stock_id)
        .where(Stock.symbol == symbol, PriceBar.interval == "1d")
        .order_by(PriceBar.timestamp.desc())
        .limit(1)
    ).scalar_one_or_none()
    if row is None:
        return None
    return {
        "timestamp": _iso(row.timestamp),
        "open": _float(row.open),
        "high": _float(row.high),
        "low": _float(row.low),
        "close": _float(row.close),
        "volume": row.volume,
        "source": row.source,
    }


def _forecasts(db, symbol: str) -> list[dict]:
    rows = db.execute(
        select(Forecast, ModelRegistry)
        .join(ModelRun, ModelRun.id == Forecast.model_run_id)
        .join(ModelRegistry, ModelRegistry.id == ModelRun.model_id)
        .join(Stock, Stock.id == Forecast.stock_id)
        .where(Stock.symbol == symbol)
        .order_by(Forecast.generated_at.desc(), Forecast.target_time.asc())
        .limit(100)
    ).all()

    return [
        {
            "model": model.model_key,
            "model_name": model.name,
            "generated_at": _iso(forecast.generated_at),
            "target_time": _iso(forecast.target_time),
            "horizon": forecast.horizon,
            "predicted_value": _float(forecast.predicted_value),
            "lower_bound": _float(forecast.lower_bound),
            "upper_bound": _float(forecast.upper_bound),
            "direction": forecast.direction,
            "confidence": _float(forecast.confidence),
        }
        for forecast, model in rows
    ]


@router.get("/models")
def models() -> dict:
    return {"models": MODEL_CATALOG, "policy": "pretrained_only"}


@router.get("/market/{symbol}")
def market(symbol: str) -> dict:
    symbol = symbol.strip().upper()
    db = get_session()
    try:
        stock = db.scalar(select(Stock).where(Stock.symbol == symbol))
        if stock is None:
            raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")
        return {
            "symbol": symbol,
            "company_name": stock.company_name,
            "sector": stock.sector,
            "industry": stock.industry,
            "bar": _latest_price(db, symbol),
        }
    finally:
        db.close()


@router.get("/watchlist")
def watchlist(
    symbols: str = Query(
        "RELIANCE,HDFCBANK,INFY,AXISBANK",
        description="Comma-separated NSE symbols",
    ),
) -> dict:
    requested = [item.strip().upper() for item in symbols.split(",") if item.strip()]
    db = get_session()
    try:
        result = []
        for symbol in requested:
            stock = db.scalar(select(Stock).where(Stock.symbol == symbol))
            if stock is None:
                continue
            bar = _latest_price(db, symbol)
            result.append(
                {
                    "symbol": symbol,
                    "company_name": stock.company_name,
                    "price": bar["close"] if bar else None,
                    "timestamp": bar["timestamp"] if bar else None,
                    "source": bar["source"] if bar else None,
                }
            )
        return {"items": result, "source": "EarningsOS database"}
    finally:
        db.close()


@router.get("/intelligence/{symbol}")
def intelligence(
    symbol: str,
    as_of: datetime | None = Query(
        None,
        description="Information boundary. Historical consumers must pass the result boundary.",
    ),
) -> dict:
    symbol = symbol.strip().upper()
    requested_boundary = as_of
    if requested_boundary is not None and requested_boundary.tzinfo is None:
        requested_boundary = requested_boundary.replace(tzinfo=timezone.utc)

    db = get_session()
    try:
        stock = db.scalar(select(Stock).where(Stock.symbol == symbol))
        if stock is None:
            raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")

        now = datetime.now(timezone.utc)
        if requested_boundary is not None:
            event = db.execute(
                select(EarningsEvent)
                .where(
                    EarningsEvent.stock_id == stock.id,
                    EarningsEvent.result_announcement_datetime.is_not(None),
                    EarningsEvent.result_announcement_datetime <= requested_boundary,
                )
                .order_by(EarningsEvent.result_announcement_datetime.desc())
                .limit(1)
            ).scalar_one_or_none()
            boundary = event.result_announcement_datetime if event else requested_boundary
        else:
            # Default intelligence is anchored to the nearest upcoming result.
            # If no future event exists, use the latest announced event.
            event = db.execute(
                select(EarningsEvent)
                .where(
                    EarningsEvent.stock_id == stock.id,
                    EarningsEvent.result_announcement_datetime.is_not(None),
                    EarningsEvent.result_announcement_datetime > now,
                )
                .order_by(EarningsEvent.result_announcement_datetime.asc())
                .limit(1)
            ).scalar_one_or_none()
            if event is None:
                event = db.execute(
                    select(EarningsEvent)
                    .where(
                        EarningsEvent.stock_id == stock.id,
                        EarningsEvent.result_announcement_datetime.is_not(None),
                    )
                    .order_by(EarningsEvent.result_announcement_datetime.desc())
                    .limit(1)
                ).scalar_one_or_none()
            boundary = event.result_announcement_datetime if event else None

        forecasts = _forecasts(db, symbol)
        return {
            "symbol": symbol,
            "company_name": stock.company_name,
            "market": _latest_price(db, symbol),
            "baseline": _baseline(db, symbol, boundary),
            "forecasting": {
                "available": bool(forecasts),
                "pretrained_only": True,
                "stored_forecasts": forecasts,
            },
            "earnings_event": (
                {
                    "event_key": event.event_key,
                    "fiscal_period": event.fiscal_period,
                    "period_ended": event.period_ended.isoformat(),
                    "event_date": event.event_date.isoformat(),
                    "result_announcement_datetime": _iso(event.result_announcement_datetime),
                    "announced_at": _iso(event.announced_at),
                    "document_count": event.document_count,
                    "has_financial_results": event.has_financial_results,
                    "has_media_release": event.has_media_release,
                    "has_earnings_call": event.has_earnings_call,
                    "has_transcript": event.has_transcript,
                    "eps_actual": _float(event.eps_actual),
                    "eps_estimate": _float(event.eps_estimate),
                    "revenue_actual": _float(event.revenue_actual),
                    "revenue_estimate": _float(event.revenue_estimate),
                    "source": event.source,
                }
                if event
                else None
            ),
            "data_boundary": _iso(boundary),

            "leakage_policy": "The default data_boundary is the selected event's result_announcement_datetime. News is filtered by published_at < data_boundary. Post-result material must not be used for pre-result features.",
        }
    finally:
        db.close()


@router.get("/forecast/{symbol}")
def stored_forecast(
    symbol: str,
    model: str | None = None,
    horizon: str | None = None,
) -> dict:
    symbol = symbol.strip().upper()
    db = get_session()
    try:
        rows = _forecasts(db, symbol)
        if model:
            rows = [row for row in rows if row["model"] == model]
        if horizon:
            rows = [row for row in rows if row["horizon"] == horizon]
        return {
            "symbol": symbol,
            "available": bool(rows),
            "items": rows,
            "execution_policy": "Forecast endpoints serve stored, validated forecasts. Heavy model inference is not triggered by a web request.",
        }
    finally:
        db.close()
