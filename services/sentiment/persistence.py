from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select

from apps.api.db.models import SentimentFeature, Stock
from apps.api.db.session import SessionLocal

from .features import DailySentimentFeature, RollingSentimentFeature


def persist_sentiment_features(
    daily: Sequence[DailySentimentFeature],
    rolling: Sequence[RollingSentimentFeature],
) -> tuple[int, int]:
    """Persist daily and rolling features with an idempotent natural key.

    The database uniqueness constraint on (stock_id, feature_date, model_name)
    is the final guard against duplicate rows. Existing rows are updated so a
    rerun can safely rebuild features when upstream news scores change.
    """
    rolling_by_key = {
        (item.symbol, item.feature_date, item.model_name): item for item in rolling
    }

    db = SessionLocal()
    inserted = updated = 0
    try:
        symbols = {item.symbol for item in daily}
        stocks = {
            stock.symbol: stock
            for stock in db.scalars(select(Stock).where(Stock.symbol.in_(symbols))).all()
        }

        missing = sorted(symbols - stocks.keys())
        if missing:
            raise ValueError(f"No Stock rows found for symbols: {', '.join(missing)}")

        for daily_item in daily:
            rolling_item = rolling_by_key.get(
                (daily_item.symbol, daily_item.feature_date, daily_item.model_name)
            )
            if rolling_item is None:
                raise ValueError(
                    "Missing rolling feature for "
                    f"{daily_item.symbol} {daily_item.feature_date} {daily_item.model_name}"
                )

            existing = db.scalar(
                select(SentimentFeature).where(
                    SentimentFeature.stock_id == stocks[daily_item.symbol].id,
                    SentimentFeature.feature_date == daily_item.feature_date,
                    SentimentFeature.model_name == daily_item.model_name,
                )
            )

            values = {
                "stock_id": stocks[daily_item.symbol].id,
                "symbol": daily_item.symbol,
                "feature_date": daily_item.feature_date,
                "model_name": daily_item.model_name,
                "article_count": daily_item.article_count,
                "positive_ratio": daily_item.positive_ratio,
                "negative_ratio": daily_item.negative_ratio,
                "neutral_ratio": daily_item.neutral_ratio,
                "mean_score": daily_item.mean_score,
                "sentiment_balance": daily_item.sentiment_balance,
                "last_published_at": daily_item.last_published_at,
                "article_count_3d": rolling_item.article_count_3d,
                "article_count_7d": rolling_item.article_count_7d,
                "article_count_14d": rolling_item.article_count_14d,
                "article_count_30d": rolling_item.article_count_30d,
                "sentiment_balance_3d": rolling_item.sentiment_balance_3d,
                "sentiment_balance_7d": rolling_item.sentiment_balance_7d,
                "sentiment_balance_14d": rolling_item.sentiment_balance_14d,
                "sentiment_balance_30d": rolling_item.sentiment_balance_30d,
                "sentiment_momentum_3d_vs_7d": rolling_item.sentiment_momentum_3d_vs_7d,
            }

            if existing is None:
                db.add(SentimentFeature(**values))
                inserted += 1
            else:
                for key, value in values.items():
                    setattr(existing, key, value)
                updated += 1

        db.commit()
        return inserted, updated
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
