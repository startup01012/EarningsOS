from __future__ import annotations

import argparse

from sqlalchemy import select

from apps.api.db.models import NewsArticle, SentimentScore, Stock
from apps.api.db.session import SessionLocal
from services.sentiment.features import build_daily_features, build_rolling_features
from services.sentiment.persistence import persist_sentiment_features


def load_scored_news(symbol: str | None = None) -> list[dict]:
    db = SessionLocal()
    try:
        stmt = (
            select(
                Stock.symbol,
                NewsArticle.published_at,
                SentimentScore.model_name,
                SentimentScore.label,
                SentimentScore.score,
                SentimentScore.positive_probability,
                SentimentScore.negative_probability,
                SentimentScore.neutral_probability,
            )
            .join(NewsArticle, SentimentScore.article_id == NewsArticle.id)
            .join(Stock, NewsArticle.stock_id == Stock.id)
            .order_by(NewsArticle.published_at.asc())
        )
        if symbol:
            stmt = stmt.where(Stock.symbol == symbol.strip().upper())

        rows = []
        for row in db.execute(stmt):
            rows.append(
                {
                    "symbol": row.symbol,
                    "published_at": row.published_at,
                    "model_name": row.model_name,
                    "label": row.label,
                    "score": float(row.score),
                    "positive_probability": float(row.positive_probability) if row.positive_probability is not None else None,
                    "negative_probability": float(row.negative_probability) if row.negative_probability is not None else None,
                    "neutral_probability": float(row.neutral_probability) if row.neutral_probability is not None else None,
                }
            )
        return rows
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist causal sentiment features")
    parser.add_argument("--symbol")
    args = parser.parse_args()

    rows = load_scored_news(symbol=args.symbol)
    print(f"Scored news rows: {len(rows)}")

    daily = build_daily_features(rows)
    rolling = build_rolling_features(daily)
    print(f"Daily feature rows: {len(daily)}")
    print(f"Rolling feature rows: {len(rolling)}")

    if not rolling:
        print("Persisted: 0 inserted, 0 updated")
        return

    inserted, updated = persist_sentiment_features(daily, rolling)
    print(f"Persisted: {inserted} inserted, {updated} updated")

    print("\nLatest rolling features:")
    for feature in rolling[-10:]:
        print(
            f"{feature.feature_date} {feature.symbol} "
            f"articles_7d={feature.article_count_7d} "
            f"balance_7d={feature.sentiment_balance_7d:.4f} "
            f"momentum={feature.sentiment_momentum_3d_vs_7d:.4f}"
        )


if __name__ == "__main__":
    main()
