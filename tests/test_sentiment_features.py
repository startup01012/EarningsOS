from datetime import datetime, timezone

from services.sentiment.features import build_daily_features


def test_daily_features_use_published_date_and_preserve_probabilities():
    rows = [
        {
            "symbol": "TCS",
            "model_name": "ProsusAI/finbert",
            "published_at": datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc),
            "label": "positive",
            "score": 0.9,
            "positive_probability": 0.9,
            "negative_probability": 0.05,
            "neutral_probability": 0.05,
        },
        {
            "symbol": "TCS",
            "model_name": "ProsusAI/finbert",
            "published_at": datetime(2026, 1, 5, 11, 0, tzinfo=timezone.utc),
            "label": "negative",
            "score": 0.8,
            "positive_probability": 0.1,
            "negative_probability": 0.8,
            "neutral_probability": 0.1,
        },
    ]

    features = build_daily_features(rows)
    assert len(features) == 1
    feature = features[0]
    assert feature.symbol == "TCS"
    assert feature.feature_date.isoformat() == "2026-01-05"
    assert feature.article_count == 2
    assert feature.positive_ratio == 0.5
    assert feature.negative_ratio == 0.5
    assert feature.neutral_ratio == 0.0
    assert feature.mean_score == 0.85
    assert feature.sentiment_balance == (0.9 - 0.05 + 0.1 - 0.8) / 2
