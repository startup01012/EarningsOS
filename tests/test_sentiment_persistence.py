from apps.api.db.models import SentimentFeature


def test_sentiment_feature_has_idempotent_natural_key():
    constraint = next(
        constraint
        for constraint in SentimentFeature.__table__.constraints
        if constraint.name == "uq_sentiment_feature"
    )

    assert {column.name for column in constraint.columns} == {
        "stock_id",
        "feature_date",
        "model_name",
    }


def test_sentiment_feature_contains_daily_and_rolling_inputs():
    columns = set(SentimentFeature.__table__.columns.keys())

    expected = {
        "symbol",
        "feature_date",
        "model_name",
        "article_count",
        "positive_ratio",
        "negative_ratio",
        "neutral_ratio",
        "mean_score",
        "sentiment_balance",
        "article_count_3d",
        "article_count_7d",
        "article_count_14d",
        "article_count_30d",
        "sentiment_balance_3d",
        "sentiment_balance_7d",
        "sentiment_balance_14d",
        "sentiment_balance_30d",
        "sentiment_momentum_3d_vs_7d",
    }

    assert expected <= columns
