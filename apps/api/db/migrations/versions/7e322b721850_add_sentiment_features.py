"""add persisted sentiment features

Revision ID: 7e322b721850
Revises: 120418304c98
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "7e322b721850"
down_revision = "120418304c98"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sentiment_features",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stock_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("feature_date", sa.Date(), nullable=False),
        sa.Column("model_name", sa.String(length=150), nullable=False),
        sa.Column("article_count", sa.Integer(), nullable=False),
        sa.Column("positive_ratio", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("negative_ratio", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("neutral_ratio", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("mean_score", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("sentiment_balance", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("last_published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("article_count_3d", sa.Integer(), nullable=False),
        sa.Column("article_count_7d", sa.Integer(), nullable=False),
        sa.Column("article_count_14d", sa.Integer(), nullable=False),
        sa.Column("article_count_30d", sa.Integer(), nullable=False),
        sa.Column("sentiment_balance_3d", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("sentiment_balance_7d", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("sentiment_balance_14d", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("sentiment_balance_30d", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("sentiment_momentum_3d_vs_7d", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["stock_id"], ["stocks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("stock_id", "feature_date", "model_name", name="uq_sentiment_feature"),
    )
    op.create_index("ix_sentiment_features_stock_id", "sentiment_features", ["stock_id"])
    op.create_index("ix_sentiment_features_symbol", "sentiment_features", ["symbol"])
    op.create_index("ix_sentiment_features_feature_date", "sentiment_features", ["feature_date"])


def downgrade() -> None:
    op.drop_index("ix_sentiment_features_feature_date", table_name="sentiment_features")
    op.drop_index("ix_sentiment_features_symbol", table_name="sentiment_features")
    op.drop_index("ix_sentiment_features_stock_id", table_name="sentiment_features")
    op.drop_table("sentiment_features")
