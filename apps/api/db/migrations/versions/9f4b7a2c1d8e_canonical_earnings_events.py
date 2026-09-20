"""canonical earnings events and related documents

Revision ID: 9f4b7a2c1d8e
Revises: 7e322b721850
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "9f4b7a2c1d8e"
down_revision = "7e322b721850"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("earnings_events", sa.Column("event_key", sa.String(length=100), nullable=True))
    op.add_column("earnings_events", sa.Column("period_ended", sa.Date(), nullable=True))
    op.add_column("earnings_events", sa.Column("result_announcement_datetime", sa.DateTime(timezone=True), nullable=True))
    op.add_column("earnings_events", sa.Column("announcement_date", sa.Date(), nullable=True))
    op.add_column("earnings_events", sa.Column("announcement_time", sa.String(length=20), nullable=True))
    op.add_column("earnings_events", sa.Column("fiscal_quarter", sa.String(length=20), nullable=True))
    op.add_column("earnings_events", sa.Column("fiscal_year", sa.Integer(), nullable=True))
    op.add_column("earnings_events", sa.Column("event_status", sa.String(length=40), nullable=True))
    op.add_column("earnings_events", sa.Column("document_count", sa.Integer(), nullable=True))
    op.add_column("earnings_events", sa.Column("has_financial_results", sa.Boolean(), nullable=True))
    op.add_column("earnings_events", sa.Column("has_media_release", sa.Boolean(), nullable=True))
    op.add_column("earnings_events", sa.Column("has_earnings_call", sa.Boolean(), nullable=True))
    op.add_column("earnings_events", sa.Column("has_transcript", sa.Boolean(), nullable=True))
    op.add_column("earnings_events", sa.Column("first_document_datetime", sa.DateTime(timezone=True), nullable=True))
    op.add_column("earnings_events", sa.Column("last_document_datetime", sa.DateTime(timezone=True), nullable=True))

    op.execute(sa.text("""
        UPDATE earnings_events
        SET period_ended = event_date,
            result_announcement_datetime = announced_at,
            announcement_date = CAST(announced_at AS DATE),
            announcement_time = CASE
                WHEN announced_at IS NULL THEN NULL
                ELSE CAST(announced_at AS TIME)::text
            END,
            event_status = CASE
                WHEN announced_at IS NULL THEN 'identified'
                ELSE 'result_announced'
            END,
            document_count = 0,
            has_financial_results = FALSE,
            has_media_release = FALSE,
            has_earnings_call = FALSE,
            has_transcript = FALSE
        WHERE period_ended IS NULL
    """))
    op.execute(sa.text("""
        UPDATE earnings_events e
        SET event_key = s.symbol || '_' || e.period_ended::text
        FROM stocks s
        WHERE e.stock_id = s.id
          AND e.event_key IS NULL
    """))

    op.alter_column("earnings_events", "period_ended", nullable=False)
    op.alter_column("earnings_events", "event_key", nullable=False)
    op.alter_column("earnings_events", "event_status", nullable=False, server_default="identified")
    op.alter_column("earnings_events", "document_count", nullable=False, server_default="0")
    op.alter_column("earnings_events", "has_financial_results", nullable=False, server_default=sa.false())
    op.alter_column("earnings_events", "has_media_release", nullable=False, server_default=sa.false())
    op.alter_column("earnings_events", "has_earnings_call", nullable=False, server_default=sa.false())
    op.alter_column("earnings_events", "has_transcript", nullable=False, server_default=sa.false())

    op.create_index("ix_earnings_events_event_key", "earnings_events", ["event_key"], unique=True)
    op.create_index("ix_earnings_events_period_ended", "earnings_events", ["period_ended"])
    op.create_index("ix_earnings_events_result_announcement_datetime", "earnings_events", ["result_announcement_datetime"])

    op.create_table(
        "earnings_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", sa.String(length=200), nullable=False),
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_key", sa.String(length=100), nullable=False),
        sa.Column("stock_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("period_ended", sa.Date(), nullable=False),
        sa.Column("announcement_datetime", sa.DateTime(timezone=True), nullable=True),
        sa.Column("document_type", sa.String(length=80), nullable=False),
        sa.Column("document_subtype", sa.String(length=120), nullable=True),
        sa.Column("announcement_text", sa.Text(), nullable=True),
        sa.Column("filing_url", sa.Text(), nullable=True),
        sa.Column("is_primary_result", sa.Boolean(), nullable=False),
        sa.Column("is_followup_document", sa.Boolean(), nullable=False),
        sa.Column("source", sa.String(length=100), nullable=True),
        sa.ForeignKeyConstraint(["event_id"], ["earnings_events.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["stock_id"], ["stocks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id"),
    )
    op.create_index("ix_earnings_documents_document_id", "earnings_documents", ["document_id"])
    op.create_index("ix_earnings_documents_event_id", "earnings_documents", ["event_id"])
    op.create_index("ix_earnings_documents_event_key", "earnings_documents", ["event_key"])
    op.create_index("ix_earnings_documents_stock_id", "earnings_documents", ["stock_id"])
    op.create_index("ix_earnings_documents_symbol", "earnings_documents", ["symbol"])
    op.create_index("ix_earnings_documents_period_ended", "earnings_documents", ["period_ended"])
    op.create_index("ix_earnings_documents_announcement_datetime", "earnings_documents", ["announcement_datetime"])


def downgrade() -> None:
    op.drop_index("ix_earnings_documents_announcement_datetime", table_name="earnings_documents")
    op.drop_index("ix_earnings_documents_period_ended", table_name="earnings_documents")
    op.drop_index("ix_earnings_documents_symbol", table_name="earnings_documents")
    op.drop_index("ix_earnings_documents_stock_id", table_name="earnings_documents")
    op.drop_index("ix_earnings_documents_event_key", table_name="earnings_documents")
    op.drop_index("ix_earnings_documents_event_id", table_name="earnings_documents")
    op.drop_index("ix_earnings_documents_document_id", table_name="earnings_documents")
    op.drop_table("earnings_documents")

    op.drop_index("ix_earnings_events_result_announcement_datetime", table_name="earnings_events")
    op.drop_index("ix_earnings_events_period_ended", table_name="earnings_events")
    op.drop_index("ix_earnings_events_event_key", table_name="earnings_events")
    for column in [
        "last_document_datetime",
        "first_document_datetime",
        "has_transcript",
        "has_earnings_call",
        "has_media_release",
        "has_financial_results",
        "document_count",
        "event_status",
        "fiscal_year",
        "fiscal_quarter",
        "announcement_time",
        "announcement_date",
        "result_announcement_datetime",
        "period_ended",
        "event_key",
    ]:
        op.drop_column("earnings_events", column)
