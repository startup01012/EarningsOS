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


def _columns(table_name: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table_name)}


def upgrade() -> None:
    existing = _columns("earnings_events")
    columns = [
        ("event_key", sa.String(100)),
        ("period_ended", sa.Date()),
        ("result_announcement_datetime", sa.DateTime(timezone=True)),
        ("announcement_date", sa.Date()),
        ("announcement_time", sa.String(20)),
        ("fiscal_quarter", sa.String(20)),
        ("fiscal_year", sa.Integer()),
        ("event_status", sa.String(40)),
        ("document_count", sa.Integer()),
        ("has_financial_results", sa.Boolean()),
        ("has_media_release", sa.Boolean()),
        ("has_earnings_call", sa.Boolean()),
        ("has_transcript", sa.Boolean()),
        ("first_document_datetime", sa.DateTime(timezone=True)),
        ("last_document_datetime", sa.DateTime(timezone=True)),
    ]
    for name, typ in columns:
        if name not in existing:
            op.add_column("earnings_events", sa.Column(name, typ, nullable=True))

    op.execute(sa.text("""
        UPDATE earnings_events
        SET period_ended = COALESCE(period_ended, event_date),
            result_announcement_datetime = COALESCE(result_announcement_datetime, announced_at),
            announcement_date = COALESCE(announcement_date, CAST(announced_at AS DATE)),
            announcement_time = COALESCE(
                announcement_time,
                CASE WHEN announced_at IS NULL THEN NULL ELSE CAST(announced_at AS TIME)::text END
            ),
            event_status = COALESCE(
                event_status,
                CASE WHEN announced_at IS NULL THEN 'identified' ELSE 'result_announced' END
            ),
            document_count = COALESCE(document_count, 0),
            has_financial_results = COALESCE(has_financial_results, FALSE),
            has_media_release = COALESCE(has_media_release, FALSE),
            has_earnings_call = COALESCE(has_earnings_call, FALSE),
            has_transcript = COALESCE(has_transcript, FALSE)
    """))
    op.execute(sa.text("""
        UPDATE earnings_events e
        SET event_key = COALESCE(e.event_key, s.symbol || '_' || e.period_ended::text)
        FROM stocks s
        WHERE e.stock_id = s.id AND e.event_key IS NULL
    """))

    op.execute(sa.text("CREATE UNIQUE INDEX IF NOT EXISTS ix_earnings_events_event_key ON earnings_events(event_key)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_earnings_events_period_ended ON earnings_events(period_ended)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_earnings_events_result_announcement_datetime ON earnings_events(result_announcement_datetime)"))

    inspector = sa.inspect(op.get_bind())
    if "earnings_documents" not in inspector.get_table_names():
        op.create_table(
            "earnings_documents",
            sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("document_id", sa.String(200), nullable=False),
            sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("event_key", sa.String(100), nullable=False),
            sa.Column("stock_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("symbol", sa.String(30), nullable=False),
            sa.Column("period_ended", sa.Date(), nullable=False),
            sa.Column("announcement_datetime", sa.DateTime(timezone=True)),
            sa.Column("document_type", sa.String(80), nullable=False),
            sa.Column("document_subtype", sa.String(120)),
            sa.Column("announcement_text", sa.Text()),
            sa.Column("filing_url", sa.Text()),
            sa.Column("is_primary_result", sa.Boolean(), nullable=False),
            sa.Column("is_followup_document", sa.Boolean(), nullable=False),
            sa.Column("source", sa.String(100)),
            sa.ForeignKeyConstraint(["event_id"], ["earnings_events.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["stock_id"], ["stocks.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("document_id"),
        )
    for name, table, column in [
        ("ix_earnings_documents_document_id", "earnings_documents", "document_id"),
        ("ix_earnings_documents_event_id", "earnings_documents", "event_id"),
        ("ix_earnings_documents_event_key", "earnings_documents", "event_key"),
        ("ix_earnings_documents_stock_id", "earnings_documents", "stock_id"),
        ("ix_earnings_documents_symbol", "earnings_documents", "symbol"),
        ("ix_earnings_documents_period_ended", "earnings_documents", "period_ended"),
        ("ix_earnings_documents_announcement_datetime", "earnings_documents", "announcement_datetime"),
    ]:
        op.execute(sa.text(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({column})"))


def downgrade() -> None:
    op.drop_table("earnings_documents")
    for name in [
        "ix_earnings_events_result_announcement_datetime",
        "ix_earnings_events_period_ended",
        "ix_earnings_events_event_key",
    ]:
        op.execute(sa.text(f"DROP INDEX IF EXISTS {name}"))
    for column in [
        "last_document_datetime","first_document_datetime","has_transcript",
        "has_earnings_call","has_media_release","has_financial_results",
        "document_count","event_status","fiscal_year","fiscal_quarter",
        "announcement_time","announcement_date","result_announcement_datetime",
        "period_ended","event_key",
    ]:
        if column in _columns("earnings_events"):
            op.drop_column("earnings_events", column)
