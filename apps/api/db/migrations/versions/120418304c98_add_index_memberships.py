"""add index memberships

Revision ID: 120418304c98
Revises: a3b25cd8ea97
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "120418304c98"
down_revision: Union[str, Sequence[str], None] = "a3b25cd8ea97"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "index_memberships",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stock_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("index_name", sa.String(length=100), nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=False),
        sa.Column("effective_to", sa.Date(), nullable=True),
        sa.Column("source", sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(["stock_id"], ["stocks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "stock_id", "index_name", "effective_from", name="uq_index_membership"
        ),
    )
    op.create_index(
        "ix_index_memberships_stock_id", "index_memberships", ["stock_id"], unique=False
    )
    op.create_index(
        "ix_index_memberships_index_name", "index_memberships", ["index_name"], unique=False
    )
    op.create_index(
        "ix_index_memberships_effective_from",
        "index_memberships",
        ["effective_from"],
        unique=False,
    )
    op.create_index(
        "ix_index_memberships_effective_to",
        "index_memberships",
        ["effective_to"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_index_memberships_effective_to", table_name="index_memberships")
    op.drop_index("ix_index_memberships_effective_from", table_name="index_memberships")
    op.drop_index("ix_index_memberships_index_name", table_name="index_memberships")
    op.drop_index("ix_index_memberships_stock_id", table_name="index_memberships")
    op.drop_table("index_memberships")
