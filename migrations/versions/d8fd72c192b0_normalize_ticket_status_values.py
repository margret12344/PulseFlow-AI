"""normalize ticket status values

Revision ID: d8fd72c192b0
Revises: 76d948c3eddb
Create Date: 2026-11-18 18:42:31.315866

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd8fd72c192b0'
down_revision: str | Sequence[str] | None = '76d948c3eddb'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE tickets
        SET status = CASE status
            WHEN 'open' THEN 'Open'
            WHEN 'assigned' THEN 'Assigned'
            WHEN 'in progress' THEN 'In Progress'
            WHEN 'resolved' THEN 'Resolved'
            WHEN 'closed' THEN 'Closed'
        END
        WHERE status IN (
            'open',
            'assigned',
            'in progress',
            'resolved',
            'closed'
        )
        """
    )


def downgrade() -> None:
    pass