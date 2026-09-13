from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.forecasting.price_loader import load_close_observations


def test_load_close_observations_returns_chronological_prices():
    db = MagicMock()
    db.scalar.return_value = "stock-id"
    db.execute.return_value.all.return_value = [
        SimpleNamespace(timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc), close=101),
        SimpleNamespace(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), close=100),
    ]

    observations = load_close_observations(db, "reliance", limit=2)

    assert [item.close for item in observations] == [100.0, 101.0]
    assert observations[0].timestamp < observations[1].timestamp
    db.scalar.assert_called_once()
    db.execute.assert_called_once()


def test_load_close_observations_rejects_unknown_symbol():
    db = MagicMock()
    db.scalar.return_value = None

    with pytest.raises(ValueError, match="unknown stock symbol"):
        load_close_observations(db, "UNKNOWN")


def test_load_close_observations_validates_limit():
    db = MagicMock()

    with pytest.raises(ValueError, match="limit must be >= 2"):
        load_close_observations(db, "RELIANCE", limit=1)
