from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


class MarketBar(BaseModel):
    symbol: str
    exchange: str
    timestamp: datetime
    interval: str

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal

    volume: int | None = None
    traded_value: Decimal | None = None

    source: str