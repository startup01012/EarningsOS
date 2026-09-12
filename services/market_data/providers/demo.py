from datetime import datetime, timezone
from decimal import Decimal

from ..base import MarketDataProvider
from ..schemas import MarketBar


class DemoMarketDataProvider(MarketDataProvider):
    """
    Temporary provider used to validate the market-data architecture.

    This does NOT provide real market data.
    """

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> list[MarketBar]:

        return [
            MarketBar(
                symbol=symbol,
                exchange="NSE",
                timestamp=datetime.now(timezone.utc),
                interval=interval,
                open=Decimal("100.00"),
                high=Decimal("105.00"),
                low=Decimal("99.00"),
                close=Decimal("103.00"),
                volume=100000,
                traded_value=Decimal("10300000"),
                source="demo",
            )
        ]

    def get_latest_bar(
        self,
        symbol: str,
    ) -> MarketBar | None:

        return MarketBar(
            symbol=symbol,
            exchange="NSE",
            timestamp=datetime.now(timezone.utc),
            interval="1d",
            open=Decimal("100.00"),
            high=Decimal("105.00"),
            low=Decimal("99.00"),
            close=Decimal("103.00"),
            volume=100000,
            traded_value=Decimal("10300000"),
            source="demo",
        )