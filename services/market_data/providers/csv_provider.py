from datetime import datetime
from decimal import Decimal

import csv

from ..base import MarketDataProvider
from ..schemas import MarketBar


class CSVMarketDataProvider(MarketDataProvider):
    """
    Development provider for normalized CSV market data.

    Expected columns:
    symbol,exchange,timestamp,interval,open,high,low,close,volume,traded_value
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def _load(self) -> list[MarketBar]:
        bars: list[MarketBar] = []

        with open(self.file_path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                bars.append(
                    MarketBar(
                        symbol=row["symbol"],
                        exchange=row["exchange"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        interval=row["interval"],
                        open=Decimal(row["open"]),
                        high=Decimal(row["high"]),
                        low=Decimal(row["low"]),
                        close=Decimal(row["close"]),
                        volume=(
                            int(row["volume"])
                            if row.get("volume")
                            else None
                        ),
                        traded_value=(
                            Decimal(row["traded_value"])
                            if row.get("traded_value")
                            else None
                        ),
                        source="csv",
                    )
                )

        return bars

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> list[MarketBar]:

        bars = self._load()

        return [
            bar
            for bar in bars
            if (
                bar.symbol == symbol
                and bar.interval == interval
                and start <= bar.timestamp <= end
            )
        ]

    def get_latest_bar(
        self,
        symbol: str,
    ) -> MarketBar | None:

        bars = [
            bar
            for bar in self._load()
            if bar.symbol == symbol
        ]

        if not bars:
            return None

        return max(bars, key=lambda bar: bar.timestamp)
