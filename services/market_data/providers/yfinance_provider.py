from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import yfinance as yf

from ..base import MarketDataProvider
from ..schemas import MarketBar

IST = ZoneInfo("Asia/Kolkata")


class YFinanceProvider(MarketDataProvider):
    """Historical market-data adapter using Yahoo Finance's public feed.

    This adapter is intended for development/research and is isolated behind the
    provider interface so it can be replaced by a licensed NSE data provider.
    """

    source = "yfinance"

    def _ticker(self, symbol: str) -> str:
        return f"{symbol.strip().upper()}.NS"

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> list[MarketBar]:
        if interval != "1d":
            raise ValueError("YFinanceProvider currently supports interval='1d' only")
        if start >= end:
            raise ValueError("start must be earlier than end")

        frame = yf.download(
            self._ticker(symbol),
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            actions=False,
        )
        if frame.empty:
            return []

        if hasattr(frame.columns, "levels") and len(frame.columns.levels) > 1:
            frame.columns = frame.columns.get_level_values(0)

        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing yfinance columns for {symbol}: {sorted(missing)}")

        bars: list[MarketBar] = []
        for timestamp, row in frame.iterrows():
            ts = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=IST)
            else:
                ts = ts.astimezone(IST)

            bars.append(
                MarketBar(
                    symbol=symbol.strip().upper(),
                    exchange="NSE",
                    timestamp=ts,
                    interval="1d",
                    open=Decimal(str(row["Open"])),
                    high=Decimal(str(row["High"])),
                    low=Decimal(str(row["Low"])),
                    close=Decimal(str(row["Close"])),
                    volume=int(row["Volume"]) if row["Volume"] == row["Volume"] else None,
                    traded_value=None,
                    source=self.source,
                )
            )
        return bars

    def get_latest_bar(self, symbol: str) -> MarketBar | None:
        now = datetime.now(IST)
        bars = self.get_historical_bars(
            symbol,
            now - timedelta(days=7),
            now + timedelta(days=1),
        )
        return bars[-1] if bars else None
