from abc import ABC, abstractmethod
from datetime import datetime

from .schemas import MarketBar


class MarketDataProvider(ABC):

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> list[MarketBar]:
        raise NotImplementedError

    @abstractmethod
    def get_latest_bar(
        self,
        symbol: str,
    ) -> MarketBar | None:
        raise NotImplementedError