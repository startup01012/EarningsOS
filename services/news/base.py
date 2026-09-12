from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from .schemas import NewsItem


class NewsProvider(ABC):
    source: str

    @abstractmethod
    def search(self, query: str, start: datetime | None = None, end: datetime | None = None) -> list[NewsItem]:
        raise NotImplementedError
