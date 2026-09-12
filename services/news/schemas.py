from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, HttpUrl


class NewsItem(BaseModel):
    title: str
    url: HttpUrl
    source: str
    published_at: datetime | None = None
    summary: str | None = None
    symbol: str | None = None
