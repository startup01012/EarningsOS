from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ForecastRequest:
    symbol: str
    values: list[float]
    timestamps: list[datetime]
    horizon: int

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol must not be empty")
        if len(self.values) != len(self.timestamps):
            raise ValueError("values and timestamps must have the same length")
        if len(self.values) < 2:
            raise ValueError("at least two observations are required")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")


@dataclass(frozen=True)
class ForecastResult:
    model_id: str
    symbol: str
    generated_at: datetime
    horizon: int
    median: list[float]
    lower: list[float] | None = None
    upper: list[float] | None = None

    def __post_init__(self) -> None:
        if len(self.median) != self.horizon:
            raise ValueError("median length must equal horizon")
        if self.lower is not None and len(self.lower) != self.horizon:
            raise ValueError("lower length must equal horizon")
        if self.upper is not None and len(self.upper) != self.horizon:
            raise ValueError("upper length must equal horizon")


class ForecastAdapter(ABC):
    model_id: str

    @abstractmethod
    def forecast(self, request: ForecastRequest) -> ForecastResult:
        raise NotImplementedError
