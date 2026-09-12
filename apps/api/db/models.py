import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Stock(Base):
    __tablename__ = "stocks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    symbol: Mapped[str] = mapped_column(String(30), unique=True, index=True)
    exchange: Mapped[str] = mapped_column(String(20), default="NSE")
    isin: Mapped[str | None] = mapped_column(String(20), unique=True, nullable=True)
    company_name: Mapped[str] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(150), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(150), nullable=True)
    currency: Mapped[str] = mapped_column(String(10), default="INR")
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    price_bars: Mapped[list["PriceBar"]] = relationship(
        back_populates="stock",
        cascade="all, delete-orphan",
    )
    news_articles: Mapped[list["NewsArticle"]] = relationship(
        back_populates="stock"
    )
    forecasts: Mapped[list["Forecast"]] = relationship(
        back_populates="stock"
    )
    earnings_events: Mapped[list["EarningsEvent"]] = relationship(
        back_populates="stock"
    )
    intelligence_signals: Mapped[list["IntelligenceSignal"]] = relationship(
        back_populates="stock"
    )


class PriceBar(Base):
    __tablename__ = "price_bars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("stocks.id", ondelete="CASCADE"),
        index=True,
    )

    interval: Mapped[str] = mapped_column(String(20))
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )

    open: Mapped[Decimal] = mapped_column(Numeric(20, 6))
    high: Mapped[Decimal] = mapped_column(Numeric(20, 6))
    low: Mapped[Decimal] = mapped_column(Numeric(20, 6))
    close: Mapped[Decimal] = mapped_column(Numeric(20, 6))
    volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    traded_value: Mapped[Decimal | None] = mapped_column(
        Numeric(24, 6), nullable=True
    )

    source: Mapped[str] = mapped_column(String(100))

    stock: Mapped["Stock"] = relationship(back_populates="price_bars")

    __table_args__ = (
        UniqueConstraint(
            "stock_id",
            "interval",
            "timestamp",
            "source",
            name="uq_price_bar",
        ),
    )


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    stock_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("stocks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text, unique=True)
    source: Mapped[str] = mapped_column(String(255))
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True
    )

    stock: Mapped["Stock | None"] = relationship(
        back_populates="news_articles"
    )
    sentiment_scores: Mapped[list["SentimentScore"]] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
    )


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    model_key: Mapped[str] = mapped_column(String(150), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    task: Mapped[str] = mapped_column(String(100))
    provider: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    license: Mapped[str | None] = mapped_column(String(255), nullable=True)
    pretrained_only: Mapped[bool] = mapped_column(Boolean, default=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    model_runs: Mapped[list["ModelRun"]] = relationship(
        back_populates="model"
    )


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_registry.id", ondelete="CASCADE"),
        index=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    status: Mapped[str] = mapped_column(String(30))
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_hash: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    metadata_json: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True
    )

    model: Mapped["ModelRegistry"] = relationship(
        back_populates="model_runs"
    )

    forecasts: Mapped[list["Forecast"]] = relationship(
        back_populates="model_run"
    )


class SentimentScore(Base):
    __tablename__ = "sentiment_scores"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    article_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("news_articles.id", ondelete="CASCADE"),
        index=True,
    )

    model_name: Mapped[str] = mapped_column(String(150))
    label: Mapped[str] = mapped_column(String(30))

    score: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    positive_probability: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 6), nullable=True
    )
    negative_probability: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 6), nullable=True
    )
    neutral_probability: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 6), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    article: Mapped["NewsArticle"] = relationship(
        back_populates="sentiment_scores"
    )


class Forecast(Base):
    __tablename__ = "forecasts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    model_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_runs.id", ondelete="CASCADE"),
        index=True,
    )

    stock_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("stocks.id", ondelete="CASCADE"),
        index=True,
    )

    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    target_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )

    horizon: Mapped[str] = mapped_column(String(50))

    predicted_value: Mapped[Decimal] = mapped_column(Numeric(20, 6))
    lower_bound: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 6), nullable=True
    )
    upper_bound: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 6), nullable=True
    )

    direction: Mapped[str | None] = mapped_column(
        String(30), nullable=True
    )
    confidence: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 6), nullable=True
    )

    model_run: Mapped["ModelRun"] = relationship(
        back_populates="forecasts"
    )
    stock: Mapped["Stock"] = relationship(
        back_populates="forecasts"
    )


class EarningsEvent(Base):
    __tablename__ = "earnings_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    stock_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("stocks.id", ondelete="CASCADE"),
        index=True,
    )

    fiscal_period: Mapped[str] = mapped_column(String(50))
    event_date: Mapped[Date] = mapped_column(Date, index=True)
    announced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    eps_actual: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 6), nullable=True
    )
    eps_estimate: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 6), nullable=True
    )

    revenue_actual: Mapped[Decimal | None] = mapped_column(
        Numeric(24, 6), nullable=True
    )
    revenue_estimate: Mapped[Decimal | None] = mapped_column(
        Numeric(24, 6), nullable=True
    )

    currency: Mapped[str] = mapped_column(String(10), default="INR")
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    guidance_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    concall_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    stock: Mapped["Stock"] = relationship(
        back_populates="earnings_events"
    )


class IntelligenceSignal(Base):
    __tablename__ = "intelligence_signals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    stock_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("stocks.id", ondelete="CASCADE"),
        index=True,
    )

    as_of: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )

    horizon: Mapped[str] = mapped_column(String(50))
    signal: Mapped[str] = mapped_column(String(50))
    score: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 6), nullable=True
    )
    confidence: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 6), nullable=True
    )

    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    stock: Mapped["Stock"] = relationship(
        back_populates="intelligence_signals"
    )
