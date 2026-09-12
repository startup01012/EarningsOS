from apps.api.db.session import SessionLocal
from services.market_data.ingestion import ingest_bars
from services.market_data.providers.csv_provider import CSVMarketDataProvider


def main():
    provider = CSVMarketDataProvider(
        "data/test/market_data.csv"
    )

    bars = provider.get_historical_bars(
        symbol="RELIANCE",
        start=__import__("datetime").datetime.fromisoformat(
            "2026-09-01T00:00:00+00:00"
        ),
        end=__import__("datetime").datetime.fromisoformat(
            "2026-09-30T23:59:59+00:00"
        ),
    )

    db = SessionLocal()

    try:
        inserted = ingest_bars(db, bars)
        print(f"Bars inserted: {inserted}")

    finally:
        db.close()


if __name__ == "__main__":
    main()