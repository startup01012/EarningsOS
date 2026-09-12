from __future__ import annotations

import argparse

from services.news.providers.rss import RSSNewsProvider
from services.news.service import NewsService


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest RSS/Atom news into EarningsOS")
    parser.add_argument("--feed-url", action="append", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--symbol")
    args = parser.parse_args()

    provider = RSSNewsProvider(args.feed_url)
    inserted, skipped = NewsService(provider).ingest(args.query, symbol=args.symbol)
    print(f"Inserted: {inserted}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
