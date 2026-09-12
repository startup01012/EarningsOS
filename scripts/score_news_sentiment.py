from __future__ import annotations

import argparse

from services.sentiment.service import SentimentService


def main() -> None:
    parser = argparse.ArgumentParser(description="Score pending news with pretrained FinBERT")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    scored, skipped = SentimentService().score_pending(limit=args.limit)
    print(f"Scored: {scored}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
