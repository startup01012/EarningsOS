import io
import os
import zipfile
import requests
import pandas as pd


URLS = {
    "nifty50": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050",
    "nifty100": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100",
    "nifty500": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500",
}

OUTPUT_DIR = "data/reference"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/139.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def create_session():
    session = requests.Session()
    session.headers.update(HEADERS)

    # Establish NSE cookies
    response = session.get(
        "https://www.nseindia.com/",
        timeout=30,
    )

    print("NSE homepage:", response.status_code)

    return session


def download_index(session, index_name, url):
    print(f"\nDownloading {index_name.upper()}...")

    response = session.get(url, timeout=30)

    print("Status:", response.status_code)

    response.raise_for_status()

    data = response.json()

    records = data.get("data", [])

    if not records:
        raise RuntimeError(
            f"No constituents returned for {index_name}"
        )

    rows = []

    for item in records:

        # NSE API also returns an index summary row.
        # We only want actual securities.
        symbol = item.get("symbol")

        if not symbol:
            continue

        if symbol.upper().startswith("NIFTY"):
            continue

        rows.append({
            "symbol": symbol,
            "company_name": item.get("meta", {}).get(
                "companyName"
            ),
            "isin": item.get("meta", {}).get("isin"),
            "series": item.get("meta", {}).get("series"),
        })

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(
        subset=["symbol"]
    ).sort_values("symbol")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output = os.path.join(
        OUTPUT_DIR,
        f"{index_name}.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print("Saved:", output)
    print("Stocks:", len(df))

    print("\nFirst 10:")
    print(df.head(10).to_string(index=False))

    return df


def main():

    session = create_session()

    for index_name, url in URLS.items():

        try:
            download_index(
                session,
                index_name,
                url
            )

        except Exception as e:

            print(
                f"ERROR downloading "
                f"{index_name}: {e}"
            )


if __name__ == "__main__":
    main()