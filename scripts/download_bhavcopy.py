import io
import zipfile
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://nsearchives.nseindia.com/content/cm"

OUTPUT_DIR = Path("data/raw/prices")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/all-reports",
}


def download_bhavcopy(date):
    """
    Download one NSE CM-UDiFF Bhavcopy.

    date format: YYYYMMDD
    """

    filename = f"BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"

    url = f"{BASE_URL}/{filename}"

    print(f"Downloading {date}...")

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    if response.status_code != 200:
        print(
            f"Failed: {date} "
            f"HTTP {response.status_code}"
        )
        return None

    with zipfile.ZipFile(
        io.BytesIO(response.content)
    ) as z:

        csv_files = [
            f for f in z.namelist()
            if f.lower().endswith(".csv")
        ]

        if not csv_files:
            print("No CSV found")
            return None

        with z.open(csv_files[0]) as f:
            df = pd.read_csv(f)

    # Keep equity stocks only
    df = df[
    (df["Sgmt"] == "CM") &
    (df["FinInstrmTp"] == "STK") &
    (df["SctySrs"] == "EQ")].copy()

    # Normalize columns
    df = df.rename(
        columns={
            "TradDt": "date",
            "TckrSymb": "symbol",
            "ISIN": "isin",
            "OpnPric": "open",
            "HghPric": "high",
            "LwPric": "low",
            "ClsPric": "close",
            "PrvsClsgPric": "prev_close",
            "TtlTradgVol": "volume",
            "TtlTrfVal": "traded_value",
        }
    )

    df = df[
        [
            "date",
            "symbol",
            "isin",
            "open",
            "high",
            "low",
            "close",
            "prev_close",
            "volume",
            "traded_value",
        ]
    ]

    return df


if __name__ == "__main__":

    df = download_bhavcopy("20260820")

    if df is not None:

        output = OUTPUT_DIR / "20260820.parquet"

        df.to_parquet(
            output,
            index=False,
        )

        print()
        print("Saved:", output)
        print("Rows:", len(df))
        print()
        print(df.head())