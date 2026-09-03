import pandas as pd
from pathlib import Path

INPUT = Path("data/reference/nifty50.csv")
OUTPUT = Path("data/reference/nifty50_clean.csv")

df = pd.read_csv(INPUT)

# Rename columns to consistent names
df = df.rename(columns={
    "Company Name": "company_name",
    "Industry": "industry",
    "Symbol": "symbol",
    "Series": "series",
    "ISIN Code": "isin",
})

# Clean strings
for col in ["company_name", "industry", "symbol", "series", "isin"]:
    df[col] = df[col].astype(str).str.strip()

# Keep only equity series
df = df[df["series"] == "EQ"].copy()

# Remove duplicate symbols
df = df.drop_duplicates(subset=["symbol"])

# Sort
df = df.sort_values("symbol").reset_index(drop=True)

# Save
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("=" * 60)
print("NIFTY 50 CLEANED")
print("=" * 60)
print("Companies:", len(df))
print("Saved:", OUTPUT)

print("\nColumns:")
print(df.columns.tolist())

print("\nStocks:")
print(df["symbol"].tolist())