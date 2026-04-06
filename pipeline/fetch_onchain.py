"""Fetch on-chain BTC metrics from Blockchain.com API."""

import warnings
from datetime import datetime

import pandas as pd
import requests

START_DATE = "2016-01-01"
BASE_URL = "https://api.blockchain.info/charts"

# Blockchain.com chart names
CHARTS = {
    "hash-rate": "hash_rate",
    "n-transactions": "tx_count",
    "difficulty": "mining_difficulty",
}


def _fetch_chart(chart_name: str, col_name: str, start: str, end: str | None) -> pd.Series | None:
    """Fetch a single chart series from Blockchain.com."""
    params = {
        "timespan": "all",
        "start": start,
        "format": "json",
        "sampled": "false",
    }
    try:
        resp = requests.get(f"{BASE_URL}/{chart_name}", params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        values = data.get("values", [])
        if not values:
            warnings.warn(f"No data returned for {chart_name}")
            return None
        records = [{"date": datetime.utcfromtimestamp(v["x"]), "value": v["y"]} for v in values]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop_duplicates(subset="date", keep="first").set_index("date").sort_index()
        series = df["value"].rename(col_name)
        # Filter date range
        series = series[series.index >= start]
        if end:
            series = series[series.index <= end]
        return series
    except Exception as e:
        warnings.warn(f"Failed to fetch {chart_name}: {e}")
        return None


def fetch_onchain(end_date: str | None = None) -> pd.DataFrame:
    """Fetch all on-chain metrics and return as DataFrame."""
    frames = []
    for chart_name, col_name in CHARTS.items():
        series = _fetch_chart(chart_name, col_name, START_DATE, end_date)
        if series is not None:
            frames.append(series)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1).sort_index()
    return df


if __name__ == "__main__":
    df = fetch_onchain()
    print(f"On-chain data: {len(df)} rows, {df.columns.tolist()}")
    print(df.tail())
