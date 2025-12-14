#!/usr/bin/env python3
"""
kraken_ohlc.py
Fetch OHLC data from Kraken and return a Pandas DataFrame.

Usage:
    python kraken_ohlc.py XBTUSD 60     # 60-min candles for BTC/USD
    python kraken_ohlc.py ETHUSD 5      # 5-min candles for ETH/USD
"""

import sys
import time
import requests
import pandas as pd

API_URL = "https://api.kraken.com/0/public/OHLC"
MAX_CANDLES = 720          # Kraken hard-limit per call
VALID_GRANULARITY = {1, 5, 15, 30, 60, 240, 1440, 10080, 21600}


def get_ohlc(pair: str, interval: int = 60) -> pd.DataFrame:
    """
    Download OHLC data from Kraken.

    Parameters
    ----------
    pair : str
        Kraken trading pair (e.g. 'XBTUSD', 'ETHUSD').
    interval : int
        Candle size in minutes.  Choose from
        1, 5, 15, 30, 60, 240, 1440, 10080, 21600.

    Returns
    -------
    pd.DataFrame
        Columns: time, open, high, low, close, vwap, volume, trades
        Index:  pd.DatetimeIndex (UTC)
    """
    if interval not in VALID_GRANULARITY:
        raise ValueError(f"Invalid interval {interval}.  Choose from {sorted(VALID_GRANULARITY)}")

    params = {"pair": pair, "interval": interval}
    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    if payload["error"]:
        raise RuntimeError("Kraken error: " + ", ".join(payload["error"]))

    # The key inside 'result' is the normalized pair name returned by Kraken
    key = list(payload["result"].keys())[0]
    raw = payload["result"][key]

    df = pd.DataFrame(
        raw,
        columns=["time", "open", "high", "low", "close", "vwap", "volume", "trades"],
    )
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df.astype(float)
    return df


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python kraken_ohlc.py PAIR [INTERVAL_MINUTES]")
        sys.exit(1)

    pair = sys.argv[1].upper()
    interval = int(sys.argv[2]) if len(sys.argv) == 3 else 60

    ohlc = get_ohlc(pair, interval)
    print(ohlc.tail())
