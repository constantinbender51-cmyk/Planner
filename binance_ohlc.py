#!/usr/bin/env python3
"""
binance_ohlc.py
Fetch maximum historical OHLCV data from Binance for model training.
Binance provides years of historical data vs Kraken's 720-candle limit.
"""

import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger("binance_ohlc")

BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"
MAX_CANDLES_PER_REQUEST = 1000  # Binance limit per API call


def get_ohlc(
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    start_time: Optional[int] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Timeframe - '1m','5m','15m','1h','4h','1d','1w','1M'
        start_time: Unix timestamp in milliseconds (None = fetch max available)
        limit: Max candles to fetch (None = fetch all available history)
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    all_candles = []
    current_start = start_time
    
    log.info(f"Fetching {symbol} {interval} candles from Binance...")
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": MAX_CANDLES_PER_REQUEST,
        }
        
        if current_start:
            params["startTime"] = current_start
        
        try:
            response = requests.get(BINANCE_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            candles = response.json()
            
            if not candles:
                break
            
            all_candles.extend(candles)
            
            # Check if we've reached the requested limit
            if limit and len(all_candles) >= limit:
                all_candles = all_candles[:limit]
                break
            
            # Update start time for next batch (last candle's close time + 1ms)
            current_start = candles[-1][6] + 1
            
            # Check if we've reached current time (no more data available)
            if candles[-1][0] >= int(time.time() * 1000):
                break
            
            # Rate limiting - be respectful to Binance API
            time.sleep(0.1)
            
        except requests.exceptions.RequestException as e:
            log.error(f"Binance API error: {e}")
            if all_candles:
                log.warning(f"Returning {len(all_candles)} candles fetched before error")
                break
            else:
                raise
    
    log.info(f"Fetched {len(all_candles)} total candles")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    
    # Keep only necessary columns and convert types
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna()
    
    # Log date range
    if not df.empty:
        log.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        log.info(f"Total days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
    
    return df


def get_ohlc_for_training(symbol: str = "BTCUSDT", interval: str = "1d") -> pd.DataFrame:
    """
    Convenience function to fetch ALL available historical data for training.
    This is what you'll call in lr_live.py for model fitting.
    Starts from 2017-01-01 to fetch maximum Bitcoin history.
    """
    # Start from January 1, 2017 (before Bitcoin futures launched)
    start_date = datetime(2017, 1, 1)
    start_time_ms = int(start_date.timestamp() * 1000)
    
    return get_ohlc(symbol=symbol, interval=interval, start_time=start_time_ms, limit=None)


def get_recent_ohlc(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 100) -> pd.DataFrame:
    """
    Fetch only recent candles for live predictions (faster).
    """
    return get_ohlc(symbol=symbol, interval=interval, start_time=None, limit=limit)


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Fetching ALL historical daily data ===")
    df_all = get_ohlc_for_training()
    print(f"\nShape: {df_all.shape}")
    print(f"\nFirst 5 rows:\n{df_all.head()}")
    print(f"\nLast 5 rows:\n{df_all.tail()}")
    
    print("\n=== Fetching recent 100 candles ===")
    df_recent = get_recent_ohlc(limit=100)
    print(f"Shape: {df_recent.shape}")
