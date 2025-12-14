#!/usr/bin/env python3
"""
planner.py - Dual-Horizon Trend System (Optimized for Fixed Futures)
Strategy 1 (Tactical): SMA 120 with 40-day parabolic decay + 13% trailing stop
Strategy 2 (Core): SMA 400 with proximity sizing + 27% trailing stop + re-entry
Combined (S3): Net position = S1 + S2 (-2x to +2x)

EXECUTION UPDATE:
- Implements "Delta Trading" (adjusts position difference only).
- Handles Fixed Maturity (FF) liquidity constraints.
- Tries Limit Order (0.02% favorable) -> Waits 10m -> Market Fallback.
- SAFETY: Places Native Stop Loss orders for intraday protection.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

import kraken_futures as kf
import kraken_ohlc

# Configuration
dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}
RUN_TRADE_NOW = os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}

# --- Instrument Settings ---
# Fixed Maturity Contract (Expires 26-03-2027)
SYMBOL_FUTS_UC = "FF_XBTUSD_260327"
SYMBOL_FUTS_LC = "ff_xbtusd_260327"
SYMBOL_OHLC_KRAKEN = "XBTUSD" # Spot price for SMA calculation
INTERVAL_KRAKEN = 1440

# --- Strategy 1 Parameters (Tactical) ---
S1_SMA = 120
S1_DECAY_DAYS = 40
S1_STOP_PCT = 0.13  # 13% trailing stop

# --- Strategy 2 Parameters (Core) ---
S2_SMA = 400
S2_PROX_PCT = 0.05  # 5% Proximity
S2_STOP_PCT = 0.27  # 27% trailing stop

# --- Execution Settings ---
STOP_WAIT_TIME = 600  # Seconds to wait for Limit fill (10 mins)
LIMIT_OFFSET_PCT = 0.0002  # 0.02% "In our favor"
# Low threshold for test account (~$8)
MIN_TRADE_SIZE = 0.0001  

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("planner")

STATE_FILE = Path("planner_state.json")

def load_state() -> Dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Failed to load state: {e}")
    
    return {
        "s1": {"entry_date": None, "peak_equity": 0.0, "stopped": False},
        "s2": {"peak_equity": 0.0, "stopped": False},
        "starting_capital": None,
        "performance": {},
        "trades": []
    }

def save_state(state: Dict):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log.error(f"Failed to save state: {e}")

def get_sma(prices: pd.Series, window: int) -> float:
    if len(prices) < window:
        return 0.0
    return prices.rolling(window=window).mean().iloc[-1]

def calculate_decay_weight(entry_date_str: str) -> float:
    """Calculates S1 parabolic decay: 1 - (days/40)^2"""
    if not entry_date_str:
        return 1.0
    
    entry_dt = datetime.fromisoformat(entry_date_str)
    if entry_dt.tzinfo is None:
        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
        
    days_since = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 86400
    
    if days_since >= S1_DECAY_DAYS:
        return 0.0
        
    weight = 1.0 - (days_since / S1_DECAY_DAYS) ** 2
    return max(0.0, weight)

def update_trailing_stops(state: Dict, current_equity: float, s1_active: bool, s2_active: bool):
    """Updates High Water Marks and checks stops based on Total Equity (Soft Check)"""
    if state["starting_capital"] is None:
        state["starting_capital"] = current_equity
        
    # --- Update S1 State ---
    s1 = state["s1"]
    if s1_active:
        if current_equity > s1["peak_equity"]:
            s1["peak_equity"] = current_equity
        
        drawdown = (s1["peak_equity"] - current_equity) / s1["peak_equity"] if s1["peak_equity"] > 0 else 0
        if drawdown > S1_STOP_PCT:
            log.warning(f"S1 STOPPED OUT (Soft Check). Drawdown: {drawdown*100:.2f}%")
            s1["stopped"] = True
            s1["entry_date"] = None
    else:
        if not s1["stopped"]: 
             s1["peak_equity"] = current_equity

    # --- Update S2 State ---
    s2 = state["s2"]
    if s2_active:
        if current_equity > s2["peak_equity"]:
            s2["peak_equity"] = current_equity
            
        drawdown = (s2["peak_equity"] - current_equity) / s2["peak_equity"] if s2["peak_equity"] > 0 else 0
        if drawdown > S2_STOP_PCT:
            log.warning(f"S2 STOPPED OUT (Soft Check). Drawdown: {drawdown*100:.2f}%")
            s2["stopped"] = True
    else:
        if not s2["stopped"]:
            s2["peak_equity"] = current_equity
            
    return state

def get_strategy_signals(price: float, sma120: float, sma400: float, state: Dict) -> Tuple[float, float, float, Dict]:
    """Calculates Net Leverage and individual Strategy components"""
    
    # --- S1 Logic (Tactical) ---
    s1_lev = 0.0
    s1_state = state["s1"]
    
    if price > sma120:
        if not s1_state["stopped"]:
            if s1_state["entry_date"] is None:
                s1_state["entry_date"] = datetime.now(timezone.utc).isoformat()
            
            weight = calculate_decay_weight(s1_state["entry_date"])
            s1_lev = 1.0 * weight
    else:
        if s1_state["stopped"]:
            s1_state["stopped"] = False
            s1_state["peak_equity"] = 0.0
            
        s1_state["entry_date"] = datetime.now(timezone.utc).isoformat()
        weight = calculate_decay_weight(s1_state["entry_date"])
        s1_lev = -1.0 * weight

    # --- S2 Logic (Core) ---
    s2_lev = 0.0
    s2_state = state["s2"]
    
    if price > sma400:
        if not s2_state["stopped"]:
            s2_lev = 1.0
        else:
            dist_pct = (price - sma400) / sma400
            if dist_pct < S2_PROX_PCT:
                s2_state["stopped"] = False
                s2_state["peak_equity"] = 0.0
                s2_lev = 0.5
    else:
        if s2_state["stopped"]:
            s2_state["stopped"] = False
        s2_lev = 0.0

    net_leverage = s1_lev + s2_lev
    net_leverage = max(-2.0, min(2.0, net_leverage))
    
    return net_leverage, s1_lev, s2_lev, state

def get_current_net_position(api, symbol) -> float:
    """Fetches net position from Kraken Futures API"""
    try:
        resp = api.get_open_positions()
        # Response format: {'result': 'success', 'openPositions': [...], ...} 
        # OR just straight dictionary depending on API version, wrapper returns .json()
        
        positions = resp.get("openPositions", [])
        for p in positions:
            if p.get('symbol').upper() == symbol.upper():
                size = float(p.get('size', 0.0))
                side = p.get('side', 'long')
                if side == 'short':
                    size = -size
                return size
        return 0.0
    except Exception as e:
        log.error(f"Error fetching positions: {e}")
        return 0.0

def get_market_price(api, symbol) -> float:
    """Fetches mark price for the contract"""
    try:
        resp = api.get_tickers()
        tickers = resp.get("tickers", [])
        for t in tickers:
            if t.get("symbol").upper() == symbol.upper():
                return float(t.get("markPrice"))
                
        # Fallback search if exact match fails
        for t in tickers:
            if symbol.upper() in t.get("symbol").upper():
                return float(t.get("markPrice"))
        
        raise ValueError(f"Ticker for {symbol} not found")
    except Exception as e:
        log.error(f"Error fetching price: {e}")
        return 0.0

def execute_delta_order(api, symbol: str, delta_size: float, current_price: float):
    # FIX: Round size to 4 decimals first to match API precision
    abs_size = round(abs(delta_size), 4)
    
    # Check against global MIN_TRADE_SIZE after rounding
    if abs_size < MIN_TRADE_SIZE:
        log.info(f"Delta {abs_size:.4f} < MIN_TRADE_SIZE {MIN_TRADE_SIZE}. No trade required.")
        return "SKIPPED"

    side = "buy" if delta_size > 0 else "sell"
    
    # Calculate Limit Price (0.02% in favor)
    # FIX: FF_XBTUSD requires prices to be Integers (Tick Size = 1.0)
    limit_price = current_price * (1.0 - LIMIT_OFFSET_PCT) if side == "buy" else current_price * (1.0 + LIMIT_OFFSET_PCT)
    limit_price = int(round(limit_price)) # Force to Integer
    
    log.info(f"EXECUTING DELTA: {side.upper()} {abs_size:.4f} @ {limit_price}")
    
    if dry:
        log.info("[DRY RUN] Order simulated.")
        return "FILLED"

    try:
        # Kraken Futures send_order
        order_params = {
            "orderType": "lmt",
            "symbol": symbol,
            "side": side,
            "size": abs_size, # Rounded to 4 decimals
            "limitPrice": limit_price # Integer
        }
        resp = api.send_order(order_params)
        log.info(f"Limit Order Placed: {resp}")
        
        log.info(f"Waiting {STOP_WAIT_TIME}s for fill...")
        time.sleep(STOP_WAIT_TIME)
        
        # Check if we need to cancel (if not fully filled)
        # We blindly cancel all orders for this symbol to ensure clean slate
        cancel_resp = api.cancel_all_orders({"symbol": symbol})
        
        # If any orders were cancelled, it means we didn't fill completely
        if cancel_resp.get("cancelStatus", {}).get("status") == "cancelled" or \
           len(cancel_resp.get("cancelledOrders", [])) > 0:
            log.info("Limit order timed out (partially or fully unfilled). Checking status...")
            return "CHECK_AGAIN"
        else:
            return "FILLED"

    except Exception as e:
        log.error(f"Order execution failed: {e}")
        return "ERROR"

def manage_stop_loss_orders(api, symbol: str, current_price: float, collateral: float, net_size: float, s1_lev: float, s2_lev: float, state: Dict):
    """
    Calculates and places Native Stop Loss orders based on Equity Drawdown.
    """
    log.info("--- Managing Safety Stops ---")
    
    if dry:
        log.info("[DRY RUN] Skipping stop placement.")
        return

    # 1. Cancel existing stops
    try:
        api.cancel_all_orders({"symbol": symbol})
    except Exception as e:
        log.warning(f"Failed to cancel old stops: {e}")

    # If hedged/neutral, stops are complex/unnecessary
    if abs(net_size) < MIN_TRADE_SIZE:
        log.info("Net position near zero. No stops needed.")
        return

    is_long = net_size > 0
    side = "sell" if is_long else "buy" 
    
    # Helper to place stop
    def place_stop(label, qty, stop_px):
        # Round qty to 4 decimals
        qty = round(qty, 4)
        
        # Stop qty must also be valid
        if qty < MIN_TRADE_SIZE:
            log.warning(f"{label} Qty {qty:.6f} < MIN {MIN_TRADE_SIZE}. Stop skipped (Account too small).")
            return
        
        # Force integer price for FF contract
        stop_px_int = int(round(stop_px))
            
        log.info(f"Placing {label}: Stop {side.upper()} {qty:.4f} @ {stop_px_int}")
        try:
            order_params = {
                "orderType": "stp", # Stop Loss
                "symbol": symbol,
                "side": side,
                "size": qty, # Rounded
                "stopPrice": stop_px_int, # Integer
                "reduceOnly": True
            }
            api.send_order(order_params)
        except Exception as e:
            log.error(f"Failed to place {label}: {e}")

    # 2. Calculate S1 Stop
    if not state["s1"]["stopped"] and abs(s1_lev) > 0.01:
        peak_s1 = state["s1"]["peak_equity"]
        if peak_s1 == 0: peak_s1 = collateral
            
        target_eq_s1 = peak_s1 * (1 - S1_STOP_PCT)
        loss_allowance = target_eq_s1 - collateral
        
        stop_price_s1 = current_price + (loss_allowance / abs(net_size)) if is_long else current_price - (loss_allowance / abs(net_size))
        s1_qty = (collateral * abs(s1_lev)) / current_price
        
        valid_stop = (stop_price_s1 < current_price) if is_long else (stop_price_s1 > current_price)
        if valid_stop:
            place_stop("S1 Protection", s1_qty, stop_price_s1)
        else:
            log.warning(f"Calculated S1 stop {stop_price_s1:.1f} is invalid (wrong side). Skipping.")

    # 3. Calculate S2 Stop
    if not state["s2"]["stopped"] and abs(s2_lev) > 0.01:
        peak_s2 = state["s2"]["peak_equity"]
        if peak_s2 == 0: peak_s2 = collateral
            
        target_eq_s2 = peak_s2 * (1 - S2_STOP_PCT)
        loss_allowance = target_eq_s2 - collateral
        
        stop_price_s2 = current_price + (loss_allowance / abs(net_size)) if is_long else current_price - (loss_allowance / abs(net_size))
        s2_qty = (collateral * abs(s2_lev)) / current_price
        
        valid_stop = (stop_price_s2 < current_price) if is_long else (stop_price_s2 > current_price)
        if valid_stop:
            place_stop("S2 Protection", s2_qty, stop_price_s2)

def daily_trade(api):
    log.info("--- Starting Daily Trade Cycle ---")
    
    # 1. Market Data (Spot for Signal)
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, interval=INTERVAL_KRAKEN)
    if df.empty: 
        log.error("Failed to fetch OHLC data")
        return
        
    current_spot = df['close'].iloc[-1]
    sma120 = get_sma(df['close'], S1_SMA)
    sma400 = get_sma(df['close'], S2_SMA)
    
    log.info(f"Spot: {current_spot} | SMA120: {sma120:.1f} | SMA400: {sma400:.1f}")
    
    # 2. State & Balance
    state = load_state()
    
    try:
        # Get Accounts returns: {'result': 'success', 'accounts': {'flex': {'portfolioValue': ...}}}
        # Or similar structure. Based on planner (1).py logic:
        accts = api.get_accounts()
        if "accounts" in accts and "flex" in accts["accounts"]:
            collateral = float(accts["accounts"]["flex"]["portfolioValue"])
        else:
            # Fallback or different structure check
            log.warning(f"Unexpected accounts structure: {accts.keys()}")
            # Try to print for debugging if failed
            collateral = 0.0
            if "accounts" in accts:
                 # Check if it's a list or dict
                 pass
            raise ValueError("Could not parse portfolioValue")
            
    except Exception as e:
        log.error(f"Failed to fetch balance: {e}")
        return

    log.info(f"Collateral: ${collateral:.2f}")
    
    # 3. Strategy Logic
    state = update_trailing_stops(state, collateral, True, True)
    target_leverage, s1_lev, s2_lev, state = get_strategy_signals(current_spot, sma120, sma400, state)
    
    # 4. Position Sizing
    futs_price = get_market_price(api, SYMBOL_FUTS_UC)
    if futs_price == 0: futs_price = current_spot # Fallback
    
    target_qty = (collateral * target_leverage) / futs_price
    
    # 5. Delta Execution
    current_qty = get_current_net_position(api, SYMBOL_FUTS_UC)
    delta_qty = target_qty - current_qty
    log.info(f"Position: {current_qty:.4f} -> {target_qty:.4f} | Delta: {delta_qty:.4f}")
    
    # Round logic is handled inside execute_delta_order, but good to check here if desired
    # execute_delta_order handles it.
    
    # We call execute even if small, let it skip internally
    res = execute_delta_order(api, SYMBOL_FUTS_UC, delta_qty, futs_price)
    
    if res == "CHECK_AGAIN" and not dry:
        time.sleep(2)
        updated_pos = get_current_net_position(api, SYMBOL_FUTS_UC)
        rem_delta = round(target_qty - updated_pos, 4) # Round Fallback
        
        if abs(rem_delta) >= MIN_TRADE_SIZE:
            log.info(f"Fallback Market: {rem_delta:.4f}")
            side = "buy" if rem_delta > 0 else "sell"
            try:
                api.send_order({
                    "orderType": "mkt",
                    "symbol": SYMBOL_FUTS_UC,
                    "side": side,
                    "size": abs(rem_delta)
                })
            except Exception as e:
                log.error(f"Fallback market order failed: {e}")
    
    # 6. Place Safety Stops
    final_net_size = get_current_net_position(api, SYMBOL_FUTS_UC)
    manage_stop_loss_orders(api, SYMBOL_FUTS_UC, futs_price, collateral, final_net_size, s1_lev, s2_lev, state)

    # 7. Save
    state["trades"].append({"date": datetime.now().isoformat(), "price": futs_price, "leverage": target_leverage, "collateral": collateral})
    save_state(state)
    log.info("Cycle Complete.")

def wait_until_00_01_utc():
    now = datetime.now(timezone.utc)
    next_run = now.replace(hour=0, minute=1, second=0, microsecond=0)
    if now >= next_run: next_run += timedelta(days=1)
    wait_sec = (next_run - now).total_seconds()
    log.info("Next run at 00:01 UTC (%s), sleeping %.0f s", next_run.strftime("%Y-%m-%d"), wait_sec)
    time.sleep(wait_sec)

def main():
    api_key = os.getenv("KRAKEN_API_KEY")
    api_sec = os.getenv("KRAKEN_API_SECRET")
    if not api_key or not api_sec:
        log.error("Env vars KRAKEN_API_KEY / KRAKEN_API_SECRET missing")
        sys.exit(1)

    api = kf.KrakenFuturesApi(api_key, api_sec)
    
    log.info("Initializing Planner (Fixed Maturity Delta-Trading)...")
    
    if RUN_TRADE_NOW:
        try: daily_trade(api)
        except Exception as e: log.exception(e)

    while True:
        wait_until_00_01_utc()
        try: daily_trade(api)
        except Exception as e: log.exception(e); time.sleep(60)

if __name__ == "__main__":
    main()
