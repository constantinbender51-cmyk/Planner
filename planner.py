#!/usr/bin/env python3
"""
planner.py - Dual-Horizon Trend System
Strategy 1 (Tactical): SMA 120 with 40-day parabolic decay + 13% trailing stop
Strategy 2 (Core): SMA 400 with proximity sizing + 27% trailing stop + re-entry
Combined (S3): Net position = S1 + S2 (-2x to +2x)
Trades daily at 00:01 UTC on BTC/EUR
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional
import subprocess
import numpy as np
import pandas as pd

import kraken_futures as kf
import kraken_ohlc

dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}
RUN_TRADE_NOW = os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}

SYMBOL_FUTS_UC = "PF_XBTEUR"
SYMBOL_FUTS_LC = "pf_xbteur"
SYMBOL_OHLC_KRAKEN = "XBTEUR"
INTERVAL_KRAKEN = 1440

# Strategy 1 Parameters (Tactical)
S1_SMA = 120
S1_DECAY_DAYS = 40
S1_STOP_PCT = 0.13  # 13% trailing stop on trade equity

# Strategy 2 Parameters (Core)
S2_SMA = 400
S2_PROX_PCT = 0.05  # 5% proximity threshold
S2_STOP_PCT = 0.27  # 27% trailing stop on trade equity

# Order Parameters
LIMIT_OFFSET_PCT = 0.0002  # 0.02% offset for limit orders
STOP_WAIT_TIME = 600  # Wait 10 minutes

STATE_FILE = Path("planner_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
log = logging.getLogger("planner_strategy")


def calculate_smas(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate SMA 120 and SMA 400"""
    df = df.copy()
    df['sma_120'] = df['close'].rolling(window=S1_SMA).mean()
    df['sma_400'] = df['close'].rolling(window=S2_SMA).mean()
    df['dist_pct_400'] = (df['close'] - df['sma_400']).abs() / df['sma_400']
    return df


def calculate_decay_weight(days_since_entry: int) -> float:
    """
    Calculate parabolic decay weight for S1
    W = 1 - (d/40)^2
    """
    if days_since_entry <= 0:
        return 1.0
    if days_since_entry > S1_DECAY_DAYS:
        return 0.0
    return 1.0 - (days_since_entry / S1_DECAY_DAYS) ** 2


def generate_s1_signal(df: pd.DataFrame, s1_state: Dict) -> Tuple[float, Dict]:
    """
    Strategy 1: Tactical Trend (SMA 120)
    - Entry: Price crosses SMA 120
    - Sizing: 1.0x decaying parabolically over 40 days
    - Exit: 13% trailing stop on trade equity OR decay completion
    
    Returns: (position_size, updated_state)
    """
    df_calc = calculate_smas(df)
    current_price = df_calc['close'].iloc[-1]
    current_sma_120 = df_calc['sma_120'].iloc[-1]
    
    if pd.isna(current_sma_120):
        return 0.0, s1_state
    
    # Determine current trend
    current_trend = 1 if current_price > current_sma_120 else -1
    
    # Get previous trend from state
    prev_trend = s1_state.get('trend', 0)
    stopped = s1_state.get('stopped', False)
    entry_date = s1_state.get('entry_date', None)
    
    # New trend signal (crossover)
    if current_trend != prev_trend:
        log.info(f"S1: New trend detected: {current_trend}")
        s1_state = {
            'trend': current_trend,
            'entry_date': datetime.now(timezone.utc).isoformat(),
            'trade_equity': 1.0,
            'peak_equity': 1.0,
            'stopped': False
        }
        return float(current_trend), s1_state
    
    # Existing trend - check if stopped
    if stopped:
        log.info("S1: Stopped out, position = 0")
        return 0.0, s1_state
    
    # Calculate days since entry
    if entry_date:
        entry_dt = datetime.fromisoformat(entry_date.replace('Z', '+00:00'))
        days_since = (datetime.now(timezone.utc) - entry_dt).days
    else:
        days_since = 0
    
    # Calculate decay weight
    decay_weight = calculate_decay_weight(days_since)
    
    # Check if decay expired
    if decay_weight <= 0:
        log.info(f"S1: Decay expired after {days_since} days, position = 0")
        s1_state['stopped'] = True
        return 0.0, s1_state
    
    # Calculate position
    position = current_trend * decay_weight
    
    log.info(f"S1: Trend={current_trend}, Days={days_since}, Decay={decay_weight:.3f}, Pos={position:.3f}")
    
    return position, s1_state


def generate_s2_signal(df: pd.DataFrame, s2_state: Dict) -> Tuple[float, Dict]:
    """
    Strategy 2: Core Trend (SMA 400)
    - Entry: Price crosses SMA 400
    - Sizing: 0.5x if within 5% proximity, else 1.0x
    - Exit: 27% trailing stop on trade equity
    - Re-entry: If stopped, can re-enter at 0.5x when price returns to proximity
    
    Returns: (position_size, updated_state)
    """
    df_calc = calculate_smas(df)
    current_price = df_calc['close'].iloc[-1]
    current_sma_400 = df_calc['sma_400'].iloc[-1]
    current_dist_pct = df_calc['dist_pct_400'].iloc[-1]
    
    if pd.isna(current_sma_400):
        return 0.0, s2_state
    
    # Determine current trend and proximity
    current_trend = 1 if current_price > current_sma_400 else -1
    in_proximity = current_dist_pct < S2_PROX_PCT
    target_weight = 0.5 if in_proximity else 1.0
    
    # Get previous state
    prev_trend = s2_state.get('trend', 0)
    stopped = s2_state.get('stopped', False)
    
    # New trend signal (crossover)
    if current_trend != prev_trend:
        log.info(f"S2: New trend detected: {current_trend}")
        s2_state = {
            'trend': current_trend,
            'entry_date': datetime.now(timezone.utc).isoformat(),
            'trade_equity': 1.0,
            'peak_equity': 1.0,
            'stopped': False
        }
        position = current_trend * target_weight
        log.info(f"S2: Entry position={position:.3f}, proximity={in_proximity}")
        return position, s2_state
    
    # Existing trend
    if stopped:
        # Check re-entry condition
        if in_proximity:
            log.info(f"S2: Re-entering at 0.5x (proximity breach)")
            s2_state = {
                'trend': current_trend,
                'entry_date': datetime.now(timezone.utc).isoformat(),
                'trade_equity': 1.0,
                'peak_equity': 1.0,
                'stopped': False
            }
            return current_trend * 0.5, s2_state
        else:
            log.info("S2: Stopped out, waiting for proximity")
            return 0.0, s2_state
    
    # Active position - adjust for proximity
    position = current_trend * target_weight
    log.info(f"S2: Trend={current_trend}, Proximity={in_proximity}, Weight={target_weight}, Pos={position:.3f}")
    
    return position, s2_state


def portfolio_usd(api: kf.KrakenFuturesApi) -> float:
    return float(api.get_accounts()["accounts"]["flex"]["portfolioValue"])


def mark_price(api: kf.KrakenFuturesApi) -> float:
    tk = api.get_tickers()
    for t in tk["tickers"]:
        if t["symbol"] == SYMBOL_FUTS_UC:
            return float(t["markPrice"])
    raise RuntimeError(f"Mark-price for {SYMBOL_FUTS_UC} not found")


def cancel_all(api: kf.KrakenFuturesApi):
    log.info("Cancelling all orders")
    try:
        api.cancel_all_orders()
    except Exception as e:
        log.warning("cancel_all_orders failed: %s", e)


def get_current_position(api: kf.KrakenFuturesApi) -> Optional[Dict]:
    """Get current open position from Kraken"""
    try:
        pos = api.get_open_positions()
        for p in pos.get("openPositions", []):
            if p["symbol"] == SYMBOL_FUTS_UC:
                return {
                    "signal": "LONG" if p["side"] == "long" else "SHORT",
                    "side": p["side"],
                    "size_btc": abs(float(p["size"])),
                }
        return None
    except Exception as e:
        log.warning(f"Failed to get position: {e}")
        return None


def flatten_position_limit(api: kf.KrakenFuturesApi, current_price: float):
    """Flatten position with limit order (0.02% in favorable direction)"""
    pos = get_current_position(api)
    if not pos:
        log.info("No position to flatten")
        return
    
    side = "sell" if pos["side"] == "long" else "buy"
    size = pos["size_btc"]
    
    if side == "sell":
        limit_price = current_price * (1 + LIMIT_OFFSET_PCT)
    else:
        limit_price = current_price * (1 - LIMIT_OFFSET_PCT)
    
    log.info(f"Flatten with limit: {side} {size:.4f} BTC at €{limit_price:.2f}")
    
    try:
        api.send_order({
            "orderType": "lmt",
            "symbol": SYMBOL_FUTS_LC,
            "side": side,
            "size": round(size, 4),
            "limitPrice": int(round(limit_price)),
        })
    except Exception as e:
        log.warning(f"Flatten limit order failed: {e}")


def flatten_position_market(api: kf.KrakenFuturesApi):
    """Flatten any remaining position with market order"""
    pos = get_current_position(api)
    if not pos:
        log.info("No remaining position to flatten")
        return
    
    side = "sell" if pos["side"] == "long" else "buy"
    size = pos["size_btc"]
    
    log.info(f"Flatten remaining with market: {side} {size:.4f} BTC")
    
    try:
        api.send_order({
            "orderType": "mkt",
            "symbol": SYMBOL_FUTS_LC,
            "side": side,
            "size": round(size, 4),
        })
    except Exception as e:
        log.warning(f"Flatten market order failed: {e}")


def place_entry_order(api: kf.KrakenFuturesApi, net_position: float, current_price: float, collateral: float):
    """Place entry order for combined S3 position"""
    if abs(net_position) < 0.01:
        log.info("Net position near zero, no entry needed")
        return 0.0
    
    # Calculate size
    notional = collateral * abs(net_position)
    size_btc = round(notional / current_price, 4)
    side = "buy" if net_position > 0 else "sell"
    
    # Place limit order
    if side == "buy":
        limit_price = current_price * (1 - LIMIT_OFFSET_PCT)
    else:
        limit_price = current_price * (1 + LIMIT_OFFSET_PCT)
    
    log.info(f"=== STEP 5: Place entry limit order ===")
    log.info(f"Entry limit: {side} {size_btc:.4f} BTC at €{limit_price:.2f}")
    
    try:
        api.send_order({
            "orderType": "lmt",
            "symbol": SYMBOL_FUTS_LC,
            "side": side,
            "size": size_btc,
            "limitPrice": int(round(limit_price)),
        })
    except Exception as e:
        log.error(f"Entry limit order failed: {e}")
        return 0.0
    
    log.info("=== STEP 6: Sleeping 600 seconds ===")
    time.sleep(600)
    
    # Check fill and place market for remaining
    log.info("=== STEP 7: Place entry market for remaining ===")
    pos = get_current_position(api)
    
    if pos and pos["side"] == ("long" if side == "buy" else "short"):
        filled_size = pos["size_btc"]
        log.info(f"Limit order filled {filled_size:.4f} BTC of {size_btc:.4f} BTC")
        
        remaining = size_btc - filled_size
        if remaining > 0.0001:
            log.info(f"Entry market for remaining: {side} {remaining:.4f} BTC")
            try:
                api.send_order({
                    "orderType": "mkt",
                    "symbol": SYMBOL_FUTS_LC,
                    "side": side,
                    "size": round(remaining, 4),
                })
                return size_btc
            except Exception as e:
                log.warning(f"Entry market order failed: {e}")
                return filled_size
        else:
            log.info("Limit order fully filled")
            return filled_size
    else:
        log.warning("No position found after limit, placing full market order")
        try:
            api.send_order({
                "orderType": "mkt",
                "symbol": SYMBOL_FUTS_LC,
                "side": side,
                "size": size_btc,
            })
            return size_btc
        except Exception as e:
            log.error(f"Full market order failed: {e}")
            return 0.0


def update_trailing_stops(df: pd.DataFrame, s1_state: Dict, s2_state: Dict, 
                         prev_price: float, current_price: float) -> Tuple[Dict, Dict]:
    """
    Update trailing stops for both strategies based on daily returns
    Returns: (updated_s1_state, updated_s2_state)
    """
    daily_return = (current_price - prev_price) / prev_price
    
    # Update S1 if active
    if not s1_state.get('stopped', False) and s1_state.get('trend', 0) != 0:
        s1_pos = s1_state.get('trend', 0)
        
        # Get decay weight
        entry_date = s1_state.get('entry_date')
        if entry_date:
            entry_dt = datetime.fromisoformat(entry_date.replace('Z', '+00:00'))
            days_since = (datetime.now(timezone.utc) - entry_dt).days
            decay_weight = calculate_decay_weight(days_since)
        else:
            decay_weight = 1.0
        
        actual_pos = s1_pos * decay_weight
        trade_pnl = actual_pos * daily_return
        
        s1_state['trade_equity'] = s1_state.get('trade_equity', 1.0) * (1 + trade_pnl)
        if s1_state['trade_equity'] > s1_state.get('peak_equity', 1.0):
            s1_state['peak_equity'] = s1_state['trade_equity']
        
        # Check trailing stop
        stop_threshold = s1_state['peak_equity'] * (1 - S1_STOP_PCT)
        if s1_state['trade_equity'] < stop_threshold:
            log.info(f"S1: STOPPED OUT - Equity {s1_state['trade_equity']:.3f} < Threshold {stop_threshold:.3f}")
            s1_state['stopped'] = True
    
    # Update S2 if active
    if not s2_state.get('stopped', False) and s2_state.get('trend', 0) != 0:
        # Get current S2 position from last calculation
        df_calc = calculate_smas(df)
        current_dist_pct = df_calc['dist_pct_400'].iloc[-1]
        in_proximity = current_dist_pct < S2_PROX_PCT
        s2_weight = 0.5 if in_proximity else 1.0
        s2_pos = s2_state.get('trend', 0) * s2_weight
        
        trade_pnl = s2_pos * daily_return
        
        s2_state['trade_equity'] = s2_state.get('trade_equity', 1.0) * (1 + trade_pnl)
        if s2_state['trade_equity'] > s2_state.get('peak_equity', 1.0):
            s2_state['peak_equity'] = s2_state['trade_equity']
        
        # Check trailing stop
        stop_threshold = s2_state['peak_equity'] * (1 - S2_STOP_PCT)
        if s2_state['trade_equity'] < stop_threshold:
            log.info(f"S2: STOPPED OUT - Equity {s2_state['trade_equity']:.3f} < Threshold {stop_threshold:.3f}")
            s2_state['stopped'] = True
    
    return s1_state, s2_state


def smoke_test(api: kf.KrakenFuturesApi):
    """Run smoke test to verify API connectivity"""
    log.info("=== Smoke-test start ===")
    
    try:
        usd = portfolio_usd(api)
        log.info(f"Portfolio value: €{usd:.2f}")
        
        mp = mark_price(api)
        log.info(f"BTC/EUR mark price: €{mp:.2f}")
        
        current_pos = get_current_position(api)
        if current_pos:
            log.info(f"Open position: {current_pos['signal']} {current_pos['size_btc']:.4f} BTC")
        else:
            log.info("No open positions")
        
        df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
        log.info(f"Historical data: {len(df)} days available")
        
        if len(df) < S2_SMA:
            log.warning(f"Only {len(df)} days available, need {S2_SMA} for SMA 400")
        
        log.info("=== Smoke-test complete ===")
        return True
    except Exception as e:
        log.error(f"Smoke test failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False


def load_state() -> Dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {
        "trades": [],
        "starting_capital": None,
        "performance": {},
        "current_position": None,
        "s1_state": {'trend': 0, 'stopped': False},
        "s2_state": {'trend': 0, 'stopped': False},
        "prev_price": None
    }


def save_state(st: Dict):
    STATE_FILE.write_text(json.dumps(st, indent=2))


def daily_trade(api: kf.KrakenFuturesApi):
    """Execute daily trading strategy"""
    state = load_state()
    
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
    current_price = mark_price(api)
    portfolio_value = portfolio_usd(api)
    
    if state["starting_capital"] is None:
        state["starting_capital"] = portfolio_value
    
    # Update trailing stops based on yesterday's price movement
    if state.get("prev_price"):
        state["s1_state"], state["s2_state"] = update_trailing_stops(
            df, state["s1_state"], state["s2_state"], 
            state["prev_price"], current_price
        )
    
    # Generate signals
    log.info("=== GENERATING SIGNALS ===")
    s1_position, state["s1_state"] = generate_s1_signal(df, state["s1_state"])
    s2_position, state["s2_state"] = generate_s2_signal(df, state["s2_state"])
    net_position = s1_position + s2_position
    
    log.info(f"S1 Position: {s1_position:.3f}")
    log.info(f"S2 Position: {s2_position:.3f}")
    log.info(f"Net S3 Position: {net_position:.3f}")
    
    # Flatten existing position
    log.info("=== STEP 1: Flatten with limit order ===")
    flatten_position_limit(api, current_price)
    
    log.info("=== STEP 2: Sleeping 600 seconds ===")
    time.sleep(600)
    
    log.info("=== STEP 3: Flatten remaining with market order ===")
    flatten_position_market(api)
    
    log.info("=== STEP 4: Cancel all orders ===")
    cancel_all(api)
    time.sleep(2)
    
    collateral = portfolio_usd(api)
    
    # Enter new position
    if abs(net_position) < 0.01:
        log.info("Net position near zero - staying flat")
        final_size = 0.0
    else:
        if dry:
            log.info(f"DRY-RUN: Net position {net_position:.3f}x at €{current_price:.2f}")
            final_size = abs(net_position) * collateral / current_price
        else:
            final_size = place_entry_order(api, net_position, current_price, collateral)
            
            log.info("=== STEP 8: Cancel all orders ===")
            cancel_all(api)
            time.sleep(2)
    
    # Record trade
    trade_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "s1_position": s1_position,
        "s2_position": s2_position,
        "net_position": net_position,
        "size_btc": final_size,
        "price": current_price,
        "portfolio_value": collateral,
        "s1_state": state["s1_state"].copy(),
        "s2_state": state["s2_state"].copy()
    }
    
    state["trades"].append(trade_record)
    state["prev_price"] = current_price
    state["current_position"] = {
        "net_position": net_position,
        "size_btc": final_size
    }
    
    if state["starting_capital"]:
        total_return = (collateral - state["starting_capital"]) / state["starting_capital"] * 100
        state["performance"] = {
            "current_value": collateral,
            "starting_capital": state["starting_capital"],
            "total_return_pct": total_return,
            "total_trades": len(state["trades"]),
        }
    
    save_state(state)
    log.info(f"Trade executed. Portfolio: €{collateral:.2f}, Net Position: {net_position:.3f}x")


def wait_until_00_01_utc():
    """Wait until 00:01 UTC for daily execution"""
    now = datetime.now(timezone.utc)
    next_run = now.replace(hour=0, minute=1, second=0, microsecond=0)
    if now >= next_run:
        next_run += timedelta(days=1)
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
    
    log.info("Initializing Dual-Horizon Trend System (Planner)...")
    log.info("S1: SMA 120 with 40-day decay + 13% trailing stop")
    log.info("S2: SMA 400 with proximity sizing + 27% trailing stop")
    log.info("Trading BTC/EUR on Kraken Futures")
    
    if not smoke_test(api):
        log.error("Smoke test failed, exiting")
        sys.exit(1)
    
    state = load_state()
    save_state(state)
    
    if RUN_TRADE_NOW:
        log.info("RUN_TRADE_NOW=true – executing trade now")
        try:
            daily_trade(api)
        except Exception as exc:
            log.exception("Immediate trade failed: %s", exc)

    log.info("Starting web dashboard on port %s", os.getenv("PORT", 8080))
    time.sleep(1)
    subprocess.Popen([sys.executable, "web_planner.py"])

    while True:
        wait_until_00_01_utc()
        try:
            daily_trade(api)
        except KeyboardInterrupt:
            log.info("Interrupted")
            break
        except Exception as exc:
            log.exception("Daily trade failed: %s", exc)


if __name__ == "__main__":
    main()
