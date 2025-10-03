# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, Optional

def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    macd = ema_f - ema_s
    sigl = _ema(macd, signal)
    hist = macd - sigl
    return macd, sigl, hist

def get_signals(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    """
    Classic combo: EMA crossover + RSI filter + MACD histogram slope.
    params:
      ema_fast=20, ema_slow=50
      rsi_len=14, rsi_buy=52, rsi_sell=48
    """
    if params is None:
        params = {}

    n = len(df)
    if n < 60:
        return pd.Series(["Hold"] * n, index=df.index, dtype="string")

    close = df["close"].astype(float)

    ema_fast = int(params.get("ema_fast", 20))
    ema_slow = int(params.get("ema_slow", 50))
    rsi_len  = int(params.get("rsi_len", 14))
    rsi_buy  = float(params.get("rsi_buy", 52))
    rsi_sell = float(params.get("rsi_sell", 48))

    ema_f = _ema(close, ema_fast)
    ema_s = _ema(close, ema_slow)
    rsi   = _rsi(close, rsi_len)
    macd, sigl, hist = _macd(close)

    bull = (ema_f > ema_s) & (rsi >= rsi_buy) & (hist >= 0)
    bear = (ema_f < ema_s) & (rsi <= rsi_sell) & (hist <= 0)

    out = pd.Series(np.where(bull, "Buy", np.where(bear, "Sell", "Hold")), index=df.index)
    out.iloc[:max(ema_slow, rsi_len, 26) + 1] = "Hold"
    return out.astype("string")

def generate_signal(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    return get_signals(df, params=params).astype("string")
