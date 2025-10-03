# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, Optional
from indicators.supertrend import calculate_supertrend_signal  # عندنا alias جاهز بالمفرد

def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_signals(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    """
    EMA crossover + SuperTrend + RSI filter (buy/sell متناظرين).
    params:
      ema_fast=20, ema_slow=50
      rsi_len=14, rsi_min=55, rsi_max=45
      supertrend_period=10, supertrend_mult=3.0
    """
    if params is None:
        params = {}

    n = len(df)
    if n < 60:
        return pd.Series(["Hold"] * n, index=df.index, dtype="string")

    ema_fast = int(params.get("ema_fast", 20))
    ema_slow = int(params.get("ema_slow", 50))
    rsi_len  = int(params.get("rsi_len", 14))
    rsi_min  = float(params.get("rsi_min", 55))  # للشراء
    rsi_max  = float(params.get("rsi_max", 45))  # للبيع
    st_p     = int(params.get("supertrend_period", 10))
    st_m     = float(params.get("supertrend_mult", 3.0))

    close = df["close"].astype(float)
    ema_f = _ema(close, ema_fast)
    ema_s = _ema(close, ema_slow)
    rsi   = _rsi(close, rsi_len)

    st_sig = calculate_supertrend_signal(df, period=st_p, multiplier=st_m)  # Buy/Sell/Hold

    buy  = (ema_f > ema_s) & (st_sig == "Buy")  & (rsi >= rsi_min)
    sell = (ema_f < ema_s) & (st_sig == "Sell") & (rsi <= rsi_max)

    out = pd.Series(np.where(buy, "Buy", np.where(sell, "Sell", "Hold")), index=df.index)
    out.iloc[:max(ema_slow + 1, st_p + 1, rsi_len + 1)] = "Hold"
    return out.astype("string")

def generate_signal(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    return get_signals(df, params=params).astype("string")
