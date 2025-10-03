# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, Optional

def _sma(s: pd.Series, length: int) -> pd.Series:
    return s.rolling(length).mean()

def _std(s: pd.Series, length: int) -> pd.Series:
    return s.rolling(length).std(ddof=0)

def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    tr = pd.concat([(h - l).abs(),
                    (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def _choppiness(df: pd.DataFrame, length: int = 14) -> pd.Series:
    # CHOP = 100 * log10( sum(TR) / (max(high) - min(low)) ) / log10(length)
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    tr = pd.concat([(h - l).abs(),
                    (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    sum_tr = tr.rolling(length).sum()
    hi = h.rolling(length).max()
    lo = l.rolling(length).min()
    rng = (hi - lo).replace(0, np.nan)
    chop = 100 * (np.log10(sum_tr / rng) / np.log10(length))
    return chop.clip(0, 100)

def get_signals(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    """
    Bollinger breakout + ATR trend + CHOP filter.
    params:
      bb_len=20, bb_mult=2.0
      atr_len=14, atr_mult=1.0
      chop_len=14, chop_max=60
    """
    if params is None: params = {}
    n = len(df)
    if n < 60: return pd.Series(["Hold"] * n, index=df.index, dtype="string")

    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    bb_len  = int(params.get("bb_len", 20))
    bb_mult = float(params.get("bb_mult", 2.0))
    atr_len = int(params.get("atr_len", 14))
    atr_mult= float(params.get("atr_mult", 1.0))
    chop_len= int(params.get("chop_len", 14))
    chop_max= float(params.get("chop_max", 60))

    mid = _sma(close, bb_len)
    dev = _std(close, bb_len)
    upper = mid + bb_mult * dev
    lower = mid - bb_mult * dev

    atr = _atr(df, atr_len)
    trend_up = close > (mid + atr_mult * atr)
    trend_dn = close < (mid - atr_mult * atr)

    chop = _choppiness(df, chop_len)
    chop_ok = chop <= chop_max

    buy  = (close > upper) & trend_up & chop_ok
    sell = (close < lower) & trend_dn & chop_ok

    out = pd.Series(np.where(buy, "Buy", np.where(sell, "Sell", "Hold")), index=df.index)
    warm = max(bb_len, atr_len, chop_len) + 1
    out.iloc[:warm] = "Hold"
    return out.astype("string")

def generate_signal(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    return get_signals(df, params=params)
