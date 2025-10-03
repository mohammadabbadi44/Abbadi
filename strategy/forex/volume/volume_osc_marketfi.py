# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, Optional

def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()

def _volume_osc(vol: pd.Series, fast=12, slow=26) -> pd.Series:
    vf = _ema(vol, fast)
    vs = _ema(vol, slow)
    return vf - vs

def _mfi_bill(high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
    rng = (high - low).replace(0, np.nan)
    return (rng / volume.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def get_signals(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    """
    Volume Oscillator + Market Facilitation Index (Bill Williams).
    params:
      vo_fast=12, vo_slow=26
      mfi_smooth=5
    """
    if params is None: params = {}
    n = len(df)
    if n < 60: return pd.Series(["Hold"] * n, index=df.index, dtype="string")

    high = df["high"].astype(float)
    low  = df["low"].astype(float)
    close= df["close"].astype(float)
    vol  = df["volume"].astype(float)

    vo_fast = int(params.get("vo_fast", 12))
    vo_slow = int(params.get("vo_slow", 26))
    mfi_s   = int(params.get("mfi_smooth", 5))

    vo  = _volume_osc(vol, vo_fast, vo_slow)
    mfi = _mfi_bill(high, low, vol)
    mfi_ma = mfi.rolling(mfi_s).mean()

    # اتجاه سعري بسيط كفلتر
    bias_up = close > close.rolling(20).mean()
    bias_dn = close < close.rolling(20).mean()

    buy  = (vo > 0) & (mfi_ma.notna()) & bias_up
    sell = (vo < 0) & (mfi_ma.notna()) & bias_dn

    out = pd.Series(np.where(buy, "Buy", np.where(sell, "Sell", "Hold")), index=df.index)
    warm = max(26, mfi_s, 20) + 1
    out.iloc[:warm] = "Hold"
    return out.astype("string")

def generate_signal(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    return get_signals(df, params=params)
