# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, Optional

def _sma(s: pd.Series, length: int) -> pd.Series:
    return s.rolling(length).mean()

def _std(s: pd.Series, length: int) -> pd.Series:
    return s.rolling(length).std(ddof=0)

def _atr(df: pd.DataFrame, length: int = 20) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    tr = pd.concat([(h - l).abs(),
                    (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def _linreg_slope(s: pd.Series, length: int = 20) -> pd.Series:
    # slope of linear regression line (normalized)
    x = np.arange(length)
    def _sl(y):
        if np.any(~np.isfinite(y)): return np.nan
        xm = x.mean(); ym = y.mean()
        denom = ((x - xm)**2).sum()
        if denom == 0: return 0.0
        return ((x - xm) * (y - ym)).sum() / denom
    return s.rolling(length).apply(_sl, raw=True)

def get_signals(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    """
    TTM Squeeze: BB inside KC -> squeeze; release + momentum slope gives direction.
    params:
      bb_len=20, bb_mult=2.0
      kc_len=20, kc_mult=1.5   (KC = SMA +/- kc_mult * ATR)
      mom_len=20
    """
    if params is None: params = {}
    n = len(df)
    if n < 80: return pd.Series(["Hold"] * n, index=df.index, dtype="string")

    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    bb_len  = int(params.get("bb_len", 20))
    bb_mult = float(params.get("bb_mult", 2.0))
    kc_len  = int(params.get("kc_len", 20))
    kc_mult = float(params.get("kc_mult", 1.5))
    mom_len = int(params.get("mom_len", 20))

    mid = _sma(close, bb_len)
    dev = _std(close, bb_len)
    bb_u = mid + bb_mult * dev
    bb_l = mid - bb_mult * dev

    kc_mid = _sma(close, kc_len)
    atr = _atr(df, kc_len)
    kc_u = kc_mid + kc_mult * atr
    kc_l = kc_mid - kc_mult * atr

    squeeze_on  = (bb_u < kc_u) & (bb_l > kc_l)
    squeeze_off = (bb_u > kc_u) | (bb_l < kc_l)

    # Momentum proxy: slope of rolling regression on close deviations from SMA
    dev_from_mid = close - mid
    mom = _linreg_slope(dev_from_mid, mom_len)

    buy  = squeeze_off & (mom > 0)
    sell = squeeze_off & (mom < 0)

    out = pd.Series(np.where(buy, "Buy", np.where(sell, "Sell", "Hold")), index=df.index)
    warm = max(bb_len, kc_len, mom_len) + 1
    out.iloc[:warm] = "Hold"
    return out.astype("string")

def generate_signal(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    return get_signals(df, params=params)
