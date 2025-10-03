# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, Optional

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _stoch(series: pd.Series, k: int = 14, d: int = 3):
    low_k  = series.rolling(k).min()
    high_k = series.rolling(k).max().replace(0, np.nan)
    st_k = 100 * (series - low_k) / (high_k - low_k)
    st_d = st_k.rolling(d).mean()
    return st_k, st_d

def _wma(s: pd.Series, period: int) -> pd.Series:
    w = np.arange(1, period + 1)
    return s.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def _hma(s: pd.Series, period: int = 16) -> pd.Series:
    if period < 2:
        return s
    half = int(period / 2)
    sqrt = int(np.sqrt(period))
    wma_half = _wma(s, half)
    wma_full = _wma(s, period)
    hull_raw = 2 * wma_half - wma_full
    return _wma(hull_raw, sqrt)

def _cmo(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0).rolling(length).sum()
    dn = (-delta.clip(upper=0.0)).rolling(length).sum()
    denom = (up + dn).replace(0, np.nan)
    return 100 * (up - dn) / denom

def get_signals(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    """
    StochRSI + HMA direction + CMO bias.
    params:
      srsi_k=5, srsi_d=3, rsi_len=14, hull_len=16, cmo_len=14
      srsi_buy=45, srsi_sell=55
    """
    if params is None:
        params = {}
    n = len(df)
    if n < 60:
        return pd.Series(["Hold"] * n, index=df.index, dtype="string")

    close = df["close"].astype(float)

    srsi_k_len = int(params.get("srsi_k", 5))
    srsi_d_len = int(params.get("srsi_d", 3))
    rsi_len    = int(params.get("rsi_len", 14))
    hull_len   = int(params.get("hull_len", 16))
    cmo_len    = int(params.get("cmo_len", 14))
    srsi_buy   = float(params.get("srsi_buy", 45))
    srsi_sell  = float(params.get("srsi_sell", 55))

    rsi = _rsi(close, rsi_len)
    k, d = _stoch(rsi, srsi_k_len, srsi_d_len)
    hma = _hma(close, hull_len)
    cmo = _cmo(close, cmo_len)

    hull_up = hma > hma.shift(1)
    hull_dn = hma < hma.shift(1)

    buy  = (k > d) & (k >= srsi_buy) & hull_up & (cmo >= 0)
    sell = (k < d) & (k <= srsi_sell) & hull_dn & (cmo <= 0)

    out = pd.Series(np.where(buy, "Buy", np.where(sell, "Sell", "Hold")), index=df.index)
    warm = max(rsi_len, srsi_k_len + srsi_d_len, hull_len, cmo_len) + 1
    out.iloc[:warm] = "Hold"
    return out.astype("string")

def generate_signal(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    return get_signals(df, params=params).astype("string")
