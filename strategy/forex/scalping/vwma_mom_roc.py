# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, Optional

def _vwma(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    pv = close * volume
    num = pv.rolling(length).sum()
    den = volume.rolling(length).sum().replace(0, np.nan)
    return (num / den)

def _mom(close: pd.Series, length: int) -> pd.Series:
    return close.diff(length)

def _roc(close: pd.Series, length: int) -> pd.Series:
    prev = close.shift(length)
    return (close - prev) / prev.replace(0, np.nan)

def get_signals(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    """
    VWMA slope + Momentum + ROC confirmation.
    params:
      vwma_len=20, mom_len=5, roc_len=5
      mom_thr=0.0, roc_thr=0.0
    """
    if params is None: params = {}
    n = len(df)
    if n < 50: return pd.Series(["Hold"] * n, index=df.index, dtype="string")

    close = df["close"].astype(float)
    vol   = df["volume"].astype(float)

    vwma_len = int(params.get("vwma_len", 20))
    mom_len  = int(params.get("mom_len", 5))
    roc_len  = int(params.get("roc_len", 5))
    mom_thr  = float(params.get("mom_thr", 0.0))
    roc_thr  = float(params.get("roc_thr", 0.0))

    vw = _vwma(close, vol, vwma_len)
    vw_slope_up = vw > vw.shift(1)
    vw_slope_dn = vw < vw.shift(1)

    mom = _mom(close, mom_len)
    roc = _roc(close, roc_len)

    buy  = vw_slope_up & (mom >= mom_thr) & (roc >= roc_thr)
    sell = vw_slope_dn & (mom <= -mom_thr) & (roc <= -roc_thr)

    out = pd.Series(np.where(buy, "Buy", np.where(sell, "Sell", "Hold")), index=df.index)
    warm = max(vwma_len, mom_len, roc_len) + 1
    out.iloc[:warm] = "Hold"
    return out.astype("string")

def generate_signal(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    return get_signals(df, params=params)
