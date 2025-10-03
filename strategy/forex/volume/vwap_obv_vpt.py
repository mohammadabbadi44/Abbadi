# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, Optional

def _vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP = cumulative(price*volume)/cumulative(volume)"""
    tp  = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)
    cum_vol = vol.cumsum().replace(0, np.nan)
    cum_pv  = (tp * vol).cumsum()
    # استخدم bfill بدل fillna(method="bfill") لتفادي FutureWarning
    return (cum_pv / cum_vol).bfill()

def _obv(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    vol   = df["volume"].astype(float)
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * vol).cumsum()

def _vpt(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    vol   = df["volume"].astype(float)
    pct = close.pct_change().fillna(0.0)
    return (pct * vol).cumsum()

def get_signals(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    """
    VWAP bias + OBV/VPT confirmation.
    params (تقدر تغيّرها من اللودر):
      vpt_thresh=0.001   # عتبة أقوى لفلترة الضوضاء
      obv_confirm=True   # طلب تأكيد من OBV
      obv_smooth=5
    """
    if params is None:
        params = {}

    n = len(df)
    if n < 50:
        return pd.Series(["Hold"] * n, index=df.index, dtype="string")

    vpt_thresh  = float(params.get("vpt_thresh", 0.001))
    obv_confirm = bool(params.get("obv_confirm", True))
    obv_smooth  = int(params.get("obv_smooth", 5))

    vwap = _vwap(df)
    obv  = _obv(df)
    vpt  = _vpt(df)

    obv_ma = obv.rolling(obv_smooth).mean()
    price  = df["close"].astype(float)

    long_bias  = price > vwap
    short_bias = price < vwap
    vpt_up     = vpt.diff().rolling(3).mean() >= vpt_thresh
    vpt_dn     = vpt.diff().rolling(3).mean() <= -vpt_thresh

    if obv_confirm:
        obv_up = obv >= obv_ma
        obv_dn = obv <= obv_ma
    else:
        obv_up = pd.Series(True, index=df.index)
        obv_dn = pd.Series(True, index=df.index)

    buy  = long_bias  & vpt_up & obv_up
    sell = short_bias & vpt_dn & obv_dn

    out = pd.Series(np.where(buy, "Buy", np.where(sell, "Sell", "Hold")), index=df.index)
    out.iloc[:max(6, obv_smooth) + 1] = "Hold"
    return out.astype("string")

def generate_signal(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    return get_signals(df, params=params).astype("string")
