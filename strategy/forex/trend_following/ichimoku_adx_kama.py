# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, Optional

def _ichimoku(df: pd.DataFrame,
              conv=9, base=26, span_b=52, shift=26):
    high = df["high"].astype(float)
    low  = df["low"].astype(float)
    close = df["close"].astype(float)

    conv_line = (high.rolling(conv).max() + low.rolling(conv).min()) / 2.0
    base_line = (high.rolling(base).max() + low.rolling(base).min()) / 2.0
    span_a = ((conv_line + base_line) / 2.0).shift(shift)
    span_bv = ((high.rolling(span_b).max() + low.rolling(span_b).min()) / 2.0).shift(shift)
    lagging = close.shift(-shift)  # مش هنستخدمه بالقرار

    return conv_line, base_line, span_a, span_bv, lagging

def _adx_like(df: pd.DataFrame, length: int = 14):
    # نسخة خفيفة من +DI/-DI لفلترة الاتجاه (مش ADX كامل)
    high = df["high"].astype(float)
    low  = df["low"].astype(float)
    close = df["close"].astype(float)

    up_move = high.diff().clip(lower=0.0)
    dn_move = (-low.diff()).clip(lower=0.0)

    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    tr_sm = tr.rolling(length).sum().replace(0, np.nan)
    plus_di  = 100 * up_move.rolling(length).sum()  / tr_sm
    minus_di = 100 * dn_move.rolling(length).sum()  / tr_sm
    adx_like = (plus_di - minus_di).abs()  # بس كقياس قوة
    return plus_di.fillna(0.0), minus_di.fillna(0.0), adx_like.fillna(0.0)

def _kama_like(close: pd.Series, er_len=10, fast=2, slow=30) -> pd.Series:
    # KAMA مبسّطة: Efficiency Ratio ثم EMA مكيفة
    change = (close - close.shift(er_len)).abs()
    vol = close.diff().abs().rolling(er_len).sum().replace(0, np.nan)
    er = (change / vol).clip(0, 1).fillna(0.0)

    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = pd.Series(index=close.index, dtype=float)
    kama.iloc[0] = close.iloc[0]
    for i in range(1, len(close)):
        alpha = sc.iloc[i] if np.isfinite(sc.iloc[i]) else slow_sc**2
        kama.iloc[i] = kama.iloc[i-1] + alpha * (close.iloc[i] - kama.iloc[i-1])
    return kama

def get_signals(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    """
    Ichimoku bias + DI filter + KAMA trend.
    params:
      conv=9, base=26, span_b=52, shift=26
      di_min=10, kama_fast=2, kama_slow=30, er_len=10
    """
    if params is None: params = {}
    n = len(df)
    if n < 120: return pd.Series(["Hold"] * n, index=df.index, dtype="string")

    conv = int(params.get("conv", 9))
    base = int(params.get("base", 26))
    span_b = int(params.get("span_b", 52))
    shift = int(params.get("shift", 26))
    di_min = float(params.get("di_min", 10))
    er_len = int(params.get("er_len", 10))
    kama_fast = int(params.get("kama_fast", 2))
    kama_slow = int(params.get("kama_slow", 30))

    close = df["close"].astype(float)
    conv_l, base_l, span_a, span_bv, _ = _ichimoku(df, conv, base, span_b, shift)
    plus_di, minus_di, adx_like = _adx_like(df)
    kama = _kama_like(close, er_len=er_len, fast=kama_fast, slow=kama_slow)

    price_above_cloud = (close > span_a) & (close > span_bv)
    price_below_cloud = (close < span_a) & (close < span_bv)

    di_buy  = (plus_di - minus_di) >= di_min
    di_sell = (minus_di - plus_di) >= di_min

    kama_up = kama > kama.shift(1)
    kama_dn = kama < kama.shift(1)

    buy  = price_above_cloud & di_buy  & kama_up
    sell = price_below_cloud & di_sell & kama_dn

    out = pd.Series(np.where(buy, "Buy", np.where(sell, "Sell", "Hold")), index=df.index)
    warm = max(span_b + shift, 60) + 1
    out.iloc[:warm] = "Hold"
    return out.astype("string")

def generate_signal(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    return get_signals(df, params=params)
