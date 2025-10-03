# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, Optional

# نحاول نستخدم ADX من مشروعك، ولو مش موجود عندك نعمل بديل داخلي بسيط
try:
    from indicators.adx import calculate_adx  # لازم يرجّع DataFrame فيه +DI و -DI و ADX
except Exception:
    calculate_adx = None


def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def _frama_breakout(df: pd.DataFrame, frama_len: int = 20, confirm_window: int = 2) -> pd.Series:
    """
    نسخة مبسّطة: بنستخدم EMA كبديل خفيف للـ FRAMA ونعتبر الاختراقات حوله.
    - Buy إذا الإغلاق > EMA وهاي اليوم > هاي آخر N (اختراق)
    - Sell إذا الإغلاق < EMA ولو اليوم < لو آخر N
    """
    close, high, low = df["close"], df["high"], df["low"]
    base = _ema(close, frama_len)

    upper_break = high.rolling(frama_len).max().shift(1)
    lower_break = low.rolling(frama_len).min().shift(1)

    raw_buy = (close > base) & (close > upper_break)
    raw_sell = (close < base) & (close < lower_break)

    # تأكيد ضمن نافذة قصيرة
    buy = raw_buy.rolling(confirm_window, min_periods=1).max().astype(bool)
    sell = raw_sell.rolling(confirm_window, min_periods=1).max().astype(bool)

    out = pd.Series(np.where(buy, "Buy", np.where(sell, "Sell", "Hold")), index=df.index)
    out.iloc[:frama_len + 1] = "Hold"
    return out.astype("string")


def _di_filter(df: pd.DataFrame, di_thresh: float = 5.0) -> pd.DataFrame:
    """
    يرجّع DataFrame بعمودين bool:
      di_ok_buy:  +DI > -DI بفارق على الأقل di_thresh
      di_ok_sell: -DI > +DI بفارق على الأقل di_thresh
    لو calculate_adx متاح بنستخدمه، وإلا بنقيسه بشكل مبسط.
    """
    high, low, close = df["high"], df["low"], df["close"]

    if calculate_adx is not None:
        adx_df = calculate_adx(df)
        plus_di = pd.Series(adx_df.get("+DI", np.nan), index=df.index)
        minus_di = pd.Series(adx_df.get("-DI", np.nan), index=df.index)
    else:
        # بديل بسيط لـ +DI/-DI (مش دقيق زي الحقيقي، بس كفاية كفلتر اتجاه)
        up_move = high.diff().clip(lower=0.0)
        down_move = (-low.diff()).clip(lower=0.0)
        tr = pd.concat([
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        tr_sm = tr.rolling(14).sum().replace(0, np.nan)
        plus_di = 100 * up_move.rolling(14).sum() / tr_sm
        minus_di = 100 * down_move.rolling(14).sum() / tr_sm

    plus_di = plus_di.fillna(0.0)
    minus_di = minus_di.fillna(0.0)
    di_ok_buy = (plus_di - minus_di) >= di_thresh
    di_ok_sell = (minus_di - plus_di) >= di_thresh
    return pd.DataFrame({"di_ok_buy": di_ok_buy, "di_ok_sell": di_ok_sell}, index=df.index)


def get_signals(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    """
    DI Cross + (FRAMA-like) Breakout
    params:
      frama_len=20
      confirm_window=2
      di_thresh=5.0
    """
    if params is None:
        params = {}

    n = len(df)
    if n < 50:
        return pd.Series(["Hold"] * n, index=df.index, dtype="string")

    frama_len = int(params.get("frama_len", 20))
    confirm_window = int(params.get("confirm_window", 2))
    di_thresh = float(params.get("di_thresh", 5.0))

    brk = _frama_breakout(df, frama_len=frama_len, confirm_window=confirm_window)
    di = _di_filter(df, di_thresh=di_thresh)

    buy = (brk == "Buy") & di["di_ok_buy"]
    sell = (brk == "Sell") & di["di_ok_sell"]

    out = pd.Series(np.where(buy, "Buy", np.where(sell, "Sell", "Hold")), index=df.index, dtype="string")
    out.iloc[:frama_len + 2] = "Hold"
    return out


def generate_signal(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    """
    رابر متوافق مع اللودر القديم.
    """
    return get_signals(df, params=params).astype("string")
