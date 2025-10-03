# strategy/forex/mean_reversion/typprice_midprice_dpo.py
import numpy as np
import pandas as pd

# =======================
#  إعدادات افتراضية (Full Power)
# =======================
DEFAULTS = {
    "midprice_period": 16,   # أقصر = نشاط أعلى
    "dpo_period": 18,        # حساسية مقبولة
    "eps_dpo": 0.06,         # عتبة DPO
    "eps_tp": 0.0008,        # ≈ 0.08% انحراف Typical عن Midprice
    "use_zscore": False,     # فلتر Z-Score معطّل لرفع النشاط
    "z_window": 50,
    "z_gate": 0.5,
    "logic": "or",           # دمج الشروط: OR للحصول على نشاط عالٍ
    "min_bars": 80
}

# =======================
#  دوال مساعدة
# =======================
def _typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0

def _midprice(df: pd.DataFrame, period: int) -> pd.Series:
    hi = df["high"].rolling(period, min_periods=period).max()
    lo = df["low"].rolling(period, min_periods=period).min()
    return (hi + lo) / 2.0

def _dpo(close: pd.Series, period: int) -> pd.Series:
    shift = int((period / 2) + 1)
    sma = close.rolling(period, min_periods=period).mean()
    return (close.shift(shift) - sma).fillna(0.0)

def _zscore(x: pd.Series, window: int) -> pd.Series:
    mean = x.rolling(window, min_periods=window).mean()
    std  = x.rolling(window, min_periods=window).std(ddof=0)
    z = (x - mean) / std.replace(0, np.nan)
    return z.fillna(0.0)

# =======================
#  توليد الإشارات
# =======================
def typprice_midprice_dpo_signals(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """
    يرجّع سلسلة إشارات ('Buy'/'Sell'/'Hold') بنفس طول df.
    يتوقع أعمدة: ['open','high','low','close','volume'] على الأقل.
    """
    if df is None or len(df) == 0:
        return pd.Series([], dtype=object)

    P = DEFAULTS.copy()
    if params:
        P.update({k: v for k, v in params.items() if k in P})

    n = len(df)
    need = max(P["min_bars"], P["midprice_period"], P["dpo_period"])
    if n < need:
        return pd.Series(["Hold"] * n, index=df.index, dtype=object)

    tp   = _typical_price(df)
    mp   = _midprice(df, P["midprice_period"])
    dpo  = _dpo(df["close"], P["dpo_period"])
    tp_dev = ((tp - mp) / mp.replace(0, np.nan)).fillna(0.0)

    # (اختياري) فلتر Z-Score
    if P.get("use_zscore", False):
        z = _zscore(dpo, P["z_window"])
        z_ok_buy  = z >  P["z_gate"]
        z_ok_sell = z < -P["z_gate"]
    else:
        z_ok_buy = z_ok_sell = True

    eps_dpo = float(P["eps_dpo"])
    eps_tp  = float(P["eps_tp"])
    use_and = str(P.get("logic", "or")).lower() == "and"

    # شروط جزئية
    buy_core  = (tp > mp)
    sell_core = (tp < mp)
    dpo_buy   = dpo >  eps_dpo
    dpo_sell  = dpo < -eps_dpo
    dev_buy   = tp_dev >  eps_tp
    dev_sell  = tp_dev < -eps_tp

    if use_and:
        buy  = buy_core  & dpo_buy  & dev_buy  & z_ok_buy
        sell = sell_core & dpo_sell & dev_sell & z_ok_sell
    else:
        buy  = buy_core  & ((dpo_buy)  | (dev_buy))  & z_ok_buy
        sell = sell_core & ((dpo_sell) | (dev_sell)) & z_ok_sell

    sig = np.where(buy, "Buy", np.where(sell, "Sell", "Hold"))
    return pd.Series(sig, index=df.index, dtype=object)

# =======================
#  توافق مع أطر مختلفة
# =======================
def generate_signal_series(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    return typprice_midprice_dpo_signals(df, params=params)

def get_signals(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    return typprice_midprice_dpo_signals(df, params=params)
