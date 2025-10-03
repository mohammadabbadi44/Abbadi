# strategy/forex/momentum/macd_rsi_cmo.py
import numpy as np
import pandas as pd

# =======================
#  إعدادات افتراضية (Balanced, tighter buy & momentum gate)
# =======================
DEFAULTS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "rsi_period": 14,
    "cmo_period": 14,

    # توازن محسّن
    "rsi_buy": 55,      # شدّينا الشراء خطوة
    "rsi_sell": 47,     # نترك البيع كما هو
    "eps_macd": 0.03,   # فلتر MACD خفيف
    "cmo_gate": 5,      # زخم أشد من 4

    "min_bars": 100,
    "use_slope": False,  # فلتر الميل معطّل حالياً لرفع النشاط
    "slope_window": 2
}

# =======================
#  دوال المؤشرات
# =======================
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd = _ema(close, fast) - _ema(close, slow)
    macd_signal = _ema(macd, signal)
    hist = macd - macd_signal
    return macd, macd_signal, hist

def _rsi(close: pd.Series, period=14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _cmo(close: pd.Series, period=14) -> pd.Series:
    diff = close.diff()
    up_sum = diff.clip(lower=0).rolling(period, min_periods=period).sum()
    down_sum = (-diff.clip(upper=0)).rolling(period, min_periods=period).sum()
    denom = (up_sum + down_sum).replace(0, np.nan)
    cmo = 100 * (up_sum - down_sum) / denom
    return cmo.fillna(0.0)

# =======================
#  توليد الإشارات
# =======================
def macd_rsi_cmo_signals(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """
    يرجّع سلسلة إشارات ('Buy'/'Sell'/'Hold') بنفس طول df.
    يتوقع أعمدة: ['open','high','low','close','volume'] على الأقل.
    """
    if df is None or len(df) == 0:
        return pd.Series([], dtype=object)

    P = DEFAULTS.copy()
    if params:
        P.update({k: v for k, v in params.items() if k in P})

    if len(df) < P["min_bars"]:
        return pd.Series(["Hold"] * len(df), index=df.index, dtype=object)

    close = df["close"]
    macd, macd_sig, hist = _macd(close, P["macd_fast"], P["macd_slow"], P["macd_signal"])
    rsi = _rsi(close, P["rsi_period"])
    cmo = _cmo(close, P["cmo_period"])

    eps = float(P["eps_macd"])
    gate = float(P["cmo_gate"])

    if P.get("use_slope", False):
        w = int(P.get("slope_window", 1))
        dh = hist - hist.shift(w)
        slope_buy = dh > 0
        slope_sell = dh < 0
    else:
        slope_buy = slope_sell = True

    # Buy: تقاطع MACD صاعد + hist > eps + RSI>rsi_buy + CMO>+gate
    # Sell: تقاطع MACD هابط + hist<-eps + RSI<rsi_sell + CMO<-gate
    buy  = (macd > macd_sig) & (hist >  +eps) & (rsi > P["rsi_buy"]) & (cmo >  +gate) & slope_buy
    sell = (macd < macd_sig) & (hist <  -eps) & (rsi < P["rsi_sell"]) & (cmo <  -gate) & slope_sell

    sig = np.where(buy, "Buy", np.where(sell, "Sell", "Hold"))
    return pd.Series(sig, index=df.index, dtype=object)

# =======================
#  توافق مع أطر مختلفة
# =======================
def generate_signal_series(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    return macd_rsi_cmo_signals(df, params=params)

def get_signals(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    return macd_rsi_cmo_signals(df, params=params)
