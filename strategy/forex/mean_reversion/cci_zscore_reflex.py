# strategy/forex/mean_reversion/cci_zscore_reflex.py
import numpy as np
import pandas as pd

# =======================
#  إعدادات افتراضية
# =======================
DEFAULTS = {
    "cci_period": 20,      # فترة CCI
    "zscore_period": 20,   # نافذة Z-Score على Typical Price
    "reflex_lag": 2,       # تأخير لحساب Reflex (momentum normalized)
    "eps_z": 0.30,         # عتبة zscore لكسر الـ Hold (تم تعديلها من 0.35 -> 0.30)
    "cci_enter_lo": -70,   # جعلنا الشراء أصعب قليلاً (كان -50)
    "cci_enter_hi": +40,   # جعلنا البيع أسهل قليلاً (كان +50)
    "min_bars": 100,       # أقل عدد شموع قبل توليد إشارات
}

# =======================
#  مؤشرات داخلية خفيفة
# =======================
def _typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0

def _rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    mean = x.rolling(window, min_periods=window).mean()
    std = x.rolling(window, min_periods=window).std(ddof=0)
    z = (x - mean) / std.replace(0, np.nan)
    return z.fillna(0.0)

def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = _typical_price(df)
    sma = tp.rolling(period, min_periods=period).mean()
    md = tp.rolling(period, min_periods=period).apply(
        lambda w: np.mean(np.abs(w - np.mean(w))), raw=True
    )
    cci = (tp - sma) / (0.015 * md.replace(0, np.nan))
    return cci.fillna(0.0)

def _reflex(close: pd.Series, lag: int = 2, norm_window: int = 50) -> pd.Series:
    """
    Reflex مبسّط: فرق سعري مطبّع بالانحراف المعياري المتحرك.
    """
    raw = close - close.shift(lag)
    norm = raw / (raw.rolling(norm_window, min_periods=norm_window).std(ddof=0).replace(0, np.nan))
    return norm.fillna(0.0)

# =======================
#  إنشاء الإشارات
# =======================
def cci_zscore_reflex_signals(
    df: pd.DataFrame,
    params: dict | None = None
) -> pd.Series:
    """
    يرجع سلسلة إشارات ('Buy'/'Sell'/'Hold') بنفس طول df.
    يتوقع أعمدة: ['open','high','low','close','volume'] على الأقل.
    """
    if df is None or len(df) == 0:
        return pd.Series([], dtype=object)

    P = DEFAULTS.copy()
    if params:
        P.update({k: v for k, v in params.items() if k in P})

    n = len(df)
    if n < P["min_bars"]:
        return pd.Series(["Hold"] * n, index=df.index, dtype=object)

    # حساب المؤشرات
    cci = _cci(df, P["cci_period"])
    z   = _rolling_zscore(_typical_price(df), P["zscore_period"])
    rx  = _reflex(df["close"], lag=P["reflex_lag"])

    eps = float(P["eps_z"])

    # بوابات مرنة:
    # Buy: زخم إيجابي + zscore فوق العتبة + CCI أعلى من مدخل الشراء
    # Sell: عكسها
    cci_buy_gate  = (cci >  P["cci_enter_lo"]) & (z > +eps) & (rx > 0)
    cci_sell_gate = (cci <  P["cci_enter_hi"]) & (z < -eps) & (rx < 0)

    signals = np.where(cci_buy_gate, "Buy",
                np.where(cci_sell_gate, "Sell", "Hold"))

    s = pd.Series(signals, index=df.index, dtype=object)

    # ملاحظة: لو بدك anti-chop بسيط (تثبيت آخر إشارة لفترة قصيرة)،
    # ممكن تضيف post-processing هنا. حالياً تركناه نظيفًا للنشاط المتوازن.
    return s

# aliases لتوافق أُطر مختلفة
def generate_signal_series(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    return cci_zscore_reflex_signals(df, params=params)

def get_signals(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    return cci_zscore_reflex_signals(df, params=params)
