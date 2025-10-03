# indicators/frama.py
import numpy as np
import pandas as pd

def frama(close: pd.Series, window: int = 16) -> pd.Series:
    """ FRAMA آمنة وسريعة. """
    close = pd.Series(close, dtype="float64").copy()
    n = len(close)
    if n == 0 or window < 4:
        return pd.Series([np.nan] * n, index=close.index)

    h = window // 2
    roll_max_w  = close.rolling(window, min_periods=window).max()
    roll_min_w  = close.rolling(window, min_periods=window).min()
    roll_max_h1 = close.rolling(h,      min_periods=h).max()
    roll_min_h1 = close.rolling(h,      min_periods=h).min()
    roll_max_h2 = close.shift(h).rolling(h, min_periods=h).max()
    roll_min_h2 = close.shift(h).rolling(h, min_periods=h).min()

    N1 = (roll_max_h1 - roll_min_h1).replace(0.0, np.nan)
    N2 = (roll_max_h2 - roll_min_h2).replace(0.0, np.nan)
    N3 = (roll_max_w  - roll_min_w ).replace(0.0, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        D = (np.log(N1 + N2) - np.log(N3)) / np.log(2.0)
    D = pd.Series(D, index=close.index).clip(1.0, 2.0)

    alpha = np.exp(-4.6 * (D - 1.0))
    alpha = pd.Series(alpha, index=close.index).replace([np.inf, -np.inf], np.nan).fillna(0.2).clip(0.001, 1.0)

    fr = pd.Series(np.nan, index=close.index, dtype="float64")
    start = window - 1
    if start >= n:
        return fr
    fr.iloc[start] = close.iloc[start]
    for i in range(start + 1, n):
        a = float(alpha.iloc[i]); c = float(close.iloc[i]); prev = fr.iloc[i - 1]
        if not np.isfinite(prev):
            prev = float(close.iloc[i - 1]); fr.iloc[i - 1] = prev
        val = a * c + (1.0 - a) * prev
        if not np.isfinite(val) or abs(val) > 1e12:  # حارس أمان
            val = prev
        fr.iloc[i] = val
    return fr

def calculate_frama_series(df: pd.DataFrame, price_col: str = "close", window: int = 16) -> pd.Series:
    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col}")
    return frama(df[price_col].astype("float64"), window=window)

def calculate_frama_signal(df: pd.DataFrame,
                           price_col: str = "close",
                           window: int = 16) -> pd.Series:
    """
    إشارات FRAMA سريعة بدون لوب ثقيل:
    - Buy عند اختراق السعر أعلى FRAMA
    - Sell عند كسر السعر أسفل FRAMA
    - Hold خلاف ذلك
    """
    fr = calculate_frama_series(df, price_col=price_col, window=window)
    close = df[price_col].astype("float64")

    sig = pd.Series("Hold", index=df.index)
    valid = fr.notna()
    cross_up = (close.shift(1) <= fr.shift(1)) & (close > fr) & valid & valid.shift(1)
    cross_dn = (close.shift(1) >= fr.shift(1)) & (close < fr) & valid & valid.shift(1)

    sig = sig.mask(cross_up, "Buy").mask(cross_dn, "Sell")
    return sig
