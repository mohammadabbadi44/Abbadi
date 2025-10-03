# indicators/cci.py

import pandas as pd
import numpy as np
from numba import njit

@njit(cache=True)
def calculate_cci_loop(tp, period):
    n = tp.shape[0]
    result = np.empty(n)
    result[:] = np.nan
    if n < period:
        return result
    for i in range(period - 1, n):
        sma = np.mean(tp[i - period + 1:i + 1])
        md = np.mean(np.abs(tp[i - period + 1:i + 1] - sma))
        result[i] = (tp[i] - sma) / (0.015 * md) if md != 0 else 0.0
    return result

def calculate_cci_signal(df: pd.DataFrame, period: int = 14) -> str:
    """
    يولّد إشارة تداول باستخدام CCI:
    - Buy: إذا اخترق من تحت -100 إلى فوقها
    - Sell: إذا نزل من فوق +100 إلى تحتها
    - Hold: غير ذلك
    """
    try:
        if len(df) < period + 2:
            return "Hold"

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        tp = (high + low + close) / 3.0

        values = calculate_cci_loop(tp, period)

        prev = values[-2]
        curr = values[-1]

        if np.isnan(prev) or np.isnan(curr):
            return "Hold"

        if prev < -100 and curr > -100:
            return "Buy"
        elif prev > 100 and curr < 100:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_cci_signal: {e}")
        return "Hold"
