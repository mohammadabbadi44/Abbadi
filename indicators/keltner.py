# indicators/keltner.py

import pandas as pd
import numpy as np
from collections import namedtuple
from numba import njit

KeltnerChannel = namedtuple('KeltnerChannel', ['upperband', 'middleband', 'lowerband'])

@njit
def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    tr = np.empty(n)
    atr_vals = np.full(n, np.nan)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    if n < period:
        return atr_vals

    atr_vals[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr[i]) / period

    return atr_vals

def calculate_keltner_signal(df: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> str:
    """
    يولّد إشارة واحدة فقط:
    - Buy: إذا السعر أغلق تحت lower band
    - Sell: إذا أغلق فوق upper band
    - Hold: بين القناتين
    """
    if len(df) < period + 2:
        return "Hold"

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    ema = df['close'].ewm(span=period, adjust=False).mean().values
    atr = _atr(high, low, close, period)

    upper = ema + atr * multiplier
    lower = ema - atr * multiplier

    latest_close = close[-1]
    upper_val = upper[-2]
    lower_val = lower[-2]

    if np.isnan(upper_val) or np.isnan(lower_val):
        return "Hold"

    if latest_close > upper_val:
        return "Sell"
    elif latest_close < lower_val:
        return "Buy"
    else:
        return "Hold"
