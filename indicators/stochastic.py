# indicators/stochastic.py

import pandas as pd
import numpy as np
from collections import namedtuple

Stochastic = namedtuple('Stochastic', ['k', 'd'])

def get_candle_source(df: pd.DataFrame, source_type: str = "close") -> np.ndarray:
    return df[source_type].values

def stoch(df: pd.DataFrame, fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3) -> Stochastic:
    """
    The Stochastic Oscillator
    """
    if len(df) < fastk_period + slowk_period + slowd_period:
        return Stochastic(np.nan, np.nan)

    low_min = df['low'].rolling(window=fastk_period).min()
    high_max = df['high'].rolling(window=fastk_period).max()
    fast_k = 100 * ((df['close'] - low_min) / (high_max - low_min))

    slow_k = fast_k.rolling(window=slowk_period).mean()
    slow_d = slow_k.rolling(window=slowd_period).mean()

    return Stochastic(slow_k.iloc[-1], slow_d.iloc[-1])

def calculate_stochastic_signal(df: pd.DataFrame) -> str:
    """
    يولّد إشارة تداول باستخدام STOCHASTIC:
    - Buy: إذا %K قطعت فوق %D
    - Sell: إذا %K قطعت تحت %D
    - Hold: باقي الحالات
    """
    try:
        stoch_now = stoch(df)

        k = stoch_now.k
        d = stoch_now.d

        if np.isnan(k) or np.isnan(d):
            return "Hold"

        prev = stoch(df.iloc[:-1])
        prev_k = prev.k
        prev_d = prev.d

        if np.isnan(prev_k) or np.isnan(prev_d):
            return "Hold"

        if prev_k < prev_d and k > d:
            return "Buy"
        elif prev_k > prev_d and k < d:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_stochastic_signal: {e}")
        return "Hold"
