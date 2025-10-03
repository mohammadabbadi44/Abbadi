# indicators/srsi.py

import pandas as pd
import numpy as np
from indicators.rsi import rsi

def calculate_srsi_signal(df: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14) -> str:
    """
    Stochastic RSI Signal:
    - Buy: إذا SRSI اخترق فوق 0.2 من الأسفل
    - Sell: إذا SRSI اخترق تحت 0.8 من الأعلى
    - Hold: غير ذلك
    """
    try:
        if len(df) < rsi_period + stoch_period + 2:
            return "Hold"

        rsi_values = rsi(df, period=rsi_period, source_type='close', sequential=True)
        rsi_series = pd.Series(rsi_values)

        lowest = rsi_series.rolling(window=stoch_period).min()
        highest = rsi_series.rolling(window=stoch_period).max()

        srsi = (rsi_series - lowest) / (highest - lowest)

        prev = srsi.iloc[-2]
        curr = srsi.iloc[-1]

        if np.isnan(prev) or np.isnan(curr):
            return "Hold"

        if prev < 0.2 and curr > 0.2:
            return "Buy"
        elif prev > 0.8 and curr < 0.8:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_srsi_signal: {e}")
        return "Hold"
