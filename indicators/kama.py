# indicators/kama.py

import pandas as pd
import numpy as np

def calculate_kama_signal(df: pd.DataFrame, period: int = 10, fast: int = 2, slow: int = 30) -> str:
    """
    KAMA Signal:
    - Buy: السعر اخترق KAMA من الأسفل
    - Sell: السعر كسر KAMA من الأعلى
    - Hold: غير ذلك
    """
    try:
        if len(df) < period + 2:
            return "Hold"

        close = df["close"]
        change = abs(close - close.shift(period))
        volatility = close.diff().abs().rolling(window=period).sum()
        er = change / (volatility + 1e-9)  # Efficiency Ratio

        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2  # Smoothing Constant

        kama = close.copy()
        for i in range(period, len(close)):
            kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i - 1])

        curr_price = close.iloc[-1]
        prev_price = close.iloc[-2]
        curr_kama = kama.iloc[-1]
        prev_kama = kama.iloc[-2]

        if prev_price < prev_kama and curr_price > curr_kama:
            return "Buy"
        elif prev_price > prev_kama and curr_price < curr_kama:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_kama_signal: {e}")
        return "Hold"
