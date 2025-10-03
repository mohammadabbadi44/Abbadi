# indicators/momentum.py

import pandas as pd
import numpy as np

def calculate_momentum_signal(df: pd.DataFrame, period: int = 10) -> str:
    """
    Momentum Signal:
    - Buy: إذا الزخم اخترق من سالب إلى موجب
    - Sell: إذا الزخم اخترق من موجب إلى سالب
    - Hold: غير ذلك
    """
    try:
        if len(df) < period + 2:
            return "Hold"

        close = df['close']
        momentum = close - close.shift(period)

        prev = momentum.iloc[-2]
        curr = momentum.iloc[-1]

        if np.isnan(prev) or np.isnan(curr):
            return "Hold"

        if prev < 0 and curr > 0:
            return "Buy"
        elif prev > 0 and curr < 0:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_momentum_signal: {e}")
        return "Hold"
