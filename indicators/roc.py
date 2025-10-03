# indicators/roc.py

import pandas as pd
import numpy as np

def calculate_roc_signal(df: pd.DataFrame, period: int = 12) -> str:
    """
    Rate of Change (ROC) Signal:
    - Buy: إذا ROC اخترق من سالب إلى موجب
    - Sell: إذا ROC اخترق من موجب إلى سالب
    - Hold: غير ذلك
    """
    try:
        if len(df) < period + 2:
            return "Hold"

        close = df["close"]
        roc = ((close - close.shift(period)) / close.shift(period)) * 100

        prev = roc.iloc[-2]
        curr = roc.iloc[-1]

        if np.isnan(prev) or np.isnan(curr):
            return "Hold"

        if prev < 0 and curr > 0:
            return "Buy"
        elif prev > 0 and curr < 0:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_roc_signal: {e}")
        return "Hold"
