# indicators/dpo.py

import pandas as pd
import numpy as np

def calculate_dpo_signal(df: pd.DataFrame, period: int = 20) -> str:
    """
    Detrended Price Oscillator (DPO):
    - Buy: إذا DPO اخترق من سالب إلى موجب
    - Sell: إذا DPO اخترق من موجب إلى سالب
    - Hold: غير ذلك
    """
    try:
        if len(df) < period + 2:
            return "Hold"

        close = df["close"]
        shift = int(period / 2) + 1
        sma = close.rolling(window=period).mean()
        dpo = close.shift(shift) - sma

        prev = dpo.iloc[-2]
        curr = dpo.iloc[-1]

        if np.isnan(prev) or np.isnan(curr):
            return "Hold"

        if prev < 0 and curr > 0:
            return "Buy"
        elif prev > 0 and curr < 0:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_dpo_signal: {e}")
        return "Hold"
