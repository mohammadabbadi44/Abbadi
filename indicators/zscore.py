# indicators/zscore.py

import pandas as pd
import numpy as np

def calculate_zscore_signal(df: pd.DataFrame, period: int = 20, threshold: float = 1.0) -> str:
    """
    Z-Score Signal:
    - Buy: Z-Score < -threshold
    - Sell: Z-Score > threshold
    - Hold: otherwise
    """
    try:
        if len(df) < period + 2:
            return "Hold"

        close = df["close"]
        mean = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        zscore = (close - mean) / std

        z = zscore.iloc[-1]

        if np.isnan(z):
            return "Hold"
        elif z < -threshold:
            return "Buy"
        elif z > threshold:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_zscore_signal: {e}")
        return "Hold"
