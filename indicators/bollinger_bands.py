# indicators/bollinger_bands.py

import pandas as pd
import numpy as np

def calculate_bollinger_signal(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> str:
    """
    Bollinger Bands Signal:
    - Buy: السعر تحت Lower Band
    - Sell: السعر فوق Upper Band
    - Hold: غير ذلك
    """
    try:
        if len(df) < period + 2:
            return "Hold"

        close = df["close"]
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()

        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)

        curr_close = close.iloc[-1]
        curr_upper = upper_band.iloc[-1]
        curr_lower = lower_band.iloc[-1]

        if np.isnan(curr_upper) or np.isnan(curr_lower):
            return "Hold"

        if curr_close > curr_upper:
            return "Sell"
        elif curr_close < curr_lower:
            return "Buy"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_bollinger_signal: {e}")
        return "Hold"
