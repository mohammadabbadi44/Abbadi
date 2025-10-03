# indicators/vwma.py

import pandas as pd
import numpy as np

def calculate_vwma_signal(df: pd.DataFrame, period: int = 20) -> str:
    """
    VWMA - Volume Weighted Moving Average
    - Buy: إذا السعر > VWMA
    - Sell: إذا السعر < VWMA
    - Hold: إذا قريب منه
    """
    try:
        if len(df) < period + 2:
            return "Hold"

        price = df["close"]
        volume = df["volume"]

        pv = price * volume
        vwma = pv.rolling(window=period).sum() / volume.rolling(window=period).sum()

        curr_price = price.iloc[-1]
        curr_vwma = vwma.iloc[-1]

        if np.isnan(curr_vwma):
            return "Hold"

        if curr_price > curr_vwma:
            return "Buy"
        elif curr_price < curr_vwma:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_vwma_signal: {e}")
        return "Hold"
