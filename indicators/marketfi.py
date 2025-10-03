# indicators/marketfi.py

import pandas as pd
import numpy as np

def calculate_marketfi_signal(df: pd.DataFrame) -> str:
    """
    MarketFi Volume Confirmation Signal:
    - Buy: إذا الحجم يرتفع والسعر يرتفع (تجميع)
    - Sell: إذا الحجم يرتفع والسعر ينخفض (توزيع)
    - Hold: غير ذلك
    """
    try:
        if len(df) < 3:
            return "Hold"

        close = df["close"]
        volume = df["volume"]

        price_up = close.iloc[-1] > close.iloc[-2]
        volume_up = volume.iloc[-1] > volume.iloc[-2]

        if price_up and volume_up:
            return "Buy"
        elif not price_up and volume_up:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_marketfi_signal: {e}")
        return "Hold"
