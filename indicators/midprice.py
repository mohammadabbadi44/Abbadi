# indicators/midprice.py

import pandas as pd
import numpy as np

def calculate_midprice_signal(df: pd.DataFrame, period: int = 20) -> str:
    """
    يولّد إشارة تداول باستخدام MidPrice:
    - Buy: إذا السعر الحالي تحت MidPrice - انحراف معين
    - Sell: إذا السعر فوق MidPrice + انحراف معين
    - Hold: إذا قريب من منتصف النطاق
    """
    if len(df) < period + 2:
        return "Hold"

    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        mid = (high + low) / 2
        mid_avg = mid.rolling(window=period).mean()

        current_close = close.iloc[-1]
        current_mid = mid_avg.iloc[-2]

        if np.isnan(current_mid):
            return "Hold"

        diff = current_close - current_mid
        deviation = df["close"].rolling(window=period).std().iloc[-2]

        if diff < -deviation:
            return "Buy"
        elif diff > deviation:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_midprice_signal: {e}")
        return "Hold"
