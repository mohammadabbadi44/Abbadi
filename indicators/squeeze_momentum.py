# indicators/squeeze_momentum.py

import pandas as pd
import numpy as np

def calculate_squeeze_momentum_signal(df: pd.DataFrame, period: int = 12) -> str:
    """
    Squeeze Momentum Signal:
    - Buy: إذا المومنتوم موجب ويتزايد
    - Sell: إذا المومنتوم سالب ويتزايد بالسالب
    - Hold: غير ذلك
    """
    try:
        if len(df) < period + 3:
            return "Hold"

        close = df["close"]

        # Momentum = الفرق بين الإغلاق الحالي والمتوسط السابق
        mom = close - close.rolling(window=period).mean()
        mom = mom.dropna()

        if len(mom) < 3:
            return "Hold"

        last = mom.iloc[-1]
        prev = mom.iloc[-2]

        if last > 0 and last > prev:
            return "Buy"
        elif last < 0 and last < prev:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_squeeze_momentum_signal: {e}")
        return "Hold"
