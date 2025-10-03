# indicators/hull_ma.py

import pandas as pd
import numpy as np

def calculate_hull_ma_signal(df: pd.DataFrame, period: int = 21) -> str:
    """
    Hull Moving Average Signal:
    - Buy: إذا السعر أغلق فوق HMA
    - Sell: إذا السعر أغلق تحت HMA
    - Hold: غير ذلك
    """
    try:
        if len(df) < period + 5:
            return "Hold"

        close = df['close']
        half_length = int(period / 2)
        sqrt_length = int(np.sqrt(period))

        wma_half = close.rolling(window=half_length).apply(lambda x: np.average(x, weights=range(1, half_length + 1)))
        wma_full = close.rolling(window=period).apply(lambda x: np.average(x, weights=range(1, period + 1)))

        raw_hma = 2 * wma_half - wma_full
        hma = raw_hma.rolling(window=sqrt_length).apply(lambda x: np.average(x, weights=range(1, sqrt_length + 1)))

        curr_price = close.iloc[-1]
        curr_hma = hma.iloc[-1]

        if np.isnan(curr_hma):
            return "Hold"

        if curr_price > curr_hma:
            return "Buy"
        elif curr_price < curr_hma:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_hull_ma_signal: {e}")
        return "Hold"
