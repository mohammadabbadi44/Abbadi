# indicators/reflex.py

import pandas as pd
import numpy as np

def calculate_reflex_signal(df: pd.DataFrame, period: int = 9) -> str:
    """
    مؤشر Reflex يحاكي Price Inversion/Turning Points.
    - Buy: إذا Reflex اخترق للأعلى بعد انعكاس هابط
    - Sell: إذا Reflex اخترق للأسفل بعد انعكاس صاعد
    - Hold: غير ذلك
    """
    if df.shape[0] < period + 2:
        return "Hold"

    price = df['close']
    reflex = 2 * price - price.shift(period)
    
    prev = reflex.iloc[-2]
    curr = reflex.iloc[-1]

    if prev < 0 and curr > 0:
        return "Buy"
    elif prev > 0 and curr < 0:
        return "Sell"
    else:
        return "Hold"
