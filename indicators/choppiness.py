# indicators/choppiness.py

import pandas as pd
import numpy as np

def calculate_chop_signal(df: pd.DataFrame, period: int = 14) -> str:
    """
    Choppiness Index Signal:
    - Buy: إذا CHOP < 38 (سوق ترندي)
    - Sell: إذا CHOP > 61 (سوق متذبذب)
    - Hold: غير ذلك
    """
    try:
        if len(df) < period + 2:
            return "Hold"

        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=1).mean()

        sum_tr = atr.rolling(window=period).sum()
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        chop = 100 * np.log10(sum_tr / (highest_high - lowest_low + 1e-9)) / np.log10(period)

        curr_chop = chop.iloc[-1]

        if np.isnan(curr_chop):
            return "Hold"
        elif curr_chop < 38:
            return "Buy"
        elif curr_chop > 61:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_chop_signal: {e}")
        return "Hold"
