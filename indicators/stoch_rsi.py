# indicators/stoch_rsi.py

import pandas as pd
import numpy as np

def calculate_stoch_rsi_signal(df: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14) -> str:
    """
    Stochastic RSI Signal:
    - Buy: إذا SRSI اخترق فوق 0.2 من الأسفل
    - Sell: إذا SRSI اخترق تحت 0.8 من الأعلى
    - Hold: غير ذلك
    """
    try:
        if len(df) < rsi_period + stoch_period + 2:
            return "Hold"

        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(window=rsi_period).mean()
        avg_loss = pd.Series(loss).rolling(window=rsi_period).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        min_rsi = rsi.rolling(window=stoch_period).min()
        max_rsi = rsi.rolling(window=stoch_period).max()

        srsi = (rsi - min_rsi) / (max_rsi - min_rsi)

        prev = srsi.iloc[-2]
        curr = srsi.iloc[-1]

        if np.isnan(prev) or np.isnan(curr):
            return "Hold"

        if prev < 0.2 and curr > 0.2:
            return "Buy"
        elif prev > 0.8 and curr < 0.8:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_stoch_rsi_signal: {e}")
        return "Hold"
