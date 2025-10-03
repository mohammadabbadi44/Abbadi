# indicators/ichimoku.py

import pandas as pd
import numpy as np

def calculate_ichimoku_signal(df: pd.DataFrame) -> str:
    """
    Ichimoku Cloud Signal:
    - Buy: إذا السعر فوق السحابة و Tenkan > Kijun
    - Sell: إذا السعر تحت السحابة و Tenkan < Kijun
    - Hold: غير ذلك
    """
    try:
        if len(df) < 52 + 2:
            return "Hold"

        high = df['high']
        low = df['low']
        close = df['close']

        # Tenkan-sen (9)
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        # Kijun-sen (26)
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        # Senkou Span A
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        # Senkou Span B
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

        curr_close = close.iloc[-1]
        curr_tenkan = tenkan.iloc[-1]
        curr_kijun = kijun.iloc[-1]
        curr_senkou_a = senkou_a.iloc[-1]
        curr_senkou_b = senkou_b.iloc[-1]

        if np.isnan(curr_tenkan) or np.isnan(curr_kijun) or np.isnan(curr_senkou_a) or np.isnan(curr_senkou_b):
            return "Hold"

        if curr_close > max(curr_senkou_a, curr_senkou_b) and curr_tenkan > curr_kijun:
            return "Buy"
        elif curr_close < min(curr_senkou_a, curr_senkou_b) and curr_tenkan < curr_kijun:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_ichimoku_signal: {e}")
        return "Hold"
