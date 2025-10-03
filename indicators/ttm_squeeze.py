# indicators/ttm_squeeze.py

import pandas as pd
import numpy as np

def calculate_ttm_squeeze_signal(df: pd.DataFrame, bb_period: int = 20, kc_period: int = 20, mult: float = 1.5) -> str:
    """
    TTM Squeeze Signal:
    - Buy: كسر ضغط (squeeze) مع مومنتوم إيجابي
    - Sell: كسر ضغط مع مومنتوم سلبي
    - Hold: إذا لسه مضغوط أو مافي وضوح
    """
    try:
        if len(df) < bb_period + kc_period + 2:
            return "Hold"

        close = df['close']
        high = df['high']
        low = df['low']

        # Bollinger Bands
        sma = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        upper_bb = sma + 2 * std
        lower_bb = sma - 2 * std

        # Keltner Channels
        typical_price = (high + low + close) / 3
        range_kc = typical_price.rolling(kc_period).mean()
        tr = high.combine(low, max) - high.combine(low, min)
        atr = tr.rolling(kc_period).mean()
        upper_kc = range_kc + mult * atr
        lower_kc = range_kc - mult * atr

        # Squeeze ON → إذا BB داخل KC
        squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        # Squeeze OFF → إذا BB خرجت من KC
        squeeze_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)

        # Momentum = فرق آخر إغلاقين
        momentum = close.diff()

        if squeeze_off.iloc[-1]:
            if momentum.iloc[-1] > 0:
                return "Buy"
            elif momentum.iloc[-1] < 0:
                return "Sell"
        return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_ttm_squeeze_signal: {e}")
        return "Hold"
