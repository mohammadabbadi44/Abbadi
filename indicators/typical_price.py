# indicators/typical_price.py

import pandas as pd

def calculate_typical_price_signal(df: pd.DataFrame) -> str:
    """
    Typical Price Signal:
    - Buy: إذا Typical Price ارتفع
    - Sell: إذا انخفض
    - Hold: إذا لم يتغير
    """
    try:
        if len(df) < 2:
            return "Hold"

        typical_price = (df['high'] + df['low'] + df['close']) / 3

        prev = typical_price.iloc[-2]
        curr = typical_price.iloc[-1]

        if curr > prev:
            return "Buy"
        elif curr < prev:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_typical_price_signal: {e}")
        return "Hold"
