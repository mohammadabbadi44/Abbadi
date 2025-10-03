import pandas as pd
import numpy as np

def calculate_tsi_signal(df: pd.DataFrame, long: int = 25, short: int = 13) -> str:
    """
    TSI (True Strength Index) Signal Generator:
    - Buy: إذا TSI اخترق من تحت 0 إلى فوقه
    - Sell: إذا TSI اخترق من فوق 0 إلى تحته
    - Hold: غير ذلك
    """
    try:
        if len(df) < long + short + 2:
            return "Hold"

        price_diff = df['close'].diff()
        double_smoothed = price_diff.ewm(span=short, adjust=False).mean().ewm(span=long, adjust=False).mean()
        abs_smoothed = price_diff.abs().ewm(span=short, adjust=False).mean().ewm(span=long, adjust=False).mean()

        tsi = 100 * (double_smoothed / abs_smoothed)
        tsi = tsi.fillna(0)

        prev = tsi.iloc[-2]
        curr = tsi.iloc[-1]

        if prev < 0 and curr > 0:
            return "Buy"
        elif prev > 0 and curr < 0:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_tsi_signal: {e}")
        return "Hold"
