# indicators/obv.py

import pandas as pd
import numpy as np

def calculate_obv_signal(df: pd.DataFrame) -> str:
    """
    On-Balance Volume (OBV) Signal:
    - Buy: OBV صاعد
    - Sell: OBV هابط
    - Hold: غير ذلك
    """
    try:
        if len(df) < 3:
            return "Hold"

        close = df["close"]
        volume = df["volume"]

        obv = [0]
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])

        obv_series = pd.Series(obv)
        if obv_series.iloc[-1] > obv_series.iloc[-2]:
            return "Buy"
        elif obv_series.iloc[-1] < obv_series.iloc[-2]:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_obv_signal: {e}")
        return "Hold"
