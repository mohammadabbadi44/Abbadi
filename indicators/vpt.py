# indicators/vpt.py

import pandas as pd
import numpy as np

def calculate_vpt_signal(df: pd.DataFrame) -> str:
    """
    Volume Price Trend (VPT) Signal:
    - Buy: إذا VPT صاعد
    - Sell: إذا VPT هابط
    - Hold: غير ذلك
    """
    try:
        if len(df) < 3:
            return "Hold"

        close = df['close']
        volume = df['volume']

        vpt = [0]
        for i in range(1, len(df)):
            pct_change = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]
            vpt.append(vpt[-1] + volume.iloc[i] * pct_change)

        vpt_series = pd.Series(vpt)
        if np.isnan(vpt_series.iloc[-1]) or np.isnan(vpt_series.iloc[-2]):
            return "Hold"

        if vpt_series.iloc[-1] > vpt_series.iloc[-2]:
            return "Buy"
        elif vpt_series.iloc[-1] < vpt_series.iloc[-2]:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_vpt_signal: {e}")
        return "Hold"
