# indicators/volume_oscillator.py

import pandas as pd
import numpy as np

def calculate_volume_oscillator_signal(df: pd.DataFrame, short: int = 14, long: int = 28) -> str:
    """
    Volume Oscillator Signal:
    - Buy: إذا متوسط حجم التداول القصير > الطويل
    - Sell: إذا القصير < الطويل
    - Hold: إذا قريبين أو NaN
    """
    try:
        if len(df) < long + 2:
            return "Hold"

        vol = df['volume']

        short_ma = vol.rolling(window=short).mean()
        long_ma = vol.rolling(window=long).mean()

        curr_short = short_ma.iloc[-1]
        curr_long = long_ma.iloc[-1]

        if np.isnan(curr_short) or np.isnan(curr_long):
            return "Hold"

        if curr_short > curr_long:
            return "Buy"
        elif curr_short < curr_long:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_volume_oscillator_signal: {e}")
        return "Hold"
