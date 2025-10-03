# indicators/cmo.py

import pandas as pd
import numpy as np

def calculate_cmo_signal(df: pd.DataFrame, period: int = 14) -> str:
    """
    CMO Signal (Chande Momentum Oscillator):
    - Buy: إذا CMO < -50
    - Sell: إذا CMO > +50
    - Hold: غير ذلك
    """
    try:
        if len(df) < period + 2:
            return "Hold"

        close = df["close"].values
        delta = np.diff(close)

        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        sum_gains = pd.Series(gains).rolling(window=period).sum()
        sum_losses = pd.Series(losses).rolling(window=period).sum()

        cmo_series = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
        cmo = cmo_series.iloc[-1]

        if np.isnan(cmo):
            return "Hold"
        elif cmo < -50:
            return "Buy"
        elif cmo > 50:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_cmo_signal: {e}")
        return "Hold"
