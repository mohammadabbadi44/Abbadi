# indicators/atr.py

import pandas as pd
import numpy as np

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    يحسب ATR - Average True Range
    """
    df = df.copy()

    df['high_low'] = df['high'] - df['low']
    df['high_close'] = (df['high'] - df['close'].shift()).abs()
    df['low_close'] = (df['low'] - df['close'].shift()).abs()

    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()

    return df['atr']


def calculate_atr_signal(df: pd.DataFrame, period: int = 14) -> str:
    """
    يولد إشارة تداول من ATR:
    - Buy: إذا ATR يرتفع (زخم يتزايد)
    - Sell: إذا ATR ينخفض (زخم يضعف)
    - Hold: غير ذلك
    """
    try:
        atr = calculate_atr(df, period=period)

        curr = atr.iloc[-1]
        prev = atr.iloc[-2]

        if np.isnan(curr) or np.isnan(prev):
            return "Hold"

        if curr > prev:
            return "Buy"
        elif curr < prev:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_atr_signal: {e}")
        return "Hold"
