# smc/bos.py

import pandas as pd

def detect_bos(df: pd.DataFrame, lookback: int = 20) -> str:
    """
    يكشف إذا حدث كسر هيكل خلال آخر 'lookback' شمعة
    - يعيد "Bullish BOS" أو "Bearish BOS" أو "None"
    """
    try:
        recent_high = df['high'].iloc[-lookback:-1].max()
        recent_low = df['low'].iloc[-lookback:-1].min()
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]

        if current_high > recent_high:
            return "Bullish BOS"
        elif current_low < recent_low:
            return "Bearish BOS"
        else:
            return "None"

    except Exception as e:
        from logs.logger import log_error
        log_error(f"BOS Detection Error: {e}")
        return "None"
