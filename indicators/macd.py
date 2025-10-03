import pandas as pd
import numpy as np

def calculate_macd_signal(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9) -> str:
    """
    MACD Signal Generator:
    - Buy: MACD line crosses above Signal line
    - Sell: MACD line crosses below Signal line
    - Hold: otherwise
    """
    try:
        if len(df) < slow + signal_period:
            return "Hold"

        close = df['close']
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        prev_macd = macd_line.iloc[-2]
        curr_macd = macd_line.iloc[-1]
        prev_signal = signal_line.iloc[-2]
        curr_signal = signal_line.iloc[-1]

        if prev_macd < prev_signal and curr_macd > curr_signal:
            return "Buy"
        elif prev_macd > prev_signal and curr_macd < curr_signal:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_macd_signal: {e}")
        return "Hold"
