# indicators/rsi.py

import pandas as pd
import numpy as np
from typing import Union

def get_candle_source(df: pd.DataFrame, source_type: str = "close") -> np.ndarray:
    return df[source_type].values

def rsi(candles: Union[np.ndarray, pd.DataFrame], period: int = 14, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    if isinstance(candles, pd.DataFrame):
        source = get_candle_source(candles, source_type)
    else:
        source = candles

    delta = np.diff(source)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # استخدام float64 لتفادي مشاكل dtype
    avg_gain = np.zeros_like(source, dtype=np.float64)
    avg_loss = np.zeros_like(source, dtype=np.float64)

    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])

    for i in range(period + 1, len(source)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain, dtype=np.float64), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan

    return rsi if sequential else rsi[-1]

def calculate_rsi_signal(df: pd.DataFrame, period: int = 14) -> str:
    """
    RSI Signal (واحدة فقط):
    - Buy: إذا RSI < 30
    - Sell: إذا RSI > 70
    - Hold: بينهما
    """
    try:
        value = rsi(df, period=period, source_type='close', sequential=False)

        if np.isnan(value):
            return "Hold"
        elif value < 30:
            return "Buy"
        elif value > 70:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_rsi_signal: {e}")
        return "Hold"
