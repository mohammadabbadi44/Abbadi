import pandas as pd
import numpy as np
from typing import Union, List


def get_candle_source(df: pd.DataFrame, source_type: str = "close") -> np.ndarray:
    return df[source_type].values


def slice_candles(df: pd.DataFrame, sequential: bool, period: int = 0):
    return df if sequential else df.iloc[:-period] if period else df


def moving_average(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average (SMA)"""
    ma = np.full(len(data), np.nan)
    for i in range(period - 1, len(data)):
        ma[i] = np.mean(data[i - period + 1:i + 1])
    return ma


def sma(df: pd.DataFrame, period: int = 5, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    """
    SMA - Simple Moving Average

    :param df: pd.DataFrame with columns including 'open', 'high', 'low', 'close', etc.
    :param period: int - SMA period
    :param source_type: str - e.g., "close"
    :param sequential: bool - return full array if True, last value if False
    :return: float | np.ndarray
    """
    df = slice_candles(df, sequential)
    source = get_candle_source(df, source_type=source_type)
    result = moving_average(source, period=period)
    return result if sequential else result[-1]


def calculate_sma_signals(df: pd.DataFrame, period: int = 5, source_type: str = "close") -> List[str]:
    """
    يولّد إشارة تداول باستخدام مؤشر SMA:
    - Buy: إذا المؤشر طالع
    - Sell: إذا المؤشر نازل
    - Hold: غير ذلك
    """
    try:
        values = sma(df, period=period, source_type=source_type, sequential=True)
        signals = []

        for i in range(len(values)):
            if i < 2 or np.isnan(values[i]) or np.isnan(values[i - 1]):
                signals.append("Hold")
            elif values[i] > values[i - 1]:
                signals.append("Buy")
            elif values[i] < values[i - 1]:
                signals.append("Sell")
            else:
                signals.append("Hold")

        return signals

    except Exception as e:
        print(f"[⚠️] Error in calculate_sma_signals: {e}")
        return ["Hold"] * len(df)
