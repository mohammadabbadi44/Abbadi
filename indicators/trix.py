import pandas as pd
import numpy as np
from typing import Union
from numba import njit

def get_candle_source(df: pd.DataFrame, source_type: str = "close") -> np.ndarray:
    return df[source_type].values

def slice_candles(df: pd.DataFrame, sequential: bool, period: int = 0):
    return df if sequential else df.iloc[:-period] if period else df

@njit
def _ema_numba(data, period):
    N = len(data)
    result = np.empty(N, dtype=np.float64)
    alpha = 2.0 / (period + 1)
    result[0] = data[0]
    for i in range(1, N):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result

def ema(data: np.ndarray, period: int) -> np.ndarray:
    return _ema_numba(data, period)

def trix(df: pd.DataFrame, period: int = 18, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
    df = slice_candles(df, sequential)
    source = get_candle_source(df, source_type=source_type)

    log_source = np.log(source)
    ema1 = ema(log_source, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)

    diff = np.empty_like(ema3)
    diff[0] = np.nan
    diff[1:] = ema3[1:] - ema3[:-1]
    result = diff * 10000

    return result if sequential else result[-1]

def calculate_trix_signals(df: pd.DataFrame, period: int = 18) -> list:
    """
    يولّد إشارة تداول باستخدام مؤشر TRIX:
    - Buy إذا كانت القيمة الحالية أكبر من السابقة
    - Sell إذا أقل
    - Hold إذا ما تغيرت أو كانت NaN
    """
    try:
        values = trix(df, period=period, sequential=True)
        signal = []

        for i in range(len(values)):
            if i < 2 or np.isnan(values[i]) or np.isnan(values[i - 1]):
                signal.append("Hold")
            elif values[i] > values[i - 1]:
                signal.append("Buy")
            elif values[i] < values[i - 1]:
                signal.append("Sell")
            else:
                signal.append("Hold")

        return signal
    except Exception as e:
        print(f"[⚠️] Error in calculate_trix_signals: {e}")
        return ["Hold"] * len(df)
