import pandas as pd
import numpy as np
from typing import Union, List

def get_candle_source(df: pd.DataFrame, source_type: str = "close") -> np.ndarray:
    return df[source_type].values

def slice_candles(df: pd.DataFrame, sequential: bool, period: int = 0):
    return df if sequential else df.iloc[:-period] if period else df

def adx(df: pd.DataFrame, period: int = 14, sequential: bool = False) -> Union[float, np.ndarray]:
    df = slice_candles(df, sequential)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    tr = np.zeros_like(close)
    plus_dm = np.zeros_like(close)
    minus_dm = np.zeros_like(close)

    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0

    atr = np.full_like(close, np.nan, dtype=np.float64)
    plus_di = np.full_like(close, np.nan, dtype=np.float64)
    minus_di = np.full_like(close, np.nan, dtype=np.float64)
    dx = np.full_like(close, np.nan, dtype=np.float64)

    if len(close) > period:
        atr[period] = np.mean(tr[1:period+1])
        plus_di[period] = 100 * np.mean(plus_dm[1:period+1]) / atr[period]
        minus_di[period] = 100 * np.mean(minus_dm[1:period+1]) / atr[period]
        dx[period] = 100 * abs(plus_di[period] - minus_di[period]) / (plus_di[period] + minus_di[period] + 1e-9)

        for i in range(period + 1, len(close)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            plus_avg = np.mean(plus_dm[i - period + 1:i + 1])
            minus_avg = np.mean(minus_dm[i - period + 1:i + 1])
            plus_di[i] = 100 * (plus_avg / atr[i]) if atr[i] != 0 else 0
            minus_di[i] = 100 * (minus_avg / atr[i]) if atr[i] != 0 else 0
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i] + 1e-9)

    adx_values = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) > period * 2:
        adx_values[period * 2] = np.mean(dx[period:period * 2 + 1])
        for i in range(period * 2 + 1, len(close)):
            adx_values[i] = ((adx_values[i - 1] * (period - 1)) + dx[i]) / period

    return adx_values if sequential else adx_values[-1]

def calculate_adx_signal(df: pd.DataFrame, period: int = 14) -> List[str]:
    try:
        values = adx(df, period=period, sequential=True)
        signals = []

        for i in range(len(values)):
            if i < period * 2 or np.isnan(values[i]) or np.isnan(values[i - 1]):
                signals.append("Hold")
            elif values[i] > values[i - 1]:
                signals.append("Buy")
            elif values[i] < values[i - 1]:
                signals.append("Sell")
            else:
                signals.append("Hold")

        return signals

    except Exception as e:
        print(f"[⚠️] Error in calculate_adx_signal: {e}")
        return ["Hold"] * len(df)
