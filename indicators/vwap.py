# indicators/vwap.py

import pandas as pd
import numpy as np
from typing import Union

def get_candle_source(df: pd.DataFrame, source_type: str = "close") -> np.ndarray:
    if source_type == "hlc3":
        return (df["high"] + df["low"] + df["close"]) / 3
    return df[source_type].values

def slice_candles(df: pd.DataFrame, sequential: bool, period: int = 0):
    return df if sequential else df.iloc[:-period] if period else df

def _calculate_vwap(source: np.ndarray, volume: np.ndarray, group_indices: np.ndarray) -> np.ndarray:
    vwap_values = np.zeros_like(source)
    cum_vol = 0.0
    cum_vol_price = 0.0
    current_group = group_indices[0]
    
    for i in range(len(source)):
        if group_indices[i] != current_group:
            cum_vol = 0.0
            cum_vol_price = 0.0
            current_group = group_indices[i]
        
        vol_price = volume[i] * source[i]
        cum_vol_price += vol_price
        cum_vol += volume[i]
        
        vwap_values[i] = cum_vol_price / cum_vol if cum_vol != 0 else np.nan
        
    return vwap_values

def vwap(df: pd.DataFrame, source_type: str = "hlc3", anchor: str = "D", sequential: bool = False) -> Union[float, np.ndarray]:
    df = slice_candles(df, sequential)
    source = get_candle_source(df, source_type=source_type)

    timestamps = pd.to_datetime(df["time"]).dt.floor(anchor)
    group_indices = np.zeros(len(timestamps), dtype=np.int64)
    group_indices[1:] = (timestamps[1:].values != timestamps[:-1].values).astype(np.int64)
    group_indices = np.cumsum(group_indices)

    vwap_values = _calculate_vwap(source.values, df["volume"].values, group_indices)
    return vwap_values if sequential else vwap_values[-1]

def calculate_vwap_signal(df: pd.DataFrame) -> str:
    """
    ترجع إشارة واحدة من مؤشر VWAP
    """
    try:
        values = vwap(df, sequential=True)
        if len(values) < 2 or np.isnan(values[-1]) or np.isnan(values[-2]):
            return "Hold"
        elif values[-1] > values[-2]:
            return "Buy"
        elif values[-1] < values[-2]:
            return "Sell"
        else:
            return "Hold"
    except Exception as e:
        print(f"[⚠️] Error in calculate_vwap_signal: {e}")
        return "Hold"
