import pandas as pd
import numpy as np

def calculate_smi_signals(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']

    min_low = low.rolling(window=period).min()
    max_high = high.rolling(window=period).max()
    center = (max_high + min_low) / 2
    diff = close - center

    diff_smoothed = diff.rolling(window=smooth_k).mean()
    range_smoothed = (max_high - min_low).rolling(window=smooth_k).mean()

    smi = 100 * (diff_smoothed / (range_smoothed / 2))
    signal = smi.rolling(window=smooth_d).mean()

    signals = []
    for i in range(len(df)):
        if smi[i] > signal[i] and smi[i] < -40:
            signals.append("Buy")
        elif smi[i] < signal[i] and smi[i] > 40:
            signals.append("Sell")
        else:
            signals.append("Hold")
    return pd.Series(signals, index=df.index)