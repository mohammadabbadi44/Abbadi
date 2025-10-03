# indicators/bollinger.py

import pandas as pd

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    يحسب Bollinger Bands: Upper, Middle, Lower
    """
    close = df['close']
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper = sma + std_dev * std
    lower = sma - std_dev * std

    return pd.DataFrame({
        "upper": upper,
        "middle": sma,
        "lower": lower
    })


def calculate_bollinger_signals(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """
    يولّد إشارات Bollinger:
    - Buy: إذا السعر تحت Lower Band
    - Sell: إذا السعر فوق Upper Band
    - Hold: في غير ذلك
    """
    bands = calculate_bollinger_bands(df, period, std_dev)
    signals = ["Hold"] * len(df)
    close = df['close']

    for i in range(period, len(df)):
        if close[i] < bands['lower'][i]:
            signals[i] = "Buy"
        elif close[i] > bands['upper'][i]:
            signals[i] = "Sell"
        else:
            signals[i] = "Hold"

    return pd.Series(signals)