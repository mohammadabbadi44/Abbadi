# strategy/sk_fibonacci.py

import pandas as pd

def calculate_fibonacci_levels(high, low):
    diff = high - low
    retracement_levels = {
        "0.0": low,
        "0.382": low + diff * 0.382,
        "0.5": low + diff * 0.5,
        "0.559": low + diff * 0.559,
        "0.618": low + diff * 0.618,
        "0.667": low + diff * 0.667,
        "0.786": low + diff * 0.786,
        "0.882": low + diff * 0.882,
        "1.0": high,
    }

    extension_levels = {
        "1.0": high + diff * 1.0,
        "1.618": high + diff * 1.618,
        "1.809": high + diff * 1.809,
        "2.0": high + diff * 2.0,
        "2.618": high + diff * 2.618,
    }

    return retracement_levels, extension_levels


def detect_fibonacci_signal(df: pd.DataFrame, tolerance: float = 0.002) -> str:
    """
    يرجع إشارة Buy/Sell/Hold بناءً على قرب السعر من مستويات فيبوناتشي
    """
    if len(df) < 50:
        return "Hold"

    recent_high = df['high'].iloc[-50:].max()
    recent_low = df['low'].iloc[-50:].min()
    close_price = df['close'].iloc[-1]

    retracements, extensions = calculate_fibonacci_levels(recent_high, recent_low)

    for level, price in retracements.items():
        if abs(close_price - price) / price < tolerance:
            return "Buy" if close_price < price else "Sell"

    for level, price in extensions.items():
        if abs(close_price - price) / price < tolerance:
            return "Sell" if close_price > price else "Buy"

    return "Hold"


def generate_signal(df: pd.DataFrame) -> str:
    return detect_fibonacci_signal(df)
