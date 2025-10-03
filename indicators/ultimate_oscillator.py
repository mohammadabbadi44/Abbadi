import pandas as pd

def calculate_uo_signals(df: pd.DataFrame, short_period=7, mid_period=14, long_period=28) -> pd.Series:
    """
    Ultimate Oscillator - توليد إشارة شراء/بيع باستخدام مؤشر UO
    """
    bp = df["close"] - df[["low", "close"]].min(axis=1)
    tr = df[["high", "close"]].max(axis=1) - df[["low", "close"]].min(axis=1)

    avg7 = bp.rolling(short_period).sum() / tr.rolling(short_period).sum()
    avg14 = bp.rolling(mid_period).sum() / tr.rolling(mid_period).sum()
    avg28 = bp.rolling(long_period).sum() / tr.rolling(long_period).sum()

    uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7

    signals = []
    for value in uo:
        if value < 30:
            signals.append("Buy")
        elif value > 70:
            signals.append("Sell")
        else:
            signals.append("Hold")
    return pd.Series(signals, index=df.index)