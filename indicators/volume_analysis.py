import pandas as pd

def calculate_volume_signal(df: pd.DataFrame) -> pd.Series:
    """
    تحليل الفوليوم البسيط - مقارنة فوليوم الشمعة الحالية بالمتوسط
    """
    average_volume = df["volume"].rolling(window=20).mean()
    signal = []
    for vol, avg in zip(df["volume"], average_volume):
        if vol > 1.5 * avg:
            signal.append("Buy")
        elif vol < 0.5 * avg:
            signal.append("Sell")
        else:
            signal.append("Hold")
    return pd.Series(signal, index=df.index)