import pandas as pd

def detect_liquidity_grabs(df: pd.DataFrame) -> pd.Series:
    """
    يكشف عن Liquidity Grabs:
    - Sell Trap: كسر قمة سابقة ثم إغلاق هبوطي
    - Buy Trap: كسر قاع سابق ثم إغلاق صعودي
    - None: لا يوجد فخ واضح
    """
    signals = ["None"] * len(df)

    for i in range(1, len(df)):
        if df['high'].iloc[i] > df['high'].iloc[i - 1] and df['close'].iloc[i] < df['open'].iloc[i]:
            signals[i] = "Sell Trap"
        elif df['low'].iloc[i] < df['low'].iloc[i - 1] and df['close'].iloc[i] > df['open'].iloc[i]:
            signals[i] = "Buy Trap"

    return pd.Series(signals, index=df.index)
