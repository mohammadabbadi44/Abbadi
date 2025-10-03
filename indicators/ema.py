import pandas as pd

def calculate_ema_signal(df: pd.DataFrame, short: int = 20, long: int = 50) -> str:
    """
    EMA Signal:
    - Buy: EMA20 > EMA50
    - Sell: EMA20 < EMA50
    - Hold: غير ذلك
    """
    try:
        if len(df) < long + 2:
            return "Hold"

        ema_short = df['close'].ewm(span=short, adjust=False).mean()
        ema_long = df['close'].ewm(span=long, adjust=False).mean()

        if ema_short.iloc[-1] > ema_long.iloc[-1]:
            return "Buy"
        elif ema_short.iloc[-1] < ema_long.iloc[-1]:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"[⚠️] Error in calculate_ema_signal: {e}")
        return "Hold"
