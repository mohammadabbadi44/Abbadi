# indicators/di.py

import pandas as pd

def calculate_di_signal(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    DI Cross Signal (Directional Indicator)
    يعطي إشارة Buy لما +DI يقطع -DI من تحت، وSell لما -DI يقطع +DI من تحت.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    tr_smooth = tr.rolling(window=period).sum()
    plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr_smooth)
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr_smooth)

    signal = ["Hold"]
    for i in range(1, len(df)):
        if plus_di[i - 1] < minus_di[i - 1] and plus_di[i] > minus_di[i]:
            signal.append("Buy")
        elif minus_di[i - 1] < plus_di[i - 1] and minus_di[i] > plus_di[i]:
            signal.append("Sell")
        else:
            signal.append("Hold")

    return pd.Series(signal, index=df.index)
