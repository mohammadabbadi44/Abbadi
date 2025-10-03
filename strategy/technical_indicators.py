import pandas as pd
import talib

def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    يحسب المؤشرات الفنية التالية:
    - EMA 20 / 50 / 200
    - RSI 14
    - MACD
    - Bollinger Bands
    - Volume MA
    """
    df = df.copy()

    # EMA
    df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
    df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
    df['ema_200'] = talib.EMA(df['close'], timeperiod=200)

    # RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)

    # MACD
    macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower

    # Volume MA
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()

    return df
