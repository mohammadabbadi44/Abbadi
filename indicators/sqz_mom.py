import pandas as pd
import numpy as np


def calculate_squeeze_momentum_signals(df: pd.DataFrame,
                                       bb_window: int = 20,
                                       kc_multiplier: float = 1.5) -> pd.Series:
    """
    يحسب إشارات Squeeze Momentum:
    - Buy: في حالة Squeeze مع Momentum إيجابي
    - Sell: في حالة Squeeze مع Momentum سلبي
    - Hold: باقي الحالات

    :param df: DataFrame يحتوي أعمدة 'close', 'high', 'low'
    :return: سلسلة إشارات (Buy/Sell/Hold)
    """

    close = df['close']
    high = df['high']
    low = df['low']

    # === Bollinger Bands ===
    ma = close.rolling(window=bb_window).mean()
    std = close.rolling(window=bb_window).std()
    upper_bb = ma + 2 * std
    lower_bb = ma - 2 * std

    # === Keltner Channel ===
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=bb_window).mean()
    upper_kc = ma + kc_multiplier * atr
    lower_kc = ma - kc_multiplier * atr

    # === Squeeze Condition ===
    squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)

    # === Momentum (بسيط: الفارق بين الإغلاق والمتوسط) ===
    momentum = close - ma

    signals = []
    for i in range(len(df)):
        if i < bb_window or pd.isna(momentum[i]) or pd.isna(squeeze_on[i]):
            signals.append("Hold")
        elif squeeze_on[i] and momentum[i] > 0:
            signals.append("Buy")
        elif squeeze_on[i] and momentum[i] < 0:
            signals.append("Sell")
        else:
            signals.append("Hold")

    return pd.Series(signals, index=df.index)
