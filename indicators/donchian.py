# indicators/donchian.py

import pandas as pd
import numpy as np
from collections import namedtuple

DonchianChannel = namedtuple('DonchianChannel', ['upperband', 'middleband', 'lowerband'])


def donchian(df: pd.DataFrame, period: int = 20) -> DonchianChannel:
    """
    Donchian Channels
    :param df: DataFrame يحتوي على ['high', 'low']
    :return: namedtuple يحتوي على upperband, middleband, lowerband
    """
    highs = df['high'].rolling(period).max()
    lows = df['low'].rolling(period).min()
    middle = (highs + lows) / 2
    return DonchianChannel(highs, middle, lows)


def calculate_donchian_signal(df: pd.DataFrame, period: int = 20) -> str:
    """
    يولّد إشارة واحدة فقط:
    - Buy إذا الإغلاق تحت lower band
    - Sell إذا الإغلاق فوق upper band
    - Hold إذا بينهما
    """
    if len(df) < period:
        return "Hold"

    dc = donchian(df, period)
    close = df['close'].iloc[-1]
    upper = dc.upperband.iloc[-2]
    lower = dc.lowerband.iloc[-2]

    if close > upper:
        return "Sell"
    elif close < lower:
        return "Buy"
    else:
        return "Hold"
