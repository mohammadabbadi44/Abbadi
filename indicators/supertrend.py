# indicators/supertrend.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    SuperTrend indicator (vanilla implementation).
    Returns DataFrame with:
      - supertrend: line value
      - direction : 1 (uptrend) / -1 (downtrend)
    Required df columns: high, low, close
    """
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("supertrend() expects columns: high, low, close")

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    # hl2 & ATR (simple rolling mean of True Range)
    hl2 = (high + low) / 2.0
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(int(period)).mean()

    # basic bands
    basic_upperband = hl2 + multiplier * atr
    basic_lowerband = hl2 - multiplier * atr

    # final bands
    final_upperband = basic_upperband.copy()
    final_lowerband = basic_lowerband.copy()

    for i in range(1, len(df)):
        # Upper band
        if (basic_upperband.iloc[i] < final_upperband.iloc[i - 1]) or (
            close.iloc[i - 1] > final_upperband.iloc[i - 1]
        ):
            final_upperband.iloc[i] = basic_upperband.iloc[i]
        else:
            final_upperband.iloc[i] = final_upperband.iloc[i - 1]

        # Lower band
        if (basic_lowerband.iloc[i] > final_lowerband.iloc[i - 1]) or (
            close.iloc[i - 1] < final_lowerband.iloc[i - 1]
        ):
            final_lowerband.iloc[i] = basic_lowerband.iloc[i]
        else:
            final_lowerband.iloc[i] = final_lowerband.iloc[i - 1]

    # direction & line
    direction = pd.Series(index=df.index, dtype=int)
    st_line = pd.Series(index=df.index, dtype=float)

    for i in range(1, len(df)):
        if close.iloc[i] > final_upperband.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < final_lowerband.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1] if pd.notna(direction.iloc[i - 1]) else 1

        st_line.iloc[i] = final_lowerband.iloc[i] if direction.iloc[i] == 1 else final_upperband.iloc[i]

    out = pd.DataFrame(
        {
            "supertrend": st_line,
            "direction": direction.fillna(0).astype(int),
        },
        index=df.index,
    )
    return out


def calculate_supertrend_signals(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.Series:
    """
    Convenience wrapper that converts SuperTrend direction to Buy/Sell/Hold series.
    """
    st = supertrend(df, period=period, multiplier=multiplier)
    dirn = st["direction"]
    sig = pd.Series(
        np.where(dirn > 0, "Buy", np.where(dirn < 0, "Sell", "Hold")),
        index=df.index,
    )
    # warmup
    sig.iloc[: int(period) + 1] = "Hold"
    return sig.astype("string")


# Backward-compat alias (some strategies import the singular name)
def calculate_supertrend_signal(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    return calculate_supertrend_signals(df, period=period, multiplier=multiplier)
