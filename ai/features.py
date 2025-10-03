# ai/features.py
import numpy as np
import pandas as pd

def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(span=period, adjust=False).mean()
    roll_down = pd.Series(down, index=close.index).ewm(span=period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def _bbands(close: pd.Series, n: int = 20, k: float = 2.0):
    ma = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper = ma + k * std
    lower = ma - k * std
    width = (upper - lower) / (ma + 1e-9)
    pct = (close - lower) / ((upper - lower) + 1e-9)
    return width, pct

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    expects columns: time, open, high, low, close, volume
    returns features aligned with df index (drops NaNs)
    """
    df = df.copy().reset_index(drop=True)
    for col in ['open','high','low','close','volume']:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    close = df['close']

    # simple returns
    df['ret_1']  = close.pct_change()
    df['ret_3']  = close.pct_change(3)
    df['ret_5']  = close.pct_change(5)
    df['ret_10'] = close.pct_change(10)

    # EMA + slope
    for n in [20, 50, 200]:
        ema = _ema(close, n)
        df[f'ema_{n}'] = ema
        df[f'ema_{n}_slope'] = ema.diff()

    # RSI
    df['rsi_14'] = _rsi(close, 14)

    # Bollinger
    bw, bpct = _bbands(close, 20, 2.0)
    df['bb_width'] = bw
    df['bb_pct']   = bpct

    # ATR%
    df['atr_14'] = _atr(df, 14)
    df['atr_pct'] = df['atr_14'] / (close + 1e-9)

    # relative volume
    vma20 = df['volume'].rolling(20).mean()
    df['vol_rel'] = (df['volume'] / (vma20 + 1e-9))

    feats = df[['ret_1','ret_3','ret_5','ret_10',
                'ema_20','ema_50','ema_200',
                'ema_20_slope','ema_50_slope','ema_200_slope',
                'rsi_14','bb_width','bb_pct','atr_pct','vol_rel']].replace([np.inf,-np.inf], np.nan).dropna()

    feats.index = feats.index  # keep alignment
    return feats
