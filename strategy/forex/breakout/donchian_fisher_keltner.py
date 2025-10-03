import numpy as np
import pandas as pd
from typing import Dict, Optional
import talib  # تأكد إن مكتبة TA-Lib منصبة عندك


# ======= Donchian Channel Signals =======
def _donchian_signals(df: pd.DataFrame, period: int = 20, breakout_eps_pct: float = 0.0) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    upper = high.rolling(period).max().shift(1)
    lower = low.rolling(period).min().shift(1)

    buy = close > upper * (1 + breakout_eps_pct)
    sell = close < lower * (1 - breakout_eps_pct)

    out = pd.Series(np.where(buy, "Buy", np.where(sell, "Sell", "Hold")), index=df.index)
    out.iloc[:period] = "Hold"
    return out


# ======= Fisher Transform Signals =======
def _fisher_transform_signals(
    df: pd.DataFrame,
    length: int = 9,
    require_zero_filter: bool = True,
    min_delta: float = 0.0
) -> pd.Series:
    hl2 = (df["high"] + df["low"]) / 2.0
    lowest = hl2.rolling(length).min()
    highest = hl2.rolling(length).max()
    rng = (highest - lowest).replace(0, np.nan)

    x = (hl2 - lowest) / rng * 2 - 1
    x = x.clip(-0.999, 0.999)

    fisher = pd.Series(0.0, index=df.index)
    for i in range(length, len(x)):
        prev = fisher.iloc[i - 1]
        val = 0.5 * np.log((1 + x.iloc[i]) / (1 - x.iloc[i]))
        fisher.iloc[i] = 0.5 * val + 0.5 * prev

    sig = fisher.shift(1)
    d = fisher - sig

    cross_up = (fisher > sig) & (d > min_delta)
    cross_dn = (fisher < sig) & (-d > min_delta)

    if require_zero_filter:
        cross_up &= (fisher > 0)
        cross_dn &= (fisher < 0)

    out = pd.Series(np.where(cross_up, "Buy", np.where(cross_dn, "Sell", "Hold")), index=df.index)
    out.iloc[:length + 2] = "Hold"
    return out


# ======= Keltner Channel Signals =======
def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _keltner_signals(
    df: pd.DataFrame,
    ma_period: int = 20,
    atr_period: int = 10,
    atr_mult: float = 2.0,
    breakout_eps_pct: float = 0.0
) -> pd.Series:
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    mid = _ema(hlc3, ma_period)
    atr = _true_range(df).rolling(atr_period).mean()
    upper = mid + atr_mult * atr
    lower = mid - atr_mult * atr

    close = df["close"]
    buy = close > upper * (1 + breakout_eps_pct)
    sell = close < lower * (1 - breakout_eps_pct)

    out = pd.Series(np.where(buy, "Buy", np.where(sell, "Sell", "Hold")), index=df.index)
    warmup = max(ma_period, atr_period) + 1
    out.iloc[:warmup] = "Hold"
    return out


# ======= Strategy Main Function =======
def get_signals(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    """
    Breakout strategy (Donchian + Fisher + Keltner) مع فلتر ترند ذكي.

    المعاملات في params:
      - donchian_period: int = 20
      - fisher_length: int = 9
      - fisher_require_zero_filter: bool = True
      - fisher_min_delta: float = 0.05
      - keltner_ma_period: int = 20
      - keltner_atr_period: int = 10
      - keltner_atr_mult: float = 1.6
      - breakout_eps_pct: float = 0.0005
      - confirm_window: int = 2
      - use_trend_filter: bool = True
      - trend_ema_period: int = 200
      - cooldown_bars: int = 1
    """
    if params is None:
        params = {}

    n = len(df)
    if n < 50:
        return pd.Series(["Hold"] * n, index=df.index)

    d_per = int(params.get("donchian_period", 20))
    f_len = int(params.get("fisher_length", 9))
    f_zero = bool(params.get("fisher_require_zero_filter", True))
    f_min_d = float(params.get("fisher_min_delta", 0.05))
    k_ma = int(params.get("keltner_ma_period", 20))
    k_atr = int(params.get("keltner_atr_period", 10))
    k_mult = float(params.get("keltner_atr_mult", 1.6))
    eps = float(params.get("breakout_eps_pct", 0.0005))
    win = int(params.get("confirm_window", 2))
    use_trend = bool(params.get("use_trend_filter", True))
    ema_p = int(params.get("trend_ema_period", 200))
    cooldown = int(params.get("cooldown_bars", 1))

    sig_don = _donchian_signals(df, period=d_per, breakout_eps_pct=eps)
    sig_fis = _fisher_transform_signals(df, length=f_len, require_zero_filter=f_zero, min_delta=f_min_d)
    sig_kel = _keltner_signals(df, ma_period=k_ma, atr_period=k_atr, atr_mult=k_mult, breakout_eps_pct=eps)

    # تصويت إشارات خلال النافذة confirm_window
    def _roll_has(series: pd.Series, label: str) -> pd.Series:
        return series.eq(label).rolling(window=win, min_periods=1).max().astype(bool)

    buy_votes = (
        _roll_has(sig_don, "Buy").astype(int) +
        _roll_has(sig_fis, "Buy").astype(int) +
        _roll_has(sig_kel, "Buy").astype(int)
    )
    sell_votes = (
        _roll_has(sig_don, "Sell").astype(int) +
        _roll_has(sig_fis, "Sell").astype(int) +
        _roll_has(sig_kel, "Sell").astype(int)
    )

    out = pd.Series("Hold", index=df.index)
    out = out.mask(buy_votes >= 2, "Buy")
    out = out.mask(sell_votes >= 2, "Sell")

    # فلتر الترند الذكي باستخدام EMA + RSI + ADX
    if use_trend:
        ema_trend = _ema(df["close"], ema_p)
        rsi = talib.RSI(df["close"], timeperiod=14)
        adx = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)

        bull = (df["close"] > ema_trend) & (rsi > 50) & (adx > 25)
        bear = (df["close"] < ema_trend) & (rsi < 50) & (adx > 25)

        out = pd.Series(np.where((out == "Sell") & bull, "Hold", out), index=df.index)
        out = pd.Series(np.where((out == "Buy") & bear, "Hold", out), index=df.index)

        out.iloc[:ema_p + 1] = "Hold"

    # فترة تسخين warmup
    warmup = max(d_per + 1, f_len + 2, k_ma + 1, k_atr + 1)
    out.iloc[:warmup] = "Hold"

    # كولداون cooldown لمنع التقلب السريع
    if cooldown > 0:
        final = out.copy()
        last_sig = "Hold"
        bars_since = cooldown + 1
        for i in range(n):
            sig = out.iat[i]
            if sig in ("Buy", "Sell"):
                if bars_since <= cooldown and sig != last_sig:
                    final.iat[i] = "Hold"
                else:
                    last_sig = sig
                    bars_since = 0
            bars_since += 1
        out = final

    return out.astype(str)
