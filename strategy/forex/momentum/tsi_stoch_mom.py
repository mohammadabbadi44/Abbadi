# strategy/forex/momentum/tsi_stoch_mom.py
import numpy as np
import pandas as pd

DEFAULTS = {
    # TSI
    "tsi_r": 25,
    "tsi_s": 13,
    "tsi_signal": 7,
    "tsi_gate": 3.0,      # مضبوط

    # Stochastic
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_buy": 53.0,    # مضبوط
    "stoch_sell": 47.0,   # مضبوط

    # Momentum (ROC)
    "mom_period": 8,      # مضبوط
    "mom_gate": 0.0,

    "min_bars": 100
}

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _tsi(close: pd.Series, r: int, s: int) -> pd.Series:
    m = close.diff()
    abs_m = m.abs()
    ema1_m = _ema(m, r)
    ema1_abs = _ema(abs_m, r)
    ema2_m = _ema(ema1_m, s)
    ema2_abs = _ema(ema1_abs, s)
    tsi = 100 * (ema2_m / ema2_abs.replace(0, np.nan))
    return tsi.fillna(0.0)

def _stochastic(df: pd.DataFrame, k: int, d: int):
    ll = df["low"].rolling(k, min_periods=k).min()
    hh = df["high"].rolling(k, min_periods=k).max()
    rng = (hh - ll).replace(0, np.nan)
    k_percent = 100 * (df["close"] - ll) / rng
    k_percent = k_percent.clip(0, 100).fillna(50.0)
    d_percent = k_percent.rolling(d, min_periods=d).mean().fillna(50.0)
    return k_percent, d_percent

def _roc(close: pd.Series, period: int) -> pd.Series:
    return ((close / close.shift(period)) - 1.0).fillna(0.0) * 100.0

def _tsi_signal(tsi: pd.Series, signal_span: int, gate: float) -> pd.Series:
    sig = _ema(tsi, signal_span)
    above = (tsi > sig + gate)
    below = (tsi < sig - gate)
    out = np.where(above, "Buy", np.where(below, "Sell", "Hold"))
    return pd.Series(out, index=tsi.index, dtype=object)

def _stoch_signal(kp: pd.Series, dp: pd.Series, buy_gate: float, sell_gate: float) -> pd.Series:
    cross_up = (kp > dp) & (kp.shift(1) <= dp.shift(1))
    cross_dn = (kp < dp) & (kp.shift(1) >= dp.shift(1))
    buy = cross_up & (kp > buy_gate)
    sell = cross_dn & (kp < sell_gate)
    out = np.where(buy, "Buy", np.where(sell, "Sell", "Hold"))
    return pd.Series(out, index=kp.index, dtype=object)

def _mom_signal(roc: pd.Series, gate: float) -> pd.Series:
    # نسخة واضحة بدون np.where المتداخل
    buy_mask = roc > +gate
    sell_mask = roc < -gate
    out = np.full(len(roc), "Hold", dtype=object)
    out[buy_mask] = "Buy"
    out[sell_mask] = "Sell"
    return pd.Series(out, index=roc.index, dtype=object)

def tsi_stoch_mom_signals(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series([], dtype=object)

    P = DEFAULTS.copy()
    if params:
        P.update({k: v for k, v in params.items() if k in P})

    if len(df) < P["min_bars"]:
        return pd.Series(["Hold"] * len(df), index=df.index, dtype=object)

    close = df["close"]
    tsi_val = _tsi(close, P["tsi_r"], P["tsi_s"])
    kp, dp = _stochastic(df, P["stoch_k"], P["stoch_d"])
    roc = _roc(close, P["mom_period"])

    sig_tsi   = _tsi_signal(tsi_val, P["tsi_signal"], float(P["tsi_gate"]))
    sig_stoch = _stoch_signal(kp, dp, float(P["stoch_buy"]), float(P["stoch_sell"]))
    sig_mom   = _mom_signal(roc, float(P["mom_gate"]))

    votes_buy  = (sig_tsi.eq("Buy").astype(int) +
                  sig_stoch.eq("Buy").astype(int) +
                  sig_mom.eq("Buy").astype(int))
    votes_sell = (sig_tsi.eq("Sell").astype(int) +
                  sig_stoch.eq("Sell").astype(int) +
                  sig_mom.eq("Sell").astype(int))

    out = np.where(votes_buy >= 2, "Buy",
          np.where(votes_sell >= 2, "Sell", "Hold"))
    return pd.Series(out, index=df.index, dtype=object)

def generate_signal_series(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    return tsi_stoch_mom_signals(df, params=params)

def get_signals(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    return tsi_stoch_mom_signals(df, params=params)
