# strategy/load_forex_strategies.py
# -*- coding: utf-8 -*-
import importlib
import inspect
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

MODULES = [
    "strategy.forex.breakout.di_cross_frama_breakout",
    "strategy.forex.breakout.donchian_fisher_keltner",
    "strategy.forex.mean_reversion.cci_zscore_reflex",
    "strategy.forex.mean_reversion.typprice_midprice_dpo",
    "strategy.forex.momentum.macd_rsi_cmo",
    "strategy.forex.momentum.tsi_stoch_mom",
    "strategy.forex.scalping.srsi_hull_cmo",
    "strategy.forex.scalping.vwma_mom_roc",
    "strategy.forex.trend_following.classic_combo",
    "strategy.forex.trend_following.ema_supertrend_rsi",
    "strategy.forex.trend_following.ichimoku_adx_kama",
    "strategy.forex.volatility.bollinger_atr_chop",
    "strategy.forex.volatility.ttm_squeeze_squeeze_momentum",
    "strategy.forex.volume.volume_osc_marketfi",
    "strategy.forex.volume.vwap_obv_vpt",
]

# بروفايلات باراميترات "مخففة" لتحريك الاستراتيجيات اللي راجعة ALL Hold
LOOSE_PROFILES: Dict[str, list] = {
    "classic_combo": [
        {"rsi_buy": 52, "rsi_sell": 48, "ema_fast": 20, "ema_slow": 50},
        {"rsi_buy": 51, "rsi_sell": 49, "ema_fast": 10, "ema_slow": 30},
    ],
    "ema_supertrend_rsi": [
        {"supertrend_atr_mult": 2.0, "rsi_min": 50},
        {"supertrend_atr_mult": 1.5, "rsi_min": 48},
    ],
    "ichimoku_adx_kama": [
        {"adx_min": 18, "span_shift": 0},
        {"adx_min": 15, "span_shift": 0},
    ],
    "bollinger_atr_chop": [
        {"bb_mult": 1.6, "atr_mult": 1.2, "chop_max": 60},
        {"bb_mult": 1.4, "atr_mult": 1.1, "chop_max": 65},
    ],
    "vwap_obv_vpt": [
        {"obv_confirm": False, "vpt_thresh": 0.0},
        {"obv_confirm": False, "vpt_thresh": -0.0},
    ],
    "srsi_hull_cmo": [
        {"srsi_k": 5, "srsi_d": 3, "srsi_buy": 45, "srsi_sell": 55},
        {"srsi_k": 5, "srsi_d": 3, "srsi_buy": 50, "srsi_sell": 50},
    ],
    "vwma_mom_roc": [
        {"mom_len": 5, "roc_len": 5},
        {"mom_len": 4, "roc_len": 4},
    ],
    "ttm_squeeze_squeeze_momentum": [
        {"sqz_bb_mult": 1.5, "sqz_kc_mult": 1.3},
        {"sqz_bb_mult": 1.4, "sqz_kc_mult": 1.2},
    ],
    "di_cross_frama_breakout": [
        {"frama_len": 20, "di_thresh": 5, "confirm_window": 2},
        {"frama_len": 14, "di_thresh": 3, "confirm_window": 3},
    ],
}


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "time" not in out.columns:
        if out.index.dtype.kind in "mM":
            out["time"] = out.index
        else:
            for c in ("timestamp", "date", "datetime"):
                if c in out.columns:
                    out["time"] = pd.to_datetime(out[c], errors="coerce")
                    break
            if "time" not in out.columns:
                out["time"] = pd.NaT
    if "volume" not in out.columns:
        out["volume"] = 0.0
    return out


BUY_WORDS  = {"buy","long","bull","bullish","up","b","+","1","true","yes"}
SELL_WORDS = {"sell","short","bear","bearish","down","s","-","-1","false","no"}

def _to_label(x: Any) -> str:
    if isinstance(x, (np.generic,)):
        x = x.item()
    if isinstance(x, (list, tuple)) and len(x):
        x = x[0]
    if isinstance(x, dict):
        for k in ("signal","sig","direction","dir","value"):
            if k in x:
                x = x[k]; break
    if isinstance(x, (bool, np.bool_)):
        return "Buy" if x else "Hold"
    if isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x):
        return "Buy" if x > 0 else ("Sell" if x < 0 else "Hold")
    if isinstance(x, str):
        s = x.strip().lower()
        if s in BUY_WORDS or "buy" in s or "long" in s or "bull" in s:
            return "Buy"
        if s in SELL_WORDS or "sell" in s or "short" in s or "bear" in s:
            return "Sell"
        if s in {"hold","wait","neutral","none","flat","0",""}:
            return "Hold"
        if s.startswith(("b","+")):
            return "Buy"
        if s.startswith(("s","-")):
            return "Sell"
    return "Hold"

def _as_series(out: Any, n: int, index: pd.Index) -> pd.Series:
    if isinstance(out, pd.Series):
        s = out
    elif hasattr(out, "__len__") and not isinstance(out, (str, bytes)) and len(out) == n:
        s = pd.Series(list(out), index=index)
    else:
        s = pd.Series(["Hold"] * n, index=index)
    s = s.map(_to_label).astype("string").fillna("Hold")
    if len(s) != n:
        s = s.reindex(index).fillna("Hold").astype("string")
    return s

def _call_strategy(fn, df: pd.DataFrame, params: Optional[Dict[str, Any]] = None):
    try:
        return fn(df, params=params)
    except TypeError:
        return fn(df)

def load_all_forex_strategies(df: pd.DataFrame) -> Dict[str, pd.Series]:
    n, idx = len(df), df.index
    safe_df = _prep_df(df)
    out: Dict[str, pd.Series] = {}

    for path in MODULES:
        name = path.split(".")[-1]
        try:
            m = importlib.import_module(path)
            fn = getattr(m, "get_signals", None) or getattr(m, "generate_signal", None)
            if fn is None or not callable(fn):
                print(f"⚠️ No generate_signal() found in: {name}")
                out[name] = _as_series(["Hold"] * n, n, idx)
                continue

            # محاولة أولى
            try:
                sig = _call_strategy(fn, safe_df, params=None)
            except Exception as e1:
                try:
                    sig = fn(safe_df)
                except Exception as e2:
                    print(f"[loader:{name}] {type(e2).__name__}: {e2}")
                    out[name] = _as_series(["Hold"] * n, n, idx)
                    continue

            s = _as_series(sig, n, idx)

            # لو ALL Hold جرّب بروفايلات مخففة
            if s.eq("Hold").all():
                profiles = LOOSE_PROFILES.get(name, [])
                for prof in profiles:
                    try:
                        sig2 = _call_strategy(fn, safe_df, params=prof)
                        s2 = _as_series(sig2, n, idx)
                        if not s2.eq("Hold").all():
                            nh = 100.0 * (len(s2) - int((s2 == "Hold").sum())) / max(1, len(s2))
                            print(f"[{name}] applied loose params {prof} -> non_hold={nh:.2f}%")
                            s = s2
                            break
                    except Exception as e3:
                        print(f"[{name}] loose params failed: {type(e3).__name__}: {e3}")

            out[name] = s

        except Exception as e:
            print(f"[loader:{name}] {type(e).__name__}: {e}")
            out[name] = _as_series(["Hold"] * n, n, idx)

    return out
