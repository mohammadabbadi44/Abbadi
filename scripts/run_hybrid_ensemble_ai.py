# -*- coding: utf-8 -*-
"""
scripts/run_hybrid_ensemble_ai.py

باكتيست Ensemble يدمج عدة استراتيجيات فنية (Breakout / Trend / Momentum /
Mean Reversion / Volatility / Volume) مع طبقة ذكاء اصطناعي لإعادة ترتيب/فلترة الصفقات.
- يقرأ من: config/ensemble_ai.yaml
- يسحب الداتا من MT5 (مع اكتشاف رمز الذهب تلقائيًا)
- يحسب مؤشرات كثيرة ويولّد إشارات لكل استراتيجية (+ درجة ثقة)
- يدمج الإشارات بالأوزان (weighted_mean أو majority_vote) + بوابة AI
- إدارة صفقة: BE + Trailing + احتفاظ حتى max_hold_bars
- يحفظ الصفقات والملخص في reports/

تشغيل:
    python -m scripts.run_hybrid_ensemble_ai
"""

import os, sys, math, json, glob, warnings
from pathlib import Path
from datetime import datetime, timedelta, time as dtime
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ========= Dependencies =========
try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ MetaTrader5 غير مثبت. ثبّت: pip install MetaTrader5")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("❌ PyYAML غير مثبت. ثبّت: pip install pyyaml")
    sys.exit(1)

try:
    import joblib
except ImportError:
    joblib = None

# تدريب سريع اختياري لو مافي موديل
try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "ensemble_ai.yaml"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-9

# ========= Helpers =========
def tf_to_mt5(tf):
    t = str(tf).upper()
    m = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    if t in m: return m[t]
    if t.endswith("M"): return m.get("M"+t[:-1], mt5.TIMEFRAME_M15)
    if t.endswith("H"): return m.get("H"+t[:-1], mt5.TIMEFRAME_M15)
    return mt5.TIMEFRAME_M15

def parse_hhmm(s: str) -> dtime:
    hh, mm = map(int, str(s).split(":"))
    return dtime(hh, mm)

def in_session(ts: pd.Timestamp, start: dtime, end: dtime) -> bool:
    t = ts.time()
    if start <= end:
        return (t >= start) and (t <= end)
    else:
        return (t >= start) or (t <= end)  # نافذة عابرة منتصف الليل

def load_cfg(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Defaults المعقولة
    cfg.setdefault("symbol", "XAUUSD")
    cfg.setdefault("timeframe", "M5")
    cfg.setdefault("days", 365)

    cfg.setdefault("spread_points", 25)
    cfg.setdefault("slippage_points", 40)
    cfg.setdefault("commission_per_lot_usd", 7)

    cfg.setdefault("sl_atr_mult", 3.2)
    cfg.setdefault("tp_atr_mult", 4.2)
    cfg.setdefault("breakeven_at_rr", 1.0)
    cfg.setdefault("use_trailing", True)
    cfg.setdefault("trail_atr_mult", 1.0)
    cfg.setdefault("max_hold_bars", 16)

    cfg.setdefault("use_ema200_trend_filter", True)
    cfg.setdefault("min_ensemble_votes", 2)
    cfg.setdefault("min_ensemble_conf", 0.6)
    cfg.setdefault("cooldown_bars", 4)
    cfg.setdefault("one_position", True)

    cfg.setdefault("trading_hours", {"enabled": False, "start": "07:00", "end": "21:00"})
    cfg.setdefault("guards", {
        "min_atr_pct": 0.0005,
        "max_atr_pct": 0.04,
        "max_spread_vs_atr": 0.30,
        "max_trades_per_day": 120
    })

    cfg.setdefault("risk", {
        "risk_per_trade_pct": 0.25,
        "max_daily_loss_pct": 2.5,
        "max_total_drawdown_pct": 35.0
    })

    cfg.setdefault("ai", {
        "enabled": True,
        "model_path": "models/ensemble_lr.joblib",
        "decision_threshold": 0.62,
        "train_if_missing": True
    })

    cfg.setdefault("strategies", {
        "breakout_donchian": {"enabled": True, "weight": 1.0},
        "trend_following":   {"enabled": True, "weight": 1.3},
        "momentum_macd":     {"enabled": True, "weight": 1.1},
        "mean_reversion":    {"enabled": True, "weight": 0.7},
        "volatility_ttm":    {"enabled": True, "weight": 0.9},
        "volume_obv":        {"enabled": True, "weight": 0.8},
    })

    cfg.setdefault("ensemble_mode", "weighted_mean")  # or "majority_vote"
    return cfg

def ensure_symbol_visible(symbol):
    info = mt5.symbol_info(symbol)
    if info and info.visible:
        return info
    mt5.symbol_select(symbol, True)
    return mt5.symbol_info(symbol)

def bars_per_day_for_tf(tf: int) -> int:
    if tf == mt5.TIMEFRAME_M1:  return 1440
    if tf == mt5.TIMEFRAME_M5:  return 288
    if tf == mt5.TIMEFRAME_M15: return 96
    if tf == mt5.TIMEFRAME_M30: return 48
    if tf == mt5.TIMEFRAME_H1:  return 24
    if tf == mt5.TIMEFRAME_H4:  return 6
    if tf == mt5.TIMEFRAME_D1:  return 1
    return 288

def find_gold_symbol(preferred: Optional[list] = None) -> Optional[str]:
    order = (preferred or []) + [
        "XAUUSD","XAUUSDm","XAUUSD.r","XAUUSDx","GOLD","GOLDm","_XAUUSD",
        "XAUUSDecn","XAUUSD.a","XAUUSD.pro"
    ]
    for name in order:
        si = mt5.symbol_info(name)
        if si and (si.visible or mt5.symbol_select(name, True)):
            return name
    patterns = ["XAU*", "*XAUUSD*", "GOLD*", "*GOLD*"]
    seen = set()
    for pat in patterns:
        symbols = mt5.symbols_get(pat) or []
        for s in symbols:
            if s.name in seen: continue
            seen.add(s.name)
            if s.visible or mt5.symbol_select(s.name, True):
                return s.name
    return None

def fetch_data(symbol, timeframe, days) -> Tuple[pd.DataFrame, object]:
    tf = tf_to_mt5(timeframe)

    si = mt5.symbol_info(symbol)
    if not (si and (si.visible or mt5.symbol_select(symbol, True))):
        alt = find_gold_symbol(preferred=[symbol])
        if alt and alt != symbol:
            print(f"ℹ️ Using detected gold symbol instead of '{symbol}': {alt}")
            symbol = alt
            si = mt5.symbol_info(symbol)
    if not si:
        raise RuntimeError(f"No visible symbol for {symbol}. فعّل الرمز من Market Watch.")

    end = datetime.now()
    start = end - timedelta(days=int(days)+5)
    rates = mt5.copy_rates_range(symbol, tf, start, end)
    if rates is None or len(rates) == 0:
        bpd = bars_per_day_for_tf(tf)
        count = int((int(days) * bpd) * 1.2)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, max(count, 5000))
    if rates is None or len(rates) == 0:
        for d in [180,120,60,30,14,7]:
            start2 = end - timedelta(days=d+3)
            rates = mt5.copy_rates_range(symbol, tf, start2, end)
            if rates is not None and len(rates)>0:
                print(f"ℹ️ Fallback data window applied: {d} days")
                break
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data for {symbol} {timeframe}. افتح الشارت وحمّل التاريخ.")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time","open","high","low","close","tick_volume"]].dropna().reset_index(drop=True)
    for col in ["open","high","low","close"]:
        df[col] = df[col].astype(float)
    return df, si

# ========= Indicators =========
def ema(a, n): return pd.Series(a, dtype='float64').ewm(span=n, adjust=False).mean().values
def sma(a, n): return pd.Series(a, dtype='float64').rolling(n, min_periods=n).mean().values

def atr(df, n=14):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr, dtype='float64').rolling(n, min_periods=n).mean().values

def rsi(close, n=14):
    s = pd.Series(close, dtype='float64')
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ma_up = up.ewm(com=n-1, adjust=False).mean()
    ma_dn = dn.ewm(com=n-1, adjust=False).mean().replace(0, np.nan)
    rs = ma_up / ma_dn
    out = 100 - (100/(1+rs))
    return out.fillna(50).values

def macd(close, fast=12, slow=26, sig=9):
    ema_f = pd.Series(close).ewm(span=fast, adjust=False).mean()
    ema_s = pd.Series(close).ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    signal = macd_line.ewm(span=sig, adjust=False).mean()
    hist = macd_line - signal
    return macd_line.values, signal.values, hist.values

def cci(df, n=20, c=0.015):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(n, min_periods=n).mean()
    md = (tp - ma).abs().rolling(n, min_periods=n).mean()
    cci = (tp - ma) / (c * md.replace(0,np.nan))
    return cci.fillna(0).values

def stoch_k(df, k=14, d=3):
    low_min = df["low"].rolling(k, min_periods=k).min()
    high_max= df["high"].rolling(k, min_periods=k).max()
    k_raw = (df["close"] - low_min) / (high_max - low_min).replace(0,np.nan) * 100.0
    k_line = k_raw.rolling(d, min_periods=d).mean()
    return k_line.fillna(50).values

def bollinger(close, n=20, mult=2.0):
    s = pd.Series(close, dtype='float64')
    m = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std()
    upper = (m + mult*sd).values
    mid = m.values
    lower = (m - mult*sd).values
    return upper, mid, lower

def donchian(df, n=20):
    up = df["high"].rolling(n, min_periods=n).max().values
    dn = df["low"].rolling(n, min_periods=n).min().values
    return up, dn

def obv(df):
    direction = np.sign(df["close"].diff().fillna(0.0).values)
    vol = df["tick_volume"].values.astype(float)
    return np.cumsum(direction * vol)

# (اختياري) سوبرترند بسيط — نستخدمه ضمنيًا داخل Trend
def supertrend_like(df, atr_mult=3, atr_period=10):
    _atr = atr(df, atr_period)
    hl2 = (df["high"].values + df["low"].values) / 2.0
    upper = hl2 + atr_mult * _atr
    lower = hl2 - atr_mult * _atr
    return upper, lower

# ========= Feature pack =========
def build_features(df) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    feats["ema200"] = ema(df["close"].values, 200)
    feats["ema50"]  = ema(df["close"].values, 50)
    feats["ema20"]  = ema(df["close"].values, 20)

    feats["atr14"]  = atr(df, 14)
    feats["rsi14"]  = rsi(df["close"].values, 14)

    m_line, m_sig, m_hist = macd(df["close"].values)
    feats["macd_line"] = m_line
    feats["macd_sig"]  = m_sig
    feats["macd_hist"] = m_hist

    feats["cci20"]  = cci(df, 20)
    feats["stochK"] = stoch_k(df, 14, 3)

    bb_u, bb_m, bb_l = bollinger(df["close"].values, 20, 2.0)
    feats["bb_upper"], feats["bb_mid"], feats["bb_lower"] = bb_u, bb_m, bb_l

    don_u, don_l = donchian(df, 20)
    feats["don_up"], feats["don_dn"] = don_u, don_l

    feats["obv"] = obv(df)

    # نطاق 48 بار (لتصفية الرينج الخانق)
    range48 = df["high"].rolling(48, min_periods=48).max() - df["low"].rolling(48, min_periods=48).min()
    feats["rng_ratio48"] = (range48 / df["close"]).fillna(0.0).values

    # سوبرترند-like
    st_u, st_l = supertrend_like(df, 3, 10)
    feats["st_upper"], feats["st_lower"] = st_u, st_l

    return feats

# ========= Strategies (signal ∈ {-1,0,+1}, conf ≥ 0) =========
def s_breakout_donchian(df, f):
    sig = np.zeros(len(df))
    close = df["close"].values
    buy  = close > f["don_up"]
    sell = close < f["don_dn"]
    sig[buy]  = +1
    sig[sell] = -1
    # ثقة: زخم MACD + توسّع نطاق بولنجر
    width = (f["bb_upper"] - f["bb_lower"]) / (close + EPS)
    conf = np.clip((np.abs(f["macd_hist"]) / (np.nanstd(f["macd_hist"]) + EPS)) + width/np.nanmean(width), 0, 5)
    return sig, conf

def s_trend_following(df, f):
    align_buy  = (f["ema20"] > f["ema50"]) & (f["ema50"] > f["ema200"])
    align_sell = (f["ema20"] < f["ema50"]) & (f["ema50"] < f["ema200"])
    # سوبرترند: السعر فوق الحد السفلي = صاعد، والعكس هابط
    close = df["close"].values
    st_up, st_lo = f["st_upper"], f["st_lower"]
    st_up = np.nan_to_num(st_up, nan=np.inf)
    st_lo = np.nan_to_num(st_lo, nan=-np.inf)
    st_buy  = close > st_up  # ملاحظة: هذا تعريف مبسّط
    st_sell = close < st_lo

    sig = np.zeros(len(df))
    sig[align_buy & (f["rsi14"] > 55)]  = +1
    sig[align_sell & (f["rsi14"] < 45)] = -1
    # زوّد التأكيد بسوبرترند
    sig[st_buy]  = np.where(sig[st_buy]==0, +1, sig[st_buy])
    sig[st_sell] = np.where(sig[st_sell]==0, -1, sig[st_sell])

    conf = np.clip((np.abs(f["ema20"] - f["ema50"]) / (f["atr14"] + EPS)), 0, 6)
    return sig, conf

def s_momentum_macd(df, f):
    macd_cross_up = (f["macd_line"] > f["macd_sig"])
    macd_cross_dn = (f["macd_line"] < f["macd_sig"])
    sig = np.zeros(len(df))
    sig[macd_cross_up & (f["rsi14"] > 60)] = +1
    sig[macd_cross_dn & (f["rsi14"] < 40)] = -1
    conf = np.clip(np.abs(f["macd_hist"]) / (np.nanstd(f["macd_hist"]) + EPS), 0, 5)
    return sig, conf

def s_mean_reversion(df, f):
    close = df["close"].values
    sig = np.zeros(len(df))
    long  = (close < f["bb_lower"]) & (f["cci20"] < -100)
    short = (close > f["bb_upper"]) & (f["cci20"] > 100)
    sig[long]  = +1
    sig[short] = -1
    dist_mid = np.abs(close - f["bb_mid"]) / (f["atr14"] + EPS)
    conf = np.clip(dist_mid, 0, 6)
    return sig, conf

def s_volatility_ttm(df, f):
    close = df["close"].values
    width = (f["bb_upper"] - f["bb_lower"]) / (close + EPS)
    narrow = width < np.nanpercentile(width, 30)
    expand = width > np.nanpercentile(width, 60)
    sig = np.zeros(len(df))
    uptrend = f["ema20"] > f["ema50"]
    downtrend= f["ema20"] < f["ema50"]
    sig[narrow & expand & uptrend]  = +1
    sig[narrow & expand & downtrend] = -1
    conf = np.clip(width / (np.nanmean(width) + EPS), 0, 5)
    return sig, conf

def s_volume_obv(df, f):
    obv_arr = f["obv"].values if isinstance(f["obv"], pd.Series) else f["obv"]
    slope = np.r_[np.zeros(10), obv_arr[10:] - obv_arr[:-10]]
    sig = np.zeros(len(df))
    sig[slope > 0] = +1
    sig[slope < 0] = -1
    conf = np.clip(np.abs(slope) / (np.nanstd(slope) + EPS), 0, 5)
    return sig, conf

STRATS = {
    "breakout_donchian": s_breakout_donchian,
    "trend_following":   s_trend_following,
    "momentum_macd":     s_momentum_macd,
    "mean_reversion":    s_mean_reversion,
    "volatility_ttm":    s_volatility_ttm,
    "volume_obv":        s_volume_obv,
}

# ========= AI =========
def build_ai_matrix(df, f, sig_map: Dict[str, np.ndarray], conf_map: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """نبني ميزات للـ AI من المؤشرات + إشارات الاستراتيجيات."""
    feats = []
    close = df["close"].values
    base_cols = [
        f["rsi14"], f["atr14"], f["ema20"], f["ema50"], f["ema200"],
        f["macd_line"], f["macd_sig"], f["macd_hist"],
        f["cci20"], f["stochK"],
        f["bb_upper"], f["bb_mid"], f["bb_lower"],
        f["don_up"], f["don_dn"],
        f["rng_ratio48"],
    ]
    base_arr = np.vstack([np.nan_to_num(col, nan=0.0) for col in base_cols]).T

    # إشارات/ثقة الاستراتيجيات
    strat_names = list(STRATS.keys())
    sig_arr = np.vstack([np.nan_to_num(sig_map[n], nan=0.0) for n in strat_names]).T
    conf_arr= np.vstack([np.nan_to_num(conf_map[n], nan=0.0) for n in strat_names]).T

    feats = np.hstack([base_arr, sig_arr, conf_arr])

    # Label (للـ تدريب الاختياري): عائد الشمعة القادمة مقابل ATR → نجاح/فشل
    # لو السعر حقّق تحرك ≥ 0.5*ATR بالاتجاه الصحيح خلال الشمعة القادمة → 1 وإلا 0
    atr = np.nan_to_num(f["atr14"], nan=0.0)
    high_next = np.r_[df["high"].values[1:], df["high"].values[-1]]
    low_next  = np.r_[df["low"].values[1:],  df["low"].values[-1]]
    up_move   = (high_next - close) / (atr + EPS)
    dn_move   = (close - low_next) / (atr + EPS)
    # نجاح لو إما up_move أو dn_move ≥ 0.5
    y = ((up_move >= 0.5) | (dn_move >= 0.5)).astype(int)
    return feats, y

def load_or_train_ai(X: np.ndarray, y: np.ndarray, ai_cfg: dict):
    model = None
    if not ai_cfg.get("enabled", False):
        return None
    mp = ai_cfg.get("model_path")
    if mp and joblib:
        p = Path(mp)
        if p.exists():
            try:
                model = joblib.load(p)
                return model
            except Exception:
                model = None
    if ai_cfg.get("train_if_missing", False) and SKLEARN_OK:
        # تدريب سريع لوجستيك
        clf = LogisticRegression(max_iter=200)
        clf.fit(X, y)
        model = clf
        if mp and joblib:
            try:
                Path(mp).parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(clf, mp)
            except Exception:
                pass
    return model

def ai_prob(model, xrow: np.ndarray) -> float:
    try:
        p = model.predict_proba(xrow.reshape(1,-1))[0,1]
        return float(p)
    except Exception:
        return 0.5

# ========= Trade/Portfolio helpers =========
def compute_value_per_point(si) -> float:
    point = si.point or 0.01
    tick_size = getattr(si, "trade_tick_size", None) or point
    tick_value = getattr(si, "trade_tick_value", None) or getattr(si, "tick_value", None) or 1.0
    value_per_point = float(tick_value) * float(point / max(tick_size, EPS))
    return float(np.clip(value_per_point, 0.5, 10.0))

def safe_profit_factor(pnl: pd.Series) -> float:
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses <= EPS:
        return float("inf") if gains > EPS else 0.0
    return float(gains / max(losses, EPS))

def max_drawdown_percent(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    roll_max = equity_series.cummax()
    dd = (roll_max - equity_series) / roll_max.replace(0, np.nan)
    dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(dd.max() * 100.0)

# ========= Ensemble logic =========
def ensemble_decision(cfg: dict, strat_out: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """يرجع (signal, conf) نهائيين للـ ensemble قبل AI gate."""
    mode = cfg.get("ensemble_mode", "weighted_mean")
    weights = {k: v.get("weight", 1.0) for k, v in cfg["strategies"].items() if v.get("enabled", True)}

    # جمع لكل استراتيجية
    names = [k for k in STRATS.keys() if cfg["strategies"].get(k, {}).get("enabled", True)]
    sigs = [np.nan_to_num(strat_out[n][0], nan=0.0) for n in names]
    confs= [np.nan_to_num(strat_out[n][1], nan=0.0) for n in names]
    ws   = np.array([weights[n] for n in names], dtype=float)

    sigs_mat = np.vstack(sigs).T  # shape (N, S)
    conf_mat = np.vstack(confs).T

    # عدد الأصوات المؤيدة للاتجاه
    votes_buy  = np.sum(sigs_mat > 0, axis=1)
    votes_sell = np.sum(sigs_mat < 0, axis=1)

    if mode == "majority_vote":
        final_sig = np.sign(votes_buy - votes_sell).astype(float)
        # ثقة = متوسط conf للمصوّتين لنفس الاتجاه
        conf_buy  = np.where(votes_buy>0, np.sum(conf_mat*(sigs_mat>0), axis=1)/np.maximum(votes_buy,1), 0)
        conf_sell = np.where(votes_sell>0, np.sum(conf_mat*(sigs_mat<0), axis=1)/np.maximum(votes_sell,1), 0)
        final_conf = np.where(final_sig>0, conf_buy, np.where(final_sig<0, conf_sell, 0))
    else:
        # weighted_mean: إشارة نهائية = sign(sum(w * sig * conf_norm))
        conf_norm = conf_mat / (np.nanmean(conf_mat, axis=0, keepdims=True) + EPS)
        weighted = (sigs_mat * conf_norm) * ws.reshape(1,-1)
        score = np.sum(weighted, axis=1)
        final_sig = np.sign(score)
        # ثقة نهائية: متوسط conf للّذين اتفقوا مع الإشارة
        mask_agree = (np.sign(sigs_mat) == final_sig.reshape(-1,1)).astype(float)
        conf_agree = np.sum(conf_mat * mask_agree, axis=1) / np.maximum(np.sum(mask_agree, axis=1), 1)
        final_conf = np.nan_to_num(conf_agree, nan=0.0)

    # حفظ عدد الأصوات للمساعدة لاحقًا
    return final_sig, final_conf, votes_buy, votes_sell

# ========= Backtest =========
def backtest(df: pd.DataFrame, f: pd.DataFrame, cfg: dict, si) -> Tuple[dict, Optional[pd.DataFrame]]:
    # استراتيجيات → إشارات وثقة
    strat_out = {}
    for name, fn in STRATS.items():
        if cfg["strategies"].get(name, {}).get("enabled", True):
            s, c = fn(df, f)
        else:
            s = np.zeros(len(df)); c = np.zeros(len(df))
        strat_out[name] = (s, c)

    # Ensemble
    ens_sig, ens_conf, votes_buy, votes_sell = ensemble_decision(cfg, strat_out)

    # AI features/matrix
    X, y = build_ai_matrix(df, f, {k:v[0] for k,v in strat_out.items()}, {k:v[1] for k,v in strat_out.items()})
    model = load_or_train_ai(X, y, cfg.get("ai", {}))
    ai_th = float(cfg["ai"].get("decision_threshold", 0.62)) if cfg.get("ai", {}).get("enabled", False) else 0.0

    # إعدادات
    pt_mult   = float(cfg["tp_atr_mult"])
    sl_mult   = float(cfg["sl_atr_mult"])
    be_at_rr  = float(cfg["breakeven_at_rr"])
    use_trail = bool(cfg["use_trailing"])
    trail_mult= float(cfg["trail_atr_mult"])
    max_hold  = int(cfg.get("max_hold_bars", 12))
    cooldown  = int(cfg.get("cooldown_bars", 3))
    one_pos   = bool(cfg.get("one_position", True))
    use_ema200= bool(cfg.get("use_ema200_trend_filter", True))

    th = cfg.get("trading_hours", {})
    th_enabled = bool(th.get("enabled", False))
    th_start = parse_hhmm(th.get("start", "00:00"))
    th_end   = parse_hhmm(th.get("end", "23:59"))

    g = cfg.get("guards", {})
    min_atr_pct = float(g.get("min_atr_pct", 0.0005))
    max_atr_pct = float(g.get("max_atr_pct", 0.04))
    max_spread_vs_atr = float(g.get("max_spread_vs_atr", 0.30))
    max_trades_per_day = int(g.get("max_trades_per_day", 120))
    min_votes = int(cfg.get("min_ensemble_votes", 2))
    min_conf  = float(cfg.get("min_ensemble_conf", 0.6))

    risk_cfg = cfg.get("risk", {})
    risk_pct = float(risk_cfg.get("risk_per_trade_pct", 0.25))/100.0
    max_daily_loss_pct = float(risk_cfg.get("max_daily_loss_pct", 2.5))/100.0
    max_total_dd_pct   = float(risk_cfg.get("max_total_drawdown_pct", 35.0))/100.0

    point = si.point or 0.01
    spread = float(cfg["spread_points"]) * point
    slip   = float(cfg["slippage_points"]) * point
    com_per_lot = float(cfg["commission_per_lot_usd"])

    vol_min  = getattr(si, "volume_min", 0.01) or 0.01
    vol_step = getattr(si, "volume_step", 0.01) or 0.01
    vol_max  = getattr(si, "volume_max", 100.0) or 100.0
    vpp = compute_value_per_point(si)

    equity0 = 10000.0
    equity  = equity0
    peak_eq = equity0

    df["date"] = df["time"].dt.date
    current_day = None
    day_start_equity = equity
    daily_trades = 0

    last_trade_i = -999
    have_position = False

    rows = []
    eq_path = [equity]

    reasons = {
        "cooldown":0,"one_pos":0,"session":0,"atr_pct":0,"spread_vs_atr":0,
        "votes":0,"conf":0,"ema200":0,"ai_gate":0,"lot_too_small":0,
        "daily_cap":0,"daily_loss":0,"total_dd_stop":0
    }

    for i in range(250, len(df)-1):
        # DD إجمالي
        total_dd = (peak_eq - equity) / max(peak_eq, EPS)
        if total_dd >= max_total_dd_pct:
            reasons["total_dd_stop"] += 1
            break

        # يوم جديد
        d = df["date"].iloc[i]
        if d != current_day:
            current_day = d
            day_start_equity = equity
            daily_trades = 0

        # خسارة يومية
        if (day_start_equity - equity)/max(day_start_equity, EPS) >= max_daily_loss_pct:
            reasons["daily_loss"] += 1
            continue

        # جلسات
        if th_enabled and not in_session(df["time"].iloc[i], th_start, th_end):
            reasons["session"] += 1
            continue

        # تبريد وموقف واحد
        if (i - last_trade_i) < cooldown:
            reasons["cooldown"] += 1; continue
        if one_pos and have_position:
            reasons["one_pos"] += 1; continue
        if daily_trades >= max_trades_per_day:
            reasons["daily_cap"] += 1; continue

        # اتجاه ensemble
        sig = ens_sig[i]
        conf = ens_conf[i]
        votes_b, votes_s = int(votes_buy[i]), int(votes_sell[i])
        votes = max(votes_b, votes_s)

        if sig == 0:
            reasons["votes"] += 1; continue
        if votes < min_votes:
            reasons["votes"] += 1; continue
        if conf < min_conf:
            reasons["conf"] += 1; continue

        # فلتر EMA200
        if use_ema200:
            if sig > 0 and not (df["close"][i] > f["ema200"][i]):
                reasons["ema200"] += 1; continue
            if sig < 0 and not (df["close"][i] < f["ema200"][i]):
                reasons["ema200"] += 1; continue

        # ATR% على السعر
        atr_val = f["atr14"][i]
        if atr_val <= 0:
            reasons["atr_pct"] += 1; continue
        atr_pct = atr_val / max(df["close"][i], EPS)
        if atr_pct < min_atr_pct or atr_pct > max_atr_pct:
            reasons["atr_pct"] += 1; continue

        # تكلفة مقابل ATR
        if (spread + slip) > max_spread_vs_atr * atr_val:
            reasons["spread_vs_atr"] += 1; continue

        # AI Gate
        if model is not None:
            xrow = X[i, :]
            p = ai_prob(model, xrow)
            if p < ai_th:
                reasons["ai_gate"] += 1; continue
        else:
            p = 0.5

        # حجم اللوت (مخاطرة إلى SL)
        atr_points = atr_val / point
        sl_points  = max(1.0, sl_mult * atr_points)
        risk_usd   = equity * risk_pct
        lots_raw   = risk_usd / max(sl_points * vpp, EPS)
        lots_steps = math.floor(lots_raw / vol_step)
        lots       = max(vol_min, min(vol_max, lots_steps * vol_step))
        if lots < vol_min + EPS:
            reasons["lot_too_small"] += 1; continue

        entry = df["close"][i]
        if sig > 0:  # BUY
            price_in = entry + spread + slip
            tp_price = entry + pt_mult * atr_val
            sl_price = entry - sl_mult * atr_val
        else:        # SELL
            price_in = entry - spread - slip
            tp_price = entry - pt_mult * atr_val
            sl_price = entry + sl_mult * atr_val

        # إدارة الصفقة لعدة شمعات
        moved_to_be = False
        trail = trail_mult * atr_val if use_trail else None
        be_sl = price_in
        price_out = None
        win = 0

        last_bar_index = min(i + max_hold, len(df) - 1)
        for k in range(i+1, last_bar_index+1):
            bar = df.iloc[k]
            bh, bl, bc = float(bar["high"]), float(bar["low"]), float(bar["close"])

            # نقل BE عند 1R
            if not moved_to_be:
                reached_1R = (bh >= (price_in + be_at_rr * (sl_mult * atr_val))) if sig>0 \
                             else (bl <= (price_in - be_at_rr * (sl_mult * atr_val)))
                if reached_1R:
                    moved_to_be = True

            hit_tp = (bh >= tp_price) if sig>0 else (bl <= tp_price)
            hit_sl_raw = (bl <= sl_price) if sig>0 else (bh >= sl_price)

            hit_trail = False
            dyn_sl = None
            if moved_to_be and use_trail and trail is not None:
                if sig>0:
                    dyn_sl = max(be_sl, bc - trail)
                    hit_trail = bl <= dyn_sl
                else:
                    dyn_sl = min(be_sl, bc + trail)
                    hit_trail = bh >= dyn_sl

            if hit_tp:
                price_out = tp_price; win = 1; break
            if hit_trail:
                price_out = dyn_sl
                win = 1 if (sig>0 and price_out>price_in) or (sig<0 and price_out<price_in) else 0
                break
            if hit_sl_raw:
                price_out = price_in if moved_to_be else sl_price
                win = 0
                break

            if k == last_bar_index:
                price_out = bc
                win = 1 if ((price_out - price_in)/point > 0 and sig>0) or ((price_in - price_out)/point > 0 and sig<0) else 0
                break

        pnl_points = (price_out - price_in)/point if sig>0 else (price_in - price_out)/point
        pnl_usd = pnl_points * vpp * lots - (com_per_lot * lots)

        equity = max(equity + pnl_usd, 0.0)
        peak_eq = max(peak_eq, equity)
        eq_path.append(equity)

        rows.append({
            "time": df["time"].iloc[min(last_bar_index, len(df)-1)],
            "date": str(df["date"].iloc[min(last_bar_index, len(df)-1)]),
            "side": "BUY" if sig>0 else "SELL",
            "votes": votes,
            "ens_conf": round(float(conf),3),
            "ai_prob": round(float(p),3) if model is not None else None,
            "lots": float(lots),
            "entry": round(price_in,3),
            "exit": round(price_out,3),
            "pnl_points": round(float(pnl_points),2),
            "pnl_usd": round(float(pnl_usd),2),
            "equity": round(float(equity),2),
            "win": int(win),
        })

        last_trade_i = i
        have_position = one_pos
        daily_trades += 1
        if one_pos:
            have_position = False

        if equity <= EPS:
            break

    res = pd.DataFrame(rows)
    if res.empty:
        print("⚠️ لا توجد صفقات ضمن الشروط.")
        print("🔎 Diagnostics:", {k:v for k,v in reasons.items() if v})
        return None, None

    # مقاييس
    wins = int(res["win"].sum())
    n = int(len(res))
    win_rate = 100.0 * wins / max(n, 1)
    pf = safe_profit_factor(res["pnl_usd"])
    equity_series = pd.Series(eq_path, dtype='float64')
    dd = max_drawdown_percent(equity_series)
    ret = (equity_series.iloc[-1] - 10000.0) / 10000.0 * 100.0

    out_csv = REPORTS_DIR / f"ensemble_ai_bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    res.to_csv(out_csv, index=False)

    stats = {
        "symbol": cfg["symbol"], "timeframe": cfg["timeframe"], "days": int(cfg["days"]),
        "trades": n, "win_rate": round(win_rate, 2),
        "profit_factor": (round(pf, 2) if pf != float("inf") else "Inf"),
        "max_dd": round(float(dd), 2),
        "return_pct": round(float(ret), 2),
        "file": str(out_csv)
    }
    return stats, res

# ========= main =========
def main():
    print(f"🔎 Loading config: {CONFIG_PATH}")
    cfg = load_cfg(CONFIG_PATH)
    show = {k: cfg[k] for k in ["symbol","timeframe","days","min_ensemble_votes","min_ensemble_conf","ensemble_mode"] if k in cfg}
    print(f"📦 Params: {json.dumps(show, ensure_ascii=False)}")

    if not mt5.initialize():
        print("❌ mt5.initialize failed:", mt5.last_error())
        sys.exit(1)

    try:
        sym = cfg["symbol"]
        si = ensure_symbol_visible(sym)
        if not si:
            alt = find_gold_symbol(preferred=[sym])
            if alt:
                print(f"ℹ️ Auto-selected gold symbol: {alt}")
                cfg["symbol"] = alt
                si = ensure_symbol_visible(alt)
        if not si:
            raise RuntimeError(f"Symbol not visible: {cfg['symbol']}")

        df, si = fetch_data(cfg["symbol"], cfg["timeframe"], cfg["days"])
        feats = build_features(df)
        stats, res = backtest(df, feats, cfg, si)
        if stats is None:
            sys.exit(0)
        print("\n✅ Backtest Done")
        print(f"Trades={stats['trades']} | Win%={stats['win_rate']:.2f} | PF={stats['profit_factor']} | MaxDD={stats['max_dd']:.2f}% | Return={stats['return_pct']:.2f}%")
        print(f"📄 Saved trades CSV: {stats['file']}")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
