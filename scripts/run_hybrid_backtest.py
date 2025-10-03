# -*- coding: utf-8 -*-
"""
scripts/run_hybrid_backtest.py

Backtest محترم ل XAUUSD (أو غيره) باستخدام MT5 + إعدادات YAML.
- يقرأ: config/hybrid_ai.yaml
- يسحب بيانات من MetaTrader5 (مرن مع فشل البيانات/الرمز)
- يطبق EMA200 + scoring بسيط + ATR SL/TP
- فلتر AI (اختياري عبر joblib)
- إدارة الصفقة: Break-Even عند 1R + Trailing بعده + احتفاظ حتى max_hold_bars
- حراس جودة: ATR% من السعر، سبريد مقابل ATR، جلسات تداول، حد صفقات يومي
- حدود مخاطر: خسارة يومية، سحب شامل
- تشخيص أسباب استبعاد الإشارات
- يحفظ النتائج: reports/hybrid_ai_bt_*.csv
"""

import os, sys, math, json
from pathlib import Path
from datetime import datetime, timedelta, time as dtime
from typing import Optional

import numpy as np
import pandas as pd

# ====== Dependencies ======
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

# ====== Paths ======
ROOT = Path(__file__).resolve().parents[1]  # project root (one level above scripts)
CONFIG_PATH = ROOT / "config" / "hybrid_ai.yaml"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-9  # anti-division-by-zero

# ====== Utils ======
def ema(a, n):
    return pd.Series(a, dtype='float64').ewm(span=n, adjust=False).mean().values

def atr(df, n=14):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr, dtype='float64').rolling(n, min_periods=n).mean().values

def rsi(close, n=14):
    s = pd.Series(close, dtype='float64')
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=n-1, adjust=False).mean()
    ma_down = down.ewm(com=n-1, adjust=False).mean().replace(0, np.nan)
    rs = ma_up / ma_down
    out = 100 - (100/(1+rs))
    return out.fillna(50).values

def tf_to_mt5(tf):
    t = str(tf).upper()
    m = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1
    }
    if t in m: return m[t]
    if t.endswith("M"): return m.get("M"+t[:-1], mt5.TIMEFRAME_M15)
    if t.endswith("H"): return m.get("H"+t[:-1], mt5.TIMEFRAME_M15)
    return mt5.TIMEFRAME_M15

def parse_hhmm(s: str) -> dtime:
    hh, mm = map(int, s.split(":"))
    return dtime(hh, mm)

def in_session(ts: pd.Timestamp, start: dtime, end: dtime) -> bool:
    t = ts.time()
    if start <= end:
        return (t >= start) and (t <= end)
    else:
        return (t >= start) or (t <= end)

def load_cfg(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # defaults (معقولة لفريم M5)
    cfg.setdefault("symbol", "XAUUSD")
    cfg.setdefault("timeframe", "M5")
    cfg.setdefault("days", 365)

    cfg.setdefault("spread_points", 25)
    cfg.setdefault("slippage_points", 40)
    cfg.setdefault("commission_per_lot_usd", 7)

    cfg.setdefault("use_ema200_trend_filter", True)
    cfg.setdefault("min_signals", 4)
    cfg.setdefault("rr_ratio", 1.9)
    cfg.setdefault("sl_atr_mult", 3.0)
    cfg.setdefault("tp_atr_mult", 3.3)

    cfg.setdefault("use_trailing", True)
    cfg.setdefault("trail_atr_mult", 0.9)
    cfg.setdefault("breakeven_at_rr", 1.0)

    cfg.setdefault("cooldown_bars", 3)
    cfg.setdefault("one_position", True)

    cfg.setdefault("ai", {"enabled": False})

    cfg.setdefault("risk", {
        "risk_per_trade_pct": 0.25,
        "max_daily_loss_pct": 3.0,
        "max_total_drawdown_pct": 45.0
    })

    cfg.setdefault("trading_hours", {"enabled": True, "start": "07:00", "end": "21:00"})
    cfg.setdefault("guards", {
        "min_atr_pct": 0.0004,
        "max_atr_pct": 0.035,
        "max_spread_vs_atr": 0.45,
        "max_trades_per_day": 200
    })

    # الاحتفاظ لعدة شمعات
    cfg.setdefault("max_hold_bars", 12)

    return cfg

def ensure_symbol_visible(symbol):
    info = mt5.symbol_info(symbol)
    if info and info.visible:
        return info
    mt5.symbol_select(symbol, True)
    return mt5.symbol_info(symbol)

# ---------- Resilient symbol/data utilities ----------
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

def fetch_data(symbol, timeframe, days):
    tf = tf_to_mt5(timeframe)

    si = mt5.symbol_info(symbol)
    if not (si and (si.visible or mt5.symbol_select(symbol, True))):
        alt = find_gold_symbol(preferred=[symbol])
        if alt and alt != symbol:
            print(f"ℹ️ Using detected gold symbol instead of '{symbol}': {alt}")
            symbol = alt
            si = mt5.symbol_info(symbol)
    if not si:
        raise RuntimeError(f"No visible symbol for {symbol}. افتح Market Watch → Symbols وفَعّل رمز الذهب.")

    end = datetime.now()
    start = end - timedelta(days=int(days)+5)
    rates = mt5.copy_rates_range(symbol, tf, start, end)

    if rates is None or len(rates) == 0:
        bpd = bars_per_day_for_tf(tf)
        count = int((int(days) * bpd) * 1.2)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, max(count, 5000))

    if rates is None or len(rates) == 0:
        for d in [180, 120, 60, 30, 14, 7]:
            start2 = end - timedelta(days=d+3)
            rates = mt5.copy_rates_range(symbol, tf, start2, end)
            if rates is not None and len(rates) > 0:
                print(f"ℹ️ Fallback data window applied: {d} days")
                break

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data for {symbol} {timeframe}. افتح MT5 واعمل Show All وتحميل التاريخ للرمز/الفريم.")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time","open","high","low","close","tick_volume"]].dropna().reset_index(drop=True)
    for col in ["open","high","low","close"]:
        df[col] = df[col].astype(float)
    return df

# ---------- Features/scoring ----------
def build_features(df):
    feats = pd.DataFrame(index=df.index)
    feats["ema200"] = ema(df["close"].values, 200)
    feats["ema50"]  = ema(df["close"].values, 50)
    feats["ema20"]  = ema(df["close"].values, 20)
    feats["atr14"]  = atr(df, 14)
    feats["rsi14"]  = rsi(df["close"].values, 14)

    # simple scoring
    score = np.zeros(len(df), dtype=float)
    score += (feats["ema20"] > feats["ema50"]).astype(int)
    score += (feats["rsi14"] > 55).astype(int)

    rng_w = 48
    lows = df["low"].rolling(rng_w, min_periods=rng_w).min()
    highs = df["high"].rolling(rng_w, min_periods=rng_w).max()
    rng = (highs - lows).replace(0, np.nan)
    pos = (df["close"] - lows) / rng
    score += (pos > 0.7).fillna(False).astype(int)
    score -= (pos < 0.3).fillna(False).astype(int)

    ema20 = feats["ema20"].values
    slope = np.r_[np.zeros(3), ema20[3:] - ema20[:-3]]
    score += (slope > 0).astype(int)

    feats["ema_align_buy"]  = (feats["ema50"] > feats["ema200"]).astype(int)
    feats["ema_align_sell"] = (feats["ema50"] < feats["ema200"]).astype(int)
    range48 = df["high"].rolling(48, min_periods=48).max() - df["low"].rolling(48, min_periods=48).min()
    feats["rng_ratio48"] = (range48 / df["close"]).fillna(0.0).values
    feats["rsi_gt60"] = (feats["rsi14"] > 60).astype(int)
    feats["rsi_lt40"] = (feats["rsi14"] < 40).astype(int)

    side = np.array([None]*len(df), dtype=object)
    above = df["close"].values > feats["ema200"].values
    below = df["close"].values < feats["ema200"].values
    side[(score >= 3) & above] = "BUY"
    side[(score >= 3) & below] = "SELL"

    feats["score"] = score
    feats["side"]  = side
    return feats

def load_ai(model_path):
    if not model_path or not joblib:
        return None
    p = Path(model_path)
    if p.exists():
        try:
            return joblib.load(p)
        except Exception:
            return None
    return None

def ai_confidence(row_dict, model):
    feats = ["rsi14","atr14","ema20","ema50","ema200","score"]
    x = np.array([[row_dict.get(k, 0.0) for k in feats]], dtype=float)
    try:
        p = model.predict_proba(x)[0, 1]
        return float(p)
    except Exception:
        return 0.5

def safe_profit_factor(pnl_series: pd.Series) -> float:
    gains = pnl_series[pnl_series > 0].sum()
    losses = -pnl_series[pnl_series < 0].sum()
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

def compute_value_per_point(si) -> float:
    point = si.point or 0.01
    tick_size = getattr(si, "trade_tick_size", None) or point
    tick_value = getattr(si, "trade_tick_value", None) or getattr(si, "tick_value", None) or 1.0
    value_per_point = float(tick_value) * float(point / max(tick_size, EPS))
    return float(np.clip(value_per_point, 0.5, 10.0))

# ====== Backtest ======
def backtest(df, feats, cfg, si):
    # config
    pt_mult     = float(cfg["tp_atr_mult"])
    sl_mult     = float(cfg["sl_atr_mult"])
    min_sig     = int(cfg["min_signals"])
    use_ema200  = bool(cfg["use_ema200_trend_filter"])
    cooldown    = int(cfg.get("cooldown_bars", 3))
    one_pos     = bool(cfg.get("one_position", True))

    risk_cfg    = cfg.get("risk", {})
    risk_pct    = float(risk_cfg.get("risk_per_trade_pct", 0.25)) / 100.0
    max_daily_loss_pct   = float(risk_cfg.get("max_daily_loss_pct", 3.0)) / 100.0
    max_total_dd_pct     = float(risk_cfg.get("max_total_drawdown_pct", 45.0)) / 100.0

    spread_pts  = float(cfg["spread_points"])
    slip_pts    = float(cfg["slippage_points"])
    com_per_lot = float(cfg["commission_per_lot_usd"])

    # sessions
    th = cfg.get("trading_hours", {})
    th_enabled = bool(th.get("enabled", False))
    th_start = parse_hhmm(th.get("start", "00:00"))
    th_end   = parse_hhmm(th.get("end", "23:59"))

    # guards
    g = cfg.get("guards", {})
    min_atr_pct = float(g.get("min_atr_pct", 0.0004))
    max_atr_pct = float(g.get("max_atr_pct", 0.035))
    max_spread_vs_atr = float(g.get("max_spread_vs_atr", 0.45))
    max_trades_per_day = int(g.get("max_trades_per_day", 200))

    # trade mgmt
    use_trailing = bool(cfg.get("use_trailing", True))
    trail_mult   = float(cfg.get("trail_atr_mult", 0.9))
    be_at_rr     = float(cfg.get("breakeven_at_rr", 1.0))
    max_hold     = int(cfg.get("max_hold_bars", 12))

    # symbol info
    point       = si.point or 0.01
    vol_min     = getattr(si, "volume_min", 0.01) or 0.01
    vol_step    = getattr(si, "volume_step", 0.01) or 0.01
    vol_max     = getattr(si, "volume_max", 100.0) or 100.0
    value_per_point_per_lot = compute_value_per_point(si)

    # AI
    ai_cfg      = cfg.get("ai", {})
    model       = load_ai(ai_cfg.get("model_path")) if ai_cfg.get("enabled") else None
    conf_th     = float(ai_cfg.get("decision_threshold", 0.65))

    # equity
    equity0 = 10000.0
    equity = equity0
    peak_equity = equity0

    last_trade_i = -999
    have_position = False
    rows = []

    df["date"] = df["time"].dt.date
    current_day = None
    day_start_equity = equity
    daily_trades = 0

    eq_list = [equity]

    # diagnostics counters
    reasons = {
        "cooldown":0,"one_pos":0,"no_side":0,"atr0":0,"ema200":0,"score":0,
        "atr_pct":0,"spread_vs_atr":0,"session":0,"daily_cap":0,"daily_loss":0,
        "ai_gate":0,"lot_too_small":0,"total_dd_stop":0
    }

    for i in range(250, len(df) - 1):
        total_dd = (peak_equity - equity) / max(peak_equity, EPS)
        if total_dd >= max_total_dd_pct:
            reasons["total_dd_stop"] += 1
            break

        day_i = df["date"].iloc[i]
        if current_day != day_i:
            current_day = day_i
            day_start_equity = equity
            daily_trades = 0

        if (day_start_equity - equity) / max(day_start_equity, EPS) >= max_daily_loss_pct:
            reasons["daily_loss"] += 1
            continue

        if th_enabled and not in_session(df["time"].iloc[i], th_start, th_end):
            reasons["session"] += 1
            continue

        if (i - last_trade_i) < cooldown:
            reasons["cooldown"] += 1
            continue
        if one_pos and have_position:
            reasons["one_pos"] += 1
            continue

        side = feats["side"][i]

        # توسيع منطق توليد side لو None
        if side is None:
            if (df["close"][i] > feats["ema200"][i]) and (feats["score"][i] >= 2 or feats["rsi_gt60"][i] == 1):
                side = "BUY"
            elif (df["close"][i] < feats["ema200"][i]) and (feats["score"][i] >= 2 or feats["rsi_lt40"][i] == 1):
                side = "SELL"

        if side is None:
            reasons["no_side"] += 1
            continue

        atr_val = feats["atr14"][i]
        if atr_val is None or atr_val <= 0:
            reasons["atr0"] += 1
            continue

        if use_ema200:
            if side == "BUY" and not (df["close"][i] > feats["ema200"][i]):
                reasons["ema200"] += 1; continue
            if side == "SELL" and not (df["close"][i] < feats["ema200"][i]):
                reasons["ema200"] += 1; continue

        # score gate: أساسي أو بديل مع pullback قريب من ema20
        score_ok = feats["score"][i] >= min_sig
        if not score_ok:
            dist_from_ema20 = abs(df["close"][i] - feats["ema20"][i])
            if feats["score"][i] >= 2 and dist_from_ema20 <= (0.5 * atr_val):
                score_ok = True
        if not score_ok:
            reasons["score"] += 1
            continue

        atr_pct = atr_val / max(df["close"][i], EPS)
        if atr_pct < min_atr_pct or atr_pct > max_atr_pct:
            reasons["atr_pct"] += 1; continue

        spread = spread_pts * point
        slip   = slip_pts * point
        if (spread + slip) > max_spread_vs_atr * atr_val:
            reasons["spread_vs_atr"] += 1; continue

        if daily_trades >= max_trades_per_day:
            reasons["daily_cap"] += 1; continue

        conf = 0.5
        if model is not None:
            row_dict = {
                "rsi14": float(feats["rsi14"][i]),
                "atr14": float(atr_val),
                "ema20": float(feats["ema20"][i]),
                "ema50": float(feats["ema50"][i]),
                "ema200": float(feats["ema200"][i]),
                "score": float(feats["score"][i]),
            }
            conf = ai_confidence(row_dict, model)
            if conf < conf_th:
                reasons["ai_gate"] += 1; continue

        atr_points = atr_val / point
        sl_points = max(1.0, sl_mult * atr_points)
        risk_usd = max(equity * risk_pct, 0.0)
        lots_raw = risk_usd / max(sl_points * value_per_point_per_lot, EPS)
        lots_steps = math.floor(lots_raw / vol_step)
        lots = max(vol_min, min(vol_max, lots_steps * vol_step))
        if lots < vol_min + EPS:
            reasons["lot_too_small"] += 1; continue

        entry = df["close"][i]
        if side == "BUY":
            price_in = entry + spread + slip
            tp_price = entry + pt_mult * atr_val
            sl_price = entry - sl_mult * atr_val
        else:
            price_in = entry - spread - slip
            tp_price = entry - pt_mult * atr_val
            sl_price = entry + sl_mult * atr_val

        # === إدارة الصفقة متعددة الشموع (حتى max_hold) ===
        price_out = None
        win = 0
        moved_to_be = False
        trail = trail_mult * atr_val if use_trailing else None
        be_sl = price_in

        last_bar_index = min(i + max_hold, len(df) - 1)
        for k in range(i + 1, last_bar_index + 1):
            bar = df.iloc[k]
            bh, bl, bc = float(bar["high"]), float(bar["low"]), float(bar["close"])

            if not moved_to_be:
                reached_1R = (bh >= (price_in + be_at_rr * (sl_mult * atr_val))) if side == "BUY" \
                             else (bl <= (price_in - be_at_rr * (sl_mult * atr_val)))
                if reached_1R:
                    moved_to_be = True

            hit_tp = (bh >= tp_price) if side == "BUY" else (bl <= tp_price)
            hit_sl_raw = (bl <= sl_price) if side == "BUY" else (bh >= sl_price)

            hit_trail = False
            dyn_sl = None
            if moved_to_be and use_trailing and trail is not None:
                if side == "BUY":
                    dyn_sl = max(be_sl, bc - trail)
                    hit_trail = bl <= dyn_sl
                else:
                    dyn_sl = min(be_sl, bc + trail)
                    hit_trail = bh >= dyn_sl

            if hit_tp:
                price_out = tp_price
                win = 1
                break
            if hit_trail:
                price_out = dyn_sl
                win = 1 if (side == "BUY" and price_out > price_in) or (side == "SELL" and price_out < price_in) else 0
                break
            if hit_sl_raw:
                if moved_to_be:
                    price_out = price_in  # BE
                    win = 0
                else:
                    price_out = sl_price
                    win = 0
                break

            if k == last_bar_index:
                price_out = bc
                win = 1 if ((price_out - price_in)/point > 0 and side=="BUY") or \
                            ((price_in - price_out)/point > 0 and side=="SELL") else 0
                break

        pnl_points = (price_out - price_in)/point if side=="BUY" else (price_in - price_out)/point
        pnl_usd = pnl_points * value_per_point_per_lot * lots - (com_per_lot * lots)

        equity = max(equity + pnl_usd, 0.0)
        peak_equity = max(peak_equity, equity)
        eq_list.append(equity)

        last_trade_i = i
        have_position = one_pos
        daily_trades += 1

        rows.append({
            "time": df["time"].iloc[min(last_bar_index, len(df)-1)],
            "date": str(df["date"].iloc[min(last_bar_index, len(df)-1)]),
            "side": side,
            "score": int(feats["score"][i]),
            "confidence": round(conf, 3),
            "lots": float(lots),
            "entry": round(price_in, 3),
            "exit": round(price_out, 3),
            "pnl_points": round(pnl_points, 2),
            "pnl_usd": round(pnl_usd, 2),
            "equity": round(equity, 2),
            "win": int(win)
        })

        if one_pos:
            have_position = False

        if equity <= EPS:
            break

    res = pd.DataFrame(rows)
    if res.empty:
        print("⚠️ لا توجد صفقات ضمن الشروط. عدّل thresholds أو زِد المدة (days) أو قلل decision_threshold.")
        print("🔎 Diagnostics:", {k:v for k,v in reasons.items() if v})
        return None, None

    wins = int(res["win"].sum())
    n = int(len(res))
    win_rate = 100.0 * wins / max(n, 1)
    pf = safe_profit_factor(res["pnl_usd"])

    equity_series = pd.Series(eq_list, dtype='float64')
    dd = max_drawdown_percent(equity_series)
    ret = (equity_series.iloc[-1] - 10000.0) / 10000.0 * 100.0

    out_csv = REPORTS_DIR / f"hybrid_ai_bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    res.to_csv(out_csv, index=False)

    stats = {
        "symbol": cfg["symbol"],
        "timeframe": cfg["timeframe"],
        "days": int(cfg["days"]),
        "trades": n,
        "win_rate": round(win_rate, 2),
        "profit_factor": (round(pf, 2) if pf != float("inf") else "Inf"),
        "max_dd": round(float(dd), 2),
        "return_pct": round(float(ret), 2),
        "file": str(out_csv)
    }
    return stats, res

# ====== main ======
def main():
    print(f"🔎 Loading config: {CONFIG_PATH}")
    cfg = load_cfg(CONFIG_PATH)
    show_keys = ['symbol','timeframe','days','min_signals','use_ema200_trend_filter']
    print(f"📦 Params: {json.dumps({k: cfg[k] for k in show_keys}, ensure_ascii=False)}")

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

        df = fetch_data(cfg["symbol"], cfg["timeframe"], cfg["days"])
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
