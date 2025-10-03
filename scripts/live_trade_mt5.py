# -*- coding: utf-8 -*-
"""
scripts/live_trade_mt5.py — Live engine (intra-bar entries + Trailing + Breakeven + Reversal Guard)
- يدعم الدخول داخل الشمعة (enter_anytime) مع entry_cooldown_sec لمنع السبام
- Breakeven مبكّر + Breakeven offset من الـ YAML
- Trailing Stop ديناميكي من الـ YAML
- Reversal Filter يتحسّس بدري (يفحص الشمعة الحيّة عند الدخول داخل الشمعة)
- وصفات SL/TP (ATR-Hybrid / Structure-SMC / Volatility-Pinch) + RR ذكي
- Fallback لحالة 10030: فتح بدون SL/TP ثم إضافتهم عبر TRADE_ACTION_SLTP
- اختيار تلقائي للفيلينغ (AUTO ثم حسب execution mode)

- NEW: max_open_positions و total_risk_cap_pct و risk_per_trade_pct تقرأ من YAML
"""

from __future__ import annotations
import os, sys, time, math, argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.run_hybrid_ensemble_ai import (
    load_cfg as load_ensemble_cfg,
    fetch_data, build_features, ensure_symbol_visible
)

try:
    import yaml
    import MetaTrader5 as mt5
except ImportError as e:
    print("❌ مفقود:", e); sys.exit(1)

CONFIG_ENSEMBLE = ROOT / "config" / "ensemble_ai.yaml"
DEFAULT_LIVE    = ROOT / "config" / "live.yaml"

# ===== سياسة مخاطرة عليا (تُستخدم كقيم افتراضية إذا غابت مفاتيح YAML) =====
PER_TRADE_RISK_PCT = 1.0     # % مخاطرة لكل صفقة (افتراضي)
MAX_OPEN_POSITIONS = 3       # أقصى صفقات مفتوحة (افتراضي)
TOTAL_RISK_CAP_PCT = 3.0     # سقف المخاطرة الكلي % (افتراضي)

# ---------- Utils ----------
def load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def within_hours(cfg_live: Dict[str, Any]) -> bool:
    h = (cfg_live.get("filters") or {}).get("trading_hours", {})
    if not h.get("enabled", False):
        return True
    now = datetime.now().time()
    st = datetime.strptime(h.get("start", "00:00"), "%H:%M").time()
    en = datetime.strptime(h.get("end", "23:59"), "%H:%M").time()
    return (st <= now <= en)

def get_account_info() -> Optional[Dict[str, Any]]:
    acc = mt5.account_info()
    if not acc:
        return None
    return {"balance": acc.balance, "equity": acc.equity, "login": acc.login}

def daily_pnl(log_dir: Path) -> float:
    day = datetime.now().strftime("%Y%m%d")
    p = log_dir / f"pnl_{day}.csv"
    if not p.exists(): return 0.0
    try:
        df = pd.read_csv(p)
        return float(df["pnl_usd"].sum()) if not df.empty else 0.0
    except Exception:
        return 0.0

def circuit_breaker_hit(cfg_live: Dict[str, Any], eq0: float, log_dir: Path) -> bool:
    max_daily_loss_pct = float((cfg_live.get("trade") or {}).get("max_daily_loss_pct", 0.0) or 0.0)
    if max_daily_loss_pct <= 0: return False
    pnl = daily_pnl(log_dir)
    return pnl <= -(eq0 * max_daily_loss_pct / 100.0)

def append_pnl_log(log_dir: Path, row: dict):
    day = datetime.now().strftime("%Y%m%d")
    p = log_dir / f"pnl_{day}.csv"
    df = pd.DataFrame([row])
    if p.exists():
        try:
            df0 = pd.read_csv(p); df = pd.concat([df0, df], ignore_index=True)
        except Exception:
            pass
    df.to_csv(p, index=False)

# ---------- Helpers: price/points ----------
def _normalize_down(x: float, step: float) -> float:
    return math.floor(x / step) * step

def _normalize_up(x: float, step: float) -> float:
    return math.ceil(x / step) * step

def _enforce_stops(side: str, price_in: float, sl_raw: float, tp_raw: float, si, expand: float = 1.0):
    """طبّع SL/TP على النقرة + احترم stop_level + هامش أمان."""
    point = si.point or 0.01
    stop_level_pts = int(getattr(si, "trade_stops_level", 0) or 0)
    min_gap = stop_level_pts * point
    safety_pad = 3 * point
    gap = (min_gap + safety_pad) * max(1.0, expand)

    if side == "BUY":
        sl = _normalize_down(sl_raw, point)
        tp = _normalize_up(tp_raw, point)
        sl = min(sl, price_in - gap)
        tp = max(tp, price_in + gap)
        if not (sl < price_in and tp > price_in):
            sl = price_in - max(gap, abs(price_in - sl) * 1.25)
            tp = price_in + max(gap, abs(price_in - tp) * 1.25)
            sl = _normalize_down(sl, point); tp = _normalize_up(tp, point)
    else:
        sl = _normalize_up(sl_raw, point)
        tp = _normalize_down(tp_raw, point)
        sl = max(sl, price_in + gap)
        tp = min(tp, price_in - gap)
        if not (sl > price_in and tp < price_in):
            sl = price_in + max(gap, abs(sl - price_in) * 1.25)
            tp = price_in - max(gap, abs(price_in - tp) * 1.25)
            sl = _normalize_up(sl, point); tp = _normalize_down(tp, point)
    return float(sl), float(tp)

# ---------- NEW: pick filling mode by execution + allow AUTO ----------
def _pick_filling_mode(si):
    """
    نرجّع قائمة مرشحات filling:
    - أولاً: AUTO (بدون type_filling)
    - بعدها ترتيب حسب execution mode
    """
    IOC = getattr(mt5, "ORDER_FILLING_IOC", 1)
    FOK = getattr(mt5, "ORDER_FILLING_FOK", 2)
    RET = getattr(mt5, "ORDER_FILLING_RETURN", 0)

    exec_mode = int(getattr(si, "trade_exemode", 0) or 0)
    EX_REQ = getattr(mt5, "SYMBOL_TRADE_EXECUTION_REQUEST", 0)
    EX_MKT = getattr(mt5, "SYMBOL_TRADE_EXECUTION_MARKET", 1)
    EX_EXC = getattr(mt5, "SYMBOL_TRADE_EXECUTION_EXCHANGE", 2)

    try:
        print(f"[SYMBOL] exec_mode={exec_mode} filling_mode={getattr(si,'filling_mode',None)}")
    except Exception:
        pass

    if exec_mode == EX_MKT:
        return [None, FOK, IOC, RET]
    elif exec_mode == EX_EXC:
        return [None, RET, IOC, FOK]
    else:  # REQUEST/INSTANT أو غير معروف
        return [None, IOC, FOK, RET]

# ---------- Position sizing (uses YAML-configurable limits) ----------
def _open_positions_count() -> int:
    poss = mt5.positions_get()
    return len(poss) if poss is not None else 0

def position_size(si, cfg_live, atr_val, sl_mult):
    """
    يحسب اللوت ديناميكيًا من مخاطرة %، ويطبّق:
      - حد أقصى للصفقات المفتوحة: trade.max_open_positions (افتراضيًا MAX_OPEN_POSITIONS)
      - سقف مخاطرة كلّي: trade.total_risk_cap_pct (افتراضيًا TOTAL_RISK_CAP_PCT)
      - مخاطرة لكل صفقة: trade.risk_per_trade_pct (افتراضيًا PER_TRADE_RISK_PCT)
    """
    trade_cfg = (cfg_live.get("trade") or {})

    # اقرأ الحدود من YAML مع fallback للثوابت الافتراضية
    max_open = int(trade_cfg.get("max_open_positions", MAX_OPEN_POSITIONS))
    risk_per_trade_pct = float(trade_cfg.get("risk_per_trade_pct", PER_TRADE_RISK_PCT))
    total_cap_pct = float(trade_cfg.get("total_risk_cap_pct", TOTAL_RISK_CAP_PCT))

    open_cnt = _open_positions_count()
    if open_cnt >= max_open:
        if cfg_live.get("debug", False):
            print(f"[DEBUG] blocked: open positions={open_cnt} >= {max_open}")
        return 0.0

    remaining_cap_pct = total_cap_pct - open_cnt * risk_per_trade_pct
    if remaining_cap_pct <= 0:
        if cfg_live.get("debug", False):
            print(f"[DEBUG] blocked: remaining risk cap <= 0 (cap={total_cap_pct}%, open={open_cnt} × {risk_per_trade_pct}%)")
        return 0.0

    risk_pct_to_use = min(risk_per_trade_pct, remaining_cap_pct)

    point = float(si.point or 0.01)
    tick_size  = float(getattr(si, "trade_tick_size", point) or point)
    tick_value = float(getattr(si, "trade_tick_value", 1.0) or 1.0)

    atr_points = atr_val / point
    sl_points  = max(1.0, sl_mult * atr_points)
    sl_distance_price = sl_points * point

    value_per_1_price_per_lot = (1.0 / tick_size) * tick_value
    denom = sl_distance_price * value_per_1_price_per_lot
    if denom <= 0: return 0.0

    eq = (get_account_info() or {}).get("equity", 0.0) or 0.0
    risk_usd = eq * (risk_pct_to_use / 100.0)

    lots_raw = risk_usd / denom

    vol_min  = float(getattr(si, "volume_min", 0.01) or 0.01)
    vol_step = float(getattr(si, "volume_step", 0.01) or 0.01)

    lot_max_yaml = float(trade_cfg.get("lot_max", float("inf")) or float("inf"))
    cap_yaml     = float(trade_cfg.get("max_volume_cap", float("inf")) or float("inf"))

    lots = math.floor(max(vol_min, lots_raw) / vol_step) * vol_step
    lots = min(lots, lot_max_yaml, cap_yaml)

    return float(max(0.0, round(lots, 4)))

# ---------- Signal ----------
def ensemble_signal(df, feats, cfg_ens, si, debug=False):
    # NB: الـ i يتم تمريره من اللوب (−1 للشمعة الحيّة أو −2 للمكتملة)
    # هذه الدالة تستخدم i من اللوب فقط للفحص
    i = len(df) - 2  # fallback لو تم استدعاؤها منفردة
    try:
        # استرشادي فقط؛ اللوب يقرّر أي شمعة يستخدم
        pass
    except Exception:
        pass

    if len(df) < 250:
        if debug: print("[DEBUG] not enough bars for features (need >=250).")
        return None, None

    tf = str(cfg_ens.get("timeframe", "")).upper()
    use_trend = bool(cfg_ens.get("use_ema200_trend_filter", True))

    # نستخدم آخر شمعة مكتملة للفحص الاسترشادي (اللوب بمرر i فعليًا)
    i = len(df) - 2
    above = df["close"].iloc[i] > feats["ema200"].iloc[i]

    rsi_buy_th  = 53 if tf == "M1" else 55
    rsi_sell_th = 47 if tf == "M1" else 45

    cond_buy  = (feats["ema20"].iloc[i] > feats["ema50"].iloc[i]) and (feats["rsi14"].iloc[i] > rsi_buy_th)
    cond_sell = (feats["ema20"].iloc[i] < feats["ema50"].iloc[i]) and (feats["rsi14"].iloc[i] < rsi_sell_th)

    if use_trend:
        cond_buy  = cond_buy  and above
        cond_sell = cond_sell and (not above)

    weights = (cfg_ens.get("strategies") or {})
    tot_w = sum(spec.get("weight", 1.0) for _, spec in weights.items() if spec.get("enabled", True)) or 1.0
    vote_buy = vote_sell = 0.0
    for _, spec in weights.items():
        if not spec.get("enabled", True): continue
        w = float(spec.get("weight", 1.0))
        if cond_buy:  vote_buy  += w
        if cond_sell: vote_sell += w

    conf = max(vote_buy, vote_sell) / tot_w
    min_conf = float(cfg_ens.get("min_ensemble_conf", 0.55))

    if debug:
        print(
            f"[DEBUG] {df['time'].iloc[i]} "
            f"RSI={feats['rsi14'].iloc[i]:.1f} "
            f"E20>E50={'Y' if feats['ema20'].iloc[i] > feats['ema50'].iloc[i] else 'N'} "
            f"above200={'Y' if above else 'N'} useTrend={'Y' if use_trend else 'N'} "
            f"conf={conf:.2f}/{min_conf:.2f}"
        )

    if vote_buy >= vote_sell and conf >= min_conf and cond_buy:
        return "BUY", i
    if vote_sell > vote_buy and conf >= min_conf and cond_sell:
        return "SELL", i
    return None, None

# ---------- NEW: SL/TP recipes ----------
def _rr_adjust_by_conditions(rr_base: float, df: pd.DataFrame, feats: pd.DataFrame, i: int, atr: float, cfg_live: dict) -> float:
    rr = rr_base
    try:
        trend_strong = abs(float(feats["ema20"].iloc[i]) - float(feats["ema200"].iloc[i])) > float(atr)
    except Exception:
        trend_strong = False
    if trend_strong: rr *= 1.10

    try:
        high_vol = float(atr) > float(df["close"].iloc[i]) * 0.005
    except Exception:
        high_vol = False
    if high_vol: rr *= 0.90

    try:
        eq0 = (get_account_info() or {}).get("equity", 0.0) or 0.0
        max_daily_loss_pct = float((cfg_live.get("trade") or {}).get("max_daily_loss_pct", 0) or 0)
        if max_daily_loss_pct > 0:
            pnl_today = daily_pnl((ROOT / (cfg_live.get("logs") or {}).get("dir", "reports/live_logs")).resolve())
            if pnl_today < 0 and abs(pnl_today) >= 0.6 * (eq0 * max_daily_loss_pct / 100.0):
                rr *= 0.85
    except Exception:
        pass
    return max(1.05, float(rr))

def pick_sl_tp_recipe(side: str, df: pd.DataFrame, feats: pd.DataFrame, i: int, atr_val: float, si, cfg_live: dict):
    """
    يرجّع (sl_pts_price, tp_pts_price) كوحدات سعر (وليس نقاط)
    """
    close = float(df["close"].iloc[i])
    atr = float(atr_val)

    # حد أدنى من YAML أو 10 ticks
    yaml_min_sl_pts = float((cfg_live.get("trade") or {}).get("min_sl_points", 0) or 0)
    min_stop_points = yaml_min_sl_pts if yaml_min_sl_pts > 0 else float(si.point or 0.01) * 10.0

    # ATR-Hybrid
    sl_pts = max(atr * 1.3, min_stop_points)
    rr = 1.8

    # Structure-SMC (اختياري)
    use_smc = bool((cfg_live.get("trade") or {}).get("use_smc_stops", False))
    if use_smc and i >= 12:
        try:
            look = 10
            swing_low  = float(df["low"].iloc[i - look:i].min())
            swing_high = float(df["high"].iloc[i - look:i].max())
            buffer = 0.25 * atr
            if side == "BUY":
                sl_pts = abs(close - (swing_low - buffer))
            else:
                sl_pts = abs((swing_high + buffer) - close)
        except Exception:
            pass
        rr = 1.8

    # Volatility-Pinch (خنق/سكون)
    try:
        window = int((cfg_live.get("trade") or {}).get("atr_percentile_window", 300) or 300)
        atr_series = pd.Series(feats.get("atr14", df.get("atr14")))
        atr_window = atr_series.iloc[max(0, i - window):i].dropna()
        if len(atr_window) >= 30:
            p30 = float(np.percentile(atr_window.values, 30))
            if atr <= p30:
                sl_pts = max(atr * 1.0, min_stop_points)
                rr = 1.35
                (cfg_live.setdefault("trade", {}))["use_fast_trailing"] = True
    except Exception:
        pass

    rr = _rr_adjust_by_conditions(rr, df, feats, i, atr, cfg_live)
    tp_pts = float(sl_pts) * float(rr)
    return float(sl_pts), float(tp_pts)

# ---------- Helpers: modify SL/TP after open ----------
def _set_sltp_for_position(symbol: str, sl: float, tp: float) -> bool:
    """إضافة/تعديل SL/TP للصفقة المفتوحة عبر TRADE_ACTION_SLTP."""
    poss = mt5.positions_get(symbol=symbol)
    if not poss:
        # حاول بدون فلترة الرمز
        poss2 = mt5.positions_get()
        if not poss2: return False
        pos = poss2[0]
        symbol = getattr(pos, "symbol", symbol)
    else:
        pos = poss[0]

    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": int(pos.ticket),
        "sl": float(sl),
        "tp": float(tp),
        "magic": int(getattr(pos, "magic", 0)),
        "comment": "Set SLTP",
    }
    res = mt5.order_send(req)
    return bool(res and res.retcode == mt5.TRADE_RETCODE_DONE)

# ---------- Loose symbol match (positions) ----------
def _get_position_for_symbol_loose(symbol: str):
    """يحاول إيجاد الصفقة حتى لو الرمز فيه لاحقة/نقاط."""
    poss = mt5.positions_get()
    if not poss:
        return None
    s_norm = (symbol or "").replace('.', '').upper()
    for p in poss:
        p_sym = getattr(p, "symbol", "") or ""
        if p_sym == symbol:
            return p
        if p_sym.replace('.', '').upper() == s_norm:
            return p
    return None

# ---------- Trailing + Breakeven ----------
def apply_trailing_and_breakeven(symbol: str, cfg_live: dict):
    """تطبيق التريلنغ والبريك إيفن على أي صفقة مفتوحة (تُستدعى كل لفة)."""
    pos = _get_position_for_symbol_loose(symbol)
    if not pos:
        if cfg_live.get("debug", False):
            print(f"[DEBUG] No position found for {symbol}.")
        return

    side = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
    price_open = float(pos.price_open)
    cur_sl = float(pos.sl) if pos.sl is not None else 0.0
    tp = float(pos.tp) if pos.tp is not None else 0.0

    tick = mt5.symbol_info_tick(pos.symbol)
    si = mt5.symbol_info(pos.symbol)
    if not tick or not si:
        if cfg_live.get("debug", False):
            print("[DEBUG] symbol_info_tick/symbol_info None.")
        return

    point = si.point or 0.01
    bid, ask = float(tick.bid), float(tick.ask)
    changed = False

    # --- Breakeven ---
    if (cfg_live.get("trade") or {}).get("use_breakeven", False):
        trigger_pts = float((cfg_live["trade"]).get("breakeven_trigger_points", 100) or 100)
        offset_pts  = float((cfg_live["trade"]).get("breakeven_offset_points", 0.0) or 0.0)
        offset = offset_pts * point
        if side == "BUY":
            moved = (bid - price_open) / point
            if cfg_live.get("debug", False):
                print(f"[DEBUG] BE BUY moved={moved:.1f} trig={trigger_pts} sl={cur_sl:.2f} open={price_open:.2f}")
            if moved >= trigger_pts and (cur_sl < price_open):
                cur_sl = price_open + offset
                changed = True
        else:
            moved = (price_open - ask) / point
            if cfg_live.get("debug", False):
                print(f"[DEBUG] BE SELL moved={moved:.1f} trig={trigger_pts} sl={cur_sl:.2f} open={price_open:.2f}")
            if moved >= trigger_pts and (cur_sl == 0.0 or cur_sl > price_open):
                cur_sl = price_open - offset
                changed = True

    # --- Trailing Stop ---
    if (cfg_live.get("trade") or {}).get("use_trailing", False):
        dist_pts = float((cfg_live["trade"]).get("trailing_distance_points", 100) or 100)
        if side == "BUY":
            new_sl = bid - dist_pts * point
            if cfg_live.get("debug", False):
                print(f"[DEBUG] TR BUY new_sl={new_sl:.2f} cur_sl={cur_sl:.2f} dist={dist_pts}")
            if new_sl > cur_sl:
                cur_sl = new_sl
                changed = True
        else:
            new_sl = ask + dist_pts * point
            if cfg_live.get("debug", False):
                print(f"[DEBUG] TR SELL new_sl={new_sl:.2f} cur_sl={cur_sl:.2f} dist={dist_pts}")
            if cur_sl == 0.0 or new_sl < cur_sl:
                cur_sl = new_sl
                changed = True

    if changed:
        ok = _set_sltp_for_position(pos.symbol, float(cur_sl), float(tp))
        if cfg_live.get("debug", False):
            print(f"[DEBUG] SLTP modify -> {'OK' if ok else 'FAILED'} | new SL={cur_sl:.2f} TP={tp:.2f}")

# ---------- Main Loop ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-config", type=str, default=str(DEFAULT_LIVE), help="مسار ملف live.yaml البديل")
    ap.add_argument("--debug", action="store_true", help="تشخيص: اطبع أسباب رفض الإشارات")
    args = ap.parse_args()

    live_path = Path(args.live_config).resolve()
    if not live_path.exists():
        print(f"❌ live config غير موجود: {live_path}"); sys.exit(1)

    cfg_live = load_yaml(live_path)
    cfg_ens  = load_ensemble_cfg(CONFIG_ENSEMBLE)

    symbol    = (cfg_live.get("trade") or {}).get("symbol")
    timeframe = (cfg_live.get("trade") or {}).get("timeframe")
    days      = max(10, int(cfg_ens.get("days", 180)))

    log_dir = (ROOT / (cfg_live.get("logs") or {}).get("dir", "reports/live_logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    if not mt5.initialize():
        print("❌ mt5.initialize failed:", mt5.last_error()); sys.exit(1)

    si = ensure_symbol_visible(symbol)
    if not si:
        print(f"❌ Symbol not visible: {symbol}"); sys.exit(1)

    acc = get_account_info()
    if not acc:
        print("❌ لا يمكن قراءة معلومات الحساب من MT5."); sys.exit(1)
    eq0 = acc["equity"]
    print(f"▶️ Live trading on {symbol} {timeframe} | acct={acc['login']} | eq={eq0:.2f}")
    print(f"⚙️ Using live config: {live_path}")

    # تتبّع آخر شمعة مكتملة
    last_bar_time = None
    # ختم وقت آخر دخول (للكوولدوان)
    last_entry_ts = 0.0

    try:
        while True:
            if not within_hours(cfg_live):
                time.sleep(5)
                apply_trailing_and_breakeven(symbol, cfg_live)
                continue

            if circuit_breaker_hit(cfg_live, eq0, log_dir):
                print("⛔ Circuit Breaker hit. Pausing new entries.")
                time.sleep(30)
                apply_trailing_and_breakeven(symbol, cfg_live)
                continue

            try:
                data = fetch_data(symbol, timeframe, days)
            except Exception as e:
                print("⚠️ fetch_data:", e); time.sleep(5)
                apply_trailing_and_breakeven(symbol, cfg_live)
                continue

            df = data[0] if isinstance(data, tuple) else data
            if df is None or not hasattr(df, "index") or len(df) < 300:
                if args.debug: print("[DEBUG] df is empty/short (len < 300). Waiting…")
                time.sleep(2)
                apply_trailing_and_breakeven(symbol, cfg_live)
                continue

            feats = build_features(df)

            cur_last = df["time"].iloc[-1]
            if last_bar_time is None:
                last_bar_time = cur_last

            # ==== دخول داخل الشمعة (enter_anytime) + كوولدوان ====
            enter_any = bool((cfg_live.get("trade") or {}).get("enter_anytime", False))
            # قرّر أي شمعة نفحص:
            # - لو enter_anytime=True → استخدم الشمعة الحيّة i = -1
            # - غير هيك → الشمعة المكتملة i = -2
            i = len(df) - (1 if enter_any else 2)

            side = None
            # افحص الآن إذا:
            #   - تشكّلت شمعة جديدة، أو
            #   - مسموح لنا نفحص intra-bar
            should_eval_now = (cur_last != last_bar_time) or enter_any

            if should_eval_now and i >= 2:
                # NB: ensemble_signal يستخدم i داخليًا استرشاديًا؛ قرار i هنا
                side, _ = ensemble_signal(df, feats, cfg_ens, si, debug=args.debug)

                # --- مانع الانعكاس المبكر ---
                try:
                    rev_cfg = ((cfg_live.get("trade") or {}).get("reversal_filter") or {})
                    if rev_cfg.get("enabled", False) and side:
                        min_body_ratio = float(rev_cfg.get("min_body_ratio", 0.6))
                        # لو داخل الشمعة، افحص الشمعة الحيّة نفسها؛ غير هيك افحص الشمعة السابقة
                        chk_candle = df.iloc[-1] if enter_any else df.iloc[i - 1]
                        body = abs(float(chk_candle["close"]) - float(chk_candle["open"]))
                        rng = float(chk_candle["high"]) - float(chk_candle["low"])
                        strong_body = (rng > 0) and ((body / rng) >= min_body_ratio)
                        if strong_body:
                            bearish = float(chk_candle["close"]) < float(chk_candle["open"])
                            bullish = float(chk_candle["close"]) > float(chk_candle["open"])
                            if (side == "BUY" and bearish) or (side == "SELL" and bullish):
                                if args.debug: print("[DEBUG] Reversal filter (live) blocked entry.")
                                side = None
                except Exception as e:
                    if args.debug: print("[DEBUG] reversal filter error:", e)

                # --- كوولدوان لمنع سبام داخل نفس الشمعة/الدقيقة ---
                if side and enter_any:
                    now_ts = time.time()
                    cooldown = float((cfg_live.get("trade") or {}).get("entry_cooldown_sec", 20))
                    if (now_ts - last_entry_ts) < cooldown:
                        if args.debug:
                            print(f"[DEBUG] entry cooldown {(now_ts - last_entry_ts):.1f}s < {cooldown}s → skip")
                        side = None

                if side:
                    # === ATR لازم يكون جاهز
                    atr_v = float(feats["atr14"].iloc[i]) if not pd.isna(feats["atr14"].iloc[i]) else None
                    if atr_v is None or atr_v <= 0:
                        if args.debug: print("[DEBUG] ATR not ready; skip entry.")
                        # لا تدخل الآن؛ حدّث آخر شمعة إذا كانت جديدة وطبّق إدارة صفقة
                        if cur_last != last_bar_time: last_bar_time = cur_last
                        time.sleep(2)
                        apply_trailing_and_breakeven(symbol, cfg_live)
                        continue

                    # === احسب SL/TP (وصفة ذكية) ===
                    sl_pts, tp_pts = pick_sl_tp_recipe(side, df, feats, i, atr_v, si, cfg_live)
                    sl_mult_effective = max(0.1, float(sl_pts) / float(atr_v))

                    lots = position_size(si, cfg_live, atr_v, sl_mult_effective)
                    if lots <= 0:
                        if args.debug: print("[DEBUG] Lot calc <= 0; skip")
                        if cur_last != last_bar_time: last_bar_time = cur_last
                        time.sleep(2)
                        apply_trailing_and_breakeven(symbol, cfg_live)
                        continue

                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        if args.debug: print("[DEBUG] symbol_info_tick None; skip")
                        if cur_last != last_bar_time: last_bar_time = cur_last
                        time.sleep(2)
                        apply_trailing_and_breakeven(symbol, cfg_live)
                        continue

                    point = si.point or 0.01
                    spread_points = int((cfg_live.get("trade") or {}).get("spread_points", 0) or 0)
                    spread = spread_points * point

                    if side == "BUY":
                        price_in = float(tick.ask) + spread
                        sl_raw = price_in - sl_pts
                        tp_raw = price_in + tp_pts
                        req_type = mt5.ORDER_TYPE_BUY
                    else:
                        price_in = float(tick.bid) - spread
                        sl_raw = price_in + sl_pts
                        tp_raw = price_in - tp_pts
                        req_type = mt5.ORDER_TYPE_SELL

                    sl, tp = _enforce_stops(side, price_in, sl_raw, tp_raw, si, expand=1.0)
                    deviation = int((cfg_live.get("trade") or {}).get("slippage_points", 120))

                    # ====== إرسال الطلب ======
                    base_req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": float(lots),
                        "type": req_type,
                        "price": float(price_in),
                        "sl": float(sl),
                        "tp": float(tp),
                        "deviation": deviation,
                        "magic": int((cfg_live.get("trade") or {}).get("magic", 0)),
                        "comment": "LiveEnsemble",
                    }

                    filling_candidates = _pick_filling_mode(si)

                    sent_ok = False
                    last_err = None

                    # 1) جرّب AUTO ثم fallback بأنماط مناسبة
                    for tfm in filling_candidates:
                        req = dict(base_req)
                        if tfm is None:
                            req.pop("type_filling", None)  # AUTO
                        else:
                            req["type_filling"] = tfm

                        res = mt5.order_send(req)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            append_pnl_log(log_dir, {"time": datetime.now().isoformat(),
                                                     "side": side, "lots": lots, "pnl_usd": 0.0, "event": "OPEN"})
                            print(datetime.now().strftime("%H:%M:%S"),
                                  f"Opened {side} {lots} @ {price_in:.2f} [fill={'AUTO' if tfm is None else tfm}]")
                            sent_ok = True
                            last_entry_ts = time.time()
                            break
                        else:
                            last_err = (getattr(res, "retcode", None), getattr(res, "comment", ""))
                            # Invalid stops → وسّع شوي وجرّب نفس النمط
                            if last_err[0] in (mt5.TRADE_RETCODE_INVALID_STOPS, 10016, 10021, 10022):
                                sl2, tp2 = _enforce_stops(side, price_in, sl_raw, tp_raw, si, expand=1.35)
                                req["sl"], req["tp"] = float(sl2), float(tp2)
                                res2 = mt5.order_send(req)
                                if res2 and res2.retcode == mt5.TRADE_RETCODE_DONE:
                                    append_pnl_log(log_dir, {"time": datetime.now().isoformat(),
                                                             "side": side, "lots": lots, "pnl_usd": 0.0, "event": "OPEN"})
                                    print(datetime.now().strftime("%H:%M:%S"),
                                          f"Opened {side} {lots} @ {price_in:.2f} [fill={'AUTO' if tfm is None else tfm}, retry]")
                                    sent_ok = True
                                    last_entry_ts = time.time()
                                    break

                    # 2) لو لسه 10030 أو فشل كل المحاولات → افتح بدون SL/TP ثم عدّلهم
                    if not sent_ok and last_err and last_err[0] == 10030:
                        req_no_sltp = dict(base_req)
                        req_no_sltp.pop("sl", None); req_no_sltp.pop("tp", None); req_no_sltp.pop("type_filling", None)
                        res3 = mt5.order_send(req_no_sltp)
                        if res3 and res3.retcode == mt5.TRADE_RETCODE_DONE:
                            # أضف SL/TP فورًا
                            ok_mod = _set_sltp_for_position(symbol, sl, tp)
                            append_pnl_log(log_dir, {"time": datetime.now().isoformat(),
                                                     "side": side, "lots": lots, "pnl_usd": 0.0, "event": "OPEN"})
                            print(datetime.now().strftime("%H:%M:%S"),
                                  f"Opened {side} {lots} @ {price_in:.2f} [fill=OPEN_NO_SLTP] "
                                  + ("+ SLTP set" if ok_mod else "+ SLTP FAILED"))
                            last_entry_ts = time.time()
                        else:
                            err_code = last_err[0] if last_err else None
                            err_msg  = last_err[1] if last_err else ""
                            print(f"❌ order_send failed: {err_code} {err_msg}")

            # حدّث وقت آخر شمعة مكتملة فقط (مشان نعرف متى تكونت شمعة جديدة)
            if cur_last != last_bar_time:
                last_bar_time = cur_last

            # في كل لفة: طبّق التريلنغ والبريك إيفن على أي صفقة مفتوحة
            apply_trailing_and_breakeven(symbol, cfg_live)

            time.sleep(2)

    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
