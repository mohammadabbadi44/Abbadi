# -*- coding: utf-8 -*-
"""
scripts/manage_positions_mt5.py

Position Manager ذكي ل MT5:
- partial take على مستويات RR
- نقل SL ل breakeven عند RR معيّن
- trailing متدرّج مبني على ATR
- يحترم magic/symbol ويشتغل 24/7 أو حسب live-config
- يقرأ إعداداته من live.yaml تحت المفتاح manager:

تشغيل:
  python -m scripts.manage_positions_mt5 --live-config config\live_m1.yaml --debug
"""

import sys, time, math, argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

try:
    import yaml
    import MetaTrader5 as mt5
except ImportError as e:
    print("❌ Missing:", e); sys.exit(1)

# نعيد استخدام دوال الداتا/الفيتشرز من مشروعك
from scripts.run_hybrid_ensemble_ai import (
    fetch_data, build_features, ensure_symbol_visible, load_cfg as load_ensemble_cfg
)

CONFIG_ENSEMBLE = ROOT / "config" / "ensemble_ai.yaml"
DEFAULT_LIVE    = ROOT / "config" / "live.yaml"

# ---------- Utils ----------
def load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def within_hours(cfg_live):
    th = cfg_live.get("filters", {}).get("trading_hours", {})
    if not th.get("enabled", False):
        return True
    now = datetime.now().time()
    from datetime import datetime as _dt
    st = _dt.strptime(th.get("start","00:00"), "%H:%M").time()
    en = _dt.strptime(th.get("end","23:59"), "%H:%M").time()
    return (st <= now <= en)

def point_info(si):
    return si.point or 0.01, getattr(si, "trade_tick_value", 1.0) or 1.0

def symbol_positions(symbol: str, magic: int):
    res = []
    for p in (mt5.positions_get(symbol=symbol) or []):
        if p.magic == magic:
            res.append(p)
    return res

def account_ok():
    acc = mt5.account_info()
    return acc is not None

def modify_sl_tp(position, sl=None, tp=None):
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": position.symbol,
        "position": position.ticket,
        "sl": sl if sl is not None else position.sl,
        "tp": tp if tp is not None else position.tp,
        "magic": position.magic,
        "comment": "MgrSLTP",
    }
    return mt5.order_send(req)

def close_partial(position, volume, price=None):
    """إغلاق جزئي: نرسل صفقة بالاتجاه المعاكس بنفس الحجم المطلوب"""
    typ = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": float(volume),
        "type": typ,
        "price": price,
        "deviation": 50,
        "magic": position.magic,
        "comment": "MgrPartial",
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    return mt5.order_send(req)

def rr_now(position, atr_val):
    """احسب R:R اللحظي = المسافة الحالية بالاتجاه / مسافة SL الابتدائية.
       لو SL تم تحريكه، نستخدم المسافة الحالية لـ SL كمرجع."""
    if atr_val is None or atr_val <= 0:
        return None
    # مسافة وقف الخسارة الحالية:
    stop_dist = abs((position.price_open - position.sl)) if position.sl else atr_val*3.0
    if stop_dist <= 1e-9:
        stop_dist = atr_val*3.0
    # ربح/خسارة عائم كنقاط سعر:
    price = mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.POSITION_TYPE_SELL else mt5.symbol_info_tick(position.symbol).ask
    move_dist = (price - position.price_open) if position.type == mt5.POSITION_TYPE_BUY else (position.price_open - price)
    return max(0.0, move_dist / stop_dist)

def min_price_step(si):
    # مسافة دنيا للسعر لتعديل SL/TP
    return getattr(si, "trade_tick_size", None) or (si.point or 0.01)

# ---------- Manager Core ----------
def manage_positions(cfg_live, cfg_ens, si, debug=False):
    symbol = cfg_live["trade"]["symbol"]
    magic  = int(cfg_live["trade"]["magic"])
    timeframe = cfg_live["trade"]["timeframe"]
    days = max(5, cfg_ens.get("days", 90))

    # إعدادات المدير
    mgr = cfg_live.get("manager", {})
    rr_levels       = mgr.get("partial_rr_levels",  [0.8, 1.2, 1.8])
    rr_closes       = mgr.get("partial_close_pcts", [0.25,0.35,0.40])  # مجموعها 1.0 اختياري
    move_be_at_rr   = float(mgr.get("move_to_be_at_rr", 1.0))
    trail_start_rr  = float(mgr.get("trail_start_rr", 1.2))
    trail_atr_mult  = float(mgr.get("trail_atr_mult", 1.0))
    trail_step_mult = float(mgr.get("trail_step_mult", 0.6))  # كل ما مشى السعر 0.6*ATR نزحل SL بمقدار مماثل
    max_hold_minutes= int(mgr.get("max_hold_minutes", 0))     # 0 = تعطيل
    respect_spread  = bool(mgr.get("respect_spread", True))

    spread_points   = int(cfg_live["trade"]["spread_points"])
    pt, tick_value  = point_info(si)
    step_min        = min_price_step(si)

    # جلب ATR حديث
    try:
        data = fetch_data(symbol, timeframe, days)
        df = data[0] if isinstance(data, tuple) else data
        feats = build_features(df)
        atr_val = float(feats["atr14"].iloc[-2])
    except Exception as e:
        if debug: print("⚠️ ATR fetch/build failed:", e)
        atr_val = None

    positions = symbol_positions(symbol, magic)
    if not positions:
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return

    for pos in positions:
        # حساب RR الحالي
        cur_rr = rr_now(pos, atr_val)
        if debug:
            print(f"[MGR] ticket={pos.ticket} type={'BUY' if pos.type==0 else 'SELL'} vol={pos.volume:.2f} rr={cur_rr}")

        # 1) نقل SL إلى BE عند RR معين
        if cur_rr is not None and cur_rr >= move_be_at_rr and pos.sl is not None:
            be = pos.price_open
            # لا ننقل إلا إذا BE أفضل من SL الحالي
            if (pos.type == mt5.POSITION_TYPE_BUY and (pos.sl is None or be > pos.sl + step_min)) or \
               (pos.type == mt5.POSITION_TYPE_SELL and (pos.sl is None or be < pos.sl - step_min)):
                res = modify_sl_tp(pos, sl=be, tp=None)
                if debug: print(f"[MGR] → BE move SL to {be:.2f} ret={getattr(res,'retcode',None)}")

        # 2) Partial closes على مستويات RR
        if cur_rr is not None and rr_levels and rr_closes and len(rr_levels)==len(rr_closes):
            # بنحدد كم مستوى تحقق ولم يُغلق جزئياً بعد (نستخدم comment workaround عبر حجم الصفقة المتبقي)
            # بسيط: لو حجم الصفقة الحالية > الحجم الابتدائي*(1 - مجموع الإغلاقات حتى المستوى)
            try:
                initial_vol = float(getattr(pos, "volume_initial", pos.volume))
            except Exception:
                initial_vol = pos.volume
            for lvl, pct in zip(rr_levels, rr_closes):
                if cur_rr >= lvl and pos.volume > max(0.01, initial_vol * (1.0 - pct) + 1e-6):
                    vol_to_close = max(0.01, round(initial_vol * pct, 2))
                    # لا تتجاوز الحجم الحالي
                    vol_to_close = min(vol_to_close, pos.volume - 0.01)
                    if vol_to_close > 0:
                        px = tick.bid if pos.type==mt5.POSITION_TYPE_BUY else tick.ask
                        res = close_partial(pos, vol_to_close, price=px)
                        if debug: print(f"[MGR] → Partial close {vol_to_close} at RR={lvl} ret={getattr(res,'retcode',None)}")

        # 3) Trailing متدرّج بعد RR معين
        if atr_val and cur_rr is not None and cur_rr >= trail_start_rr:
            # مسافة trailing الحالية = trail_atr_mult * ATR
            trail_dist = max(step_min, trail_atr_mult * atr_val)
            # سعر SL المقترح
            if pos.type == mt5.POSITION_TYPE_BUY:
                new_sl = max(pos.sl or -1e9, tick.bid - trail_dist)
                if new_sl > (pos.sl or -1e9) + step_min:
                    res = modify_sl_tp(pos, sl=new_sl, tp=None)
                    if debug: print(f"[MGR] → Trail SL BUY to {new_sl:.2f} ret={getattr(res,'retcode',None)}")
            else:
                new_sl = min(pos.sl or 1e9, tick.ask + trail_dist)
                if new_sl < (pos.sl or 1e9) - step_min:
                    res = modify_sl_tp(pos, sl=new_sl, tp=None)
                    if debug: print(f"[MGR] → Trail SL SELL to {new_sl:.2f} ret={getattr(res,'retcode',None)}")

        # 4) Max-hold time (اختياري)
        if max_hold_minutes > 0:
            open_time = datetime.fromtimestamp(pos.time)
            if datetime.now() - open_time >= timedelta(minutes=max_hold_minutes):
                # إغلاق كامل
                vol = pos.volume
                px  = tick.bid if pos.type==mt5.POSITION_TYPE_BUY else tick.ask
                res = close_partial(pos, vol, price=px)
                if debug: print(f"[MGR] → Time exit close {vol} ret={getattr(res,'retcode',None)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-config", type=str, default=str(DEFAULT_LIVE), help="مسار live.yaml")
    ap.add_argument("--interval", type=int, default=2, help="ثواني بين كل دورة إدارة")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    live_path = Path(args.live_config).resolve()
    if not live_path.exists():
        print(f"❌ live config غير موجود: {live_path}"); sys.exit(1)

    cfg_live = load_yaml(live_path)
    cfg_ens  = load_ensemble_cfg(CONFIG_ENSEMBLE)

    symbol = cfg_live["trade"]["symbol"]

    if not mt5.initialize():
        print("❌ mt5.initialize failed:", mt5.last_error()); sys.exit(1)
    si = ensure_symbol_visible(symbol)
    if not si:
        print(f"❌ Symbol not visible: {symbol}"); sys.exit(1)

    print(f"▶️ Position Manager for {symbol} | {live_path}")
    try:
        while True:
            if not within_hours(cfg_live):
                time.sleep(5); continue
            if not account_ok():
                time.sleep(3); continue
            try:
                manage_positions(cfg_live, cfg_ens, si, debug=args.debug)
            except Exception as e:
                if args.debug: print("⚠️ manage error:", e)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
