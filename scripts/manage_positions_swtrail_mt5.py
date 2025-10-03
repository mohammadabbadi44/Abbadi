# -*- coding: utf-8 -*-
"""
scripts/manage_positions_swtrail_mt5.py

مدير صفقات ذكي لِـ MT5:
- Partial TP على مستويات R:R
- Breakeven عند R:R معيّن
- Trailing بثلاث أنماط:
  1) atr: trailing باستخدام ATR
  2) swing: خلف آخر قاع/قمة مؤكدة (Swing Low/High)
  3) step: سلّمي حسب حركة السعر بالنقاط
  4) hybrid: يختار أقوى حماية (الأقرب من السعر لصالحك) بين (atr/swing/step)

الإعدادات تُقرأ من live.yaml تحت المفتاح manager: (انظر المثال أدناه).

تشغيل:
  python -m scripts.manage_positions_swtrail_mt5 --live-config config\live_m1.yaml --debug
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
    print("❌ Missing dependency:", e); sys.exit(1)

# نعيد استخدام أدوات الداتا/الفيتشرز من مشروعك
from scripts.run_hybrid_ensemble_ai import (
    fetch_data, build_features, ensure_symbol_visible, load_cfg as load_ensemble_cfg
)

CONFIG_ENSEMBLE = ROOT / "config" / "ensemble_ai.yaml"
DEFAULT_LIVE    = ROOT / "config" / "live.yaml"

# --------------- Utils ---------------
def load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def within_hours(cfg_live):
    th = cfg_live.get("filters", {}).get("trading_hours", {})
    if not th.get("enabled", False):
        return True
    from datetime import datetime as _dt
    now = datetime.now().time()
    st = _dt.strptime(th.get("start","00:00"), "%H:%M").time()
    en = _dt.strptime(th.get("end","23:59"), "%H:%M").time()
    return (st <= now <= en)

def symbol_positions(symbol: str, magic: int):
    res = []
    for p in (mt5.positions_get(symbol=symbol) or []):
        if p.magic == magic:
            res.append(p)
    return res

def point_info(si):
    return si.point or 0.01, getattr(si, "trade_tick_value", 1.0) or 1.0, getattr(si, "trade_tick_size", None) or (si.point or 0.01)

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
    if atr_val is None or atr_val <= 0:
        return None
    # مرجع SL الحالي إن وجد، وإلا ATR*3 تقريباً
    stop_dist = abs((position.price_open - position.sl)) if position.sl else atr_val*3.0
    if stop_dist <= 1e-9:
        stop_dist = atr_val*3.0
    tick = mt5.symbol_info_tick(position.symbol)
    if not tick: return None
    price = tick.bid if position.type == mt5.POSITION_TYPE_SELL else tick.ask
    move_dist = (price - position.price_open) if position.type == mt5.POSITION_TYPE_BUY else (position.price_open - price)
    return max(0.0, move_dist / stop_dist)

# --------------- Swing detection ---------------
def find_last_swing(df: pd.DataFrame, kind: str, left: int = 2, right: int = 2):
    """
    يبحث عن آخر swing مؤكد:
    - kind='low' → قاع محلي: low[i] أقل من [i-left:i+right]
    - kind='high'→ قمة محلية: high[i] أعلى من [i-left:i+right]
    نستخدم آخر شمعة مكتملة (i = len(df)-2) كنقطة بداية بحث رجوعاً.
    """
    n = len(df)
    if n < (left + right + 5):
        return None, None
    end = n - 2  # آخر شمعة مكتملة
    lows = df["low"].values
    highs= df["high"].values
    for i in range(end - right, left, -1):
        l = i - left
        r = i + right
        if l < 0 or r >= n: 
            continue
        if kind == "low":
            if lows[i] == np.min(lows[l:r+1]):
                return i, lows[i]
        else:
            if highs[i] == np.max(highs[l:r+1]):
                return i, highs[i]
    return None, None

# --------------- Step trailing helper ---------------
def step_trail_level(position, step_points: float, offset_points: float, si):
    """
    يعيد مستوى SL المقترح بالسلّم:
    - BUY: كل ما السعر يرتفع step_points، نرفع SL بمقدار step_points*offset_ratio تقريباً (نستخدم offset_points مباشرة).
    - SELL: بالعكس.
    """
    tick = mt5.symbol_info_tick(position.symbol)
    if not tick: return None
    pt, _, step_min = point_info(si)
    if position.type == mt5.POSITION_TYPE_BUY:
        price = tick.bid
        # كم سلّم قطعناه من سعر الدخول؟
        ladders = math.floor((price - position.price_open) / (step_points * pt))
        if ladders <= 0:
            return None
        new_sl = position.price_open + ladders * (offset_points * pt)
        # لا تنزل SL عن الحالي
        return max(position.sl or -1e9, new_sl + step_min)
    else:
        price = tick.ask
        ladders = math.floor((position.price_open - price) / (step_points * pt))
        if ladders <= 0:
            return None
        new_sl = position.price_open - ladders * (offset_points * pt)
        return min(position.sl or 1e9, new_sl - step_min)

# --------------- Core Manager ---------------
def manage_positions(cfg_live, cfg_ens, si, debug=False):
    symbol = cfg_live["trade"]["symbol"]
    magic  = int(cfg_live["trade"]["magic"])
    timeframe = cfg_live["trade"]["timeframe"]
    days = max(5, cfg_ens.get("days", 90))

    mgr = cfg_live.get("manager", {})
    # Partial / BE
    rr_levels       = mgr.get("partial_rr_levels",  [0.8, 1.2, 1.8])
    rr_closes       = mgr.get("partial_close_pcts", [0.25,0.35,0.40])
    move_be_at_rr   = float(mgr.get("move_to_be_at_rr", 1.0))

    # Trailing mode & params
    mode            = str(mgr.get("trailing_mode", "hybrid")).lower()  # atr/swing/step/hybrid
    # ATR
    trail_atr_mult  = float(mgr.get("trail_atr_mult", 1.0))
    # Swing
    swing_left      = int(mgr.get("swing_left", 2))
    swing_right     = int(mgr.get("swing_right", 2))
    swing_buffer_pts= float(mgr.get("swing_buffer_points", 20))  # مسافة أمان فوق/تحت السوينغ بالنقاط
    # Step
    step_points     = float(mgr.get("step_points", 50))          # كل كم نقطة نتحرك درجة
    step_offset_pts = float(mgr.get("step_offset_points", 30))   # كم نقطة ننقل SL لكل درجة
    # Common/limits
    trail_start_rr  = float(mgr.get("trail_start_rr", 1.2))      # ابدأ التريل بعد وصول RR معين
    max_hold_minutes= int(mgr.get("max_hold_minutes", 0))        # 0 = تعطيل

    pt, tick_value, step_min = point_info(si)

    # داتا حديثة لحساب ATR والسوينغ
    try:
        data = fetch_data(symbol, timeframe, days)
        df = data[0] if isinstance(data, tuple) else data
        feats = build_features(df)
        atr_val = float(feats["atr14"].iloc[-2])
    except Exception as e:
        if debug: print("⚠️ data/features fetch failed:", e)
        df, feats, atr_val = None, None, None

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return

    positions = symbol_positions(symbol, magic)
    if not positions:
        return

    for pos in positions:
        # ---------- BE / Partial ----------
        cur_rr = rr_now(pos, atr_val)
        if debug:
            print(f"[MGR] ticket={pos.ticket} type={'BUY' if pos.type==0 else 'SELL'} vol={pos.volume:.2f} rr={cur_rr}")

        # Move to BE
        if cur_rr is not None and pos.sl is not None and cur_rr >= move_be_at_rr:
            be = pos.price_open
            if (pos.type == mt5.POSITION_TYPE_BUY and (pos.sl is None or be > pos.sl + step_min)) or \
               (pos.type == mt5.POSITION_TYPE_SELL and (pos.sl is None or be < pos.sl - step_min)):
                res = modify_sl_tp(pos, sl=be, tp=None)
                if debug: print(f"[MGR] → BE SL={be:.2f} ret={getattr(res,'retcode',None)}")

        # Partial TP
        if cur_rr is not None and rr_levels and rr_closes and len(rr_levels)==len(rr_closes):
            try:
                initial_vol = float(getattr(pos, "volume_initial", pos.volume))
            except Exception:
                initial_vol = pos.volume
            for lvl, pct in zip(rr_levels, rr_closes):
                if cur_rr >= lvl and pos.volume > max(0.01, initial_vol * (1.0 - pct) + 1e-6):
                    vol_to_close = max(0.01, round(initial_vol * pct, 2))
                    vol_to_close = min(vol_to_close, pos.volume - 0.01)
                    if vol_to_close > 0:
                        px = tick.bid if pos.type==mt5.POSITION_TYPE_BUY else tick.ask
                        res = close_partial(pos, vol_to_close, price=px)
                        if debug: print(f"[MGR] → Partial {vol_to_close} @RR>={lvl} ret={getattr(res,'retcode',None)}")

        # ---------- Trailing ----------
        if cur_rr is None or cur_rr < trail_start_rr:
            continue  # لسه بدري

        proposed_sls = []

        # ATR trailing
        if atr_val and mode in ("atr", "hybrid"):
            dist = max(step_min, trail_atr_mult * atr_val)
            if pos.type == mt5.POSITION_TYPE_BUY:
                new_sl = (tick.bid - dist)
                if pos.sl is None or new_sl > pos.sl + step_min:
                    proposed_sls.append(new_sl)
            else:
                new_sl = (tick.ask + dist)
                if pos.sl is None or new_sl < pos.sl - step_min:
                    proposed_sls.append(new_sl)

        # Swing trailing
        if df is not None and mode in ("swing", "hybrid"):
            if pos.type == mt5.POSITION_TYPE_BUY:
                i_s, swing_low = find_last_swing(df, kind="low", left=swing_left, right=swing_right)
                if swing_low is not None:
                    new_sl = swing_low - swing_buffer_pts * pt
                    if pos.sl is None or new_sl > pos.sl + step_min:
                        proposed_sls.append(new_sl)
            else:
                i_s, swing_high = find_last_swing(df, kind="high", left=swing_left, right=swing_right)
                if swing_high is not None:
                    new_sl = swing_high + swing_buffer_pts * pt
                    if pos.sl is None or new_sl < pos.sl - step_min:
                        proposed_sls.append(new_sl)

        # Step trailing
        if mode in ("step", "hybrid"):
            stp = step_trail_level(pos, step_points=step_points, offset_points=step_offset_pts, si=si)
            if stp is not None:
                proposed_sls.append(stp)

        if proposed_sls:
            # hybrid: نختار أقرب SL لصالحنا (الأعلى للشراء/الأدنى للبيع)
            if pos.type == mt5.POSITION_TYPE_BUY:
                final_sl = max(proposed_sls)
                if pos.sl is None or final_sl > pos.sl + step_min:
                    res = modify_sl_tp(pos, sl=final_sl, tp=None)
                    if debug: print(f"[MGR] → Trail BUY SL={final_sl:.2f} (mode={mode}) ret={getattr(res,'retcode',None)}")
            else:
                final_sl = min(proposed_sls)
                if pos.sl is None or final_sl < pos.sl - step_min:
                    res = modify_sl_tp(pos, sl=final_sl, tp=None)
                    if debug: print(f"[MGR] → Trail SELL SL={final_sl:.2f} (mode={mode}) ret={getattr(res,'retcode',None)}")

        # Max hold time (اختياري)
        if max_hold_minutes > 0:
            open_time = datetime.fromtimestamp(pos.time)
            if datetime.now() - open_time >= timedelta(minutes=max_hold_minutes):
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

    print(f"▶️ Swing/Step Manager for {symbol} | {live_path}")
    try:
        while True:
            if within_hours(cfg_live) and mt5.account_info() is not None:
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
