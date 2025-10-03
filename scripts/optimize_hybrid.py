# -*- coding: utf-8 -*-
"""
scripts/optimize_hybrid.py

Random Search لتحسين إعدادات استراتيجية hybrid على بيانات MT5.
- يستورد الباكتيست من scripts/run_hybrid_backtest.py
- يسحب البيانات مرة واحدة، ويعيد تشغيل الباكتيست على عينات بارامترات كثيرة
- الهدف: PF أولًا مع عقوبة على MaxDD وحد أدنى للتداولات + عقوبة overtrading
- يحفظ أفضل النتائج في CSV و "YAML-مكتوب-JSON"

تشغيل مقترح:
    python -m scripts.optimize_hybrid --trials 100 --min-trades 200 --dd-cap 35
"""

import os, sys, math, json, random, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.run_hybrid_backtest import (
    load_cfg, ensure_symbol_visible, fetch_data, build_features, backtest
)

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ MetaTrader5 غير مثبت. ثبّت: pip install MetaTrader5")
    sys.exit(1)

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def sample_space(base_cfg):
    """
    عينة من فضاء بحث مُحافظ يهدف لرفع PF وتقليل DD/overtrading.
    """
    # جلسات
    th_enabled = random.choice([True, False])
    if th_enabled:
        th_start = random.choice(["06:00","07:00"])
        th_end   = random.choice(["19:30","20:30","21:00"])
    else:
        th_start, th_end = "00:00", "23:59"

    # بارامترات رئيسية
    min_signals     = random.randint(4, 7)
    sl_atr_mult     = round(random.uniform(3.0, 3.8), 2)
    tp_atr_mult     = round(random.uniform(3.4, 5.0), 2)
    cooldown_bars   = random.randint(3, 8)

    # إدارة الصفقة
    use_trailing    = True
    trail_atr_mult  = round(random.uniform(0.8, 1.4), 2)
    breakeven_at_rr = round(random.uniform(0.9, 1.2), 2)

    # حراس الجودة
    min_atr_pct     = round(random.uniform(0.0005, 0.0012), 6)
    max_atr_pct     = round(random.uniform(0.025, 0.04), 3)
    max_spread_vs_atr = round(random.uniform(0.20, 0.35), 2)
    max_trades_per_day = random.choice([90, 110, 130])

    # مخاطرة
    risk_per_trade_pct = round(random.uniform(0.15, 0.35), 2)
    max_daily_loss_pct = random.choice([2.0, 2.5])
    max_total_drawdown_pct = random.choice([30.0, 35.0])

    # احتفاظ الصفقة
    max_hold_bars = random.choice([8, 10, 12, 16, 20])

    # AI (إن مفعّل)
    ai_enabled = base_cfg.get("ai", {}).get("enabled", False)
    if ai_enabled:
        decision_threshold = round(random.uniform(0.55, 0.80), 2)
    else:
        decision_threshold = base_cfg.get("ai", {}).get("decision_threshold", 0.65)

    # كوّن نسخة cfg معدّلة
    cfg = json.loads(json.dumps(base_cfg))
    cfg["min_signals"] = min_signals
    cfg["sl_atr_mult"] = sl_atr_mult
    cfg["tp_atr_mult"] = tp_atr_mult
    cfg["cooldown_bars"] = cooldown_bars

    cfg["use_trailing"] = use_trailing
    cfg["trail_atr_mult"] = trail_atr_mult
    cfg["breakeven_at_rr"] = breakeven_at_rr

    cfg.setdefault("trading_hours", {})
    cfg["trading_hours"]["enabled"] = th_enabled
    cfg["trading_hours"]["start"] = th_start
    cfg["trading_hours"]["end"] = th_end

    cfg.setdefault("guards", {})
    cfg["guards"]["min_atr_pct"] = min_atr_pct
    cfg["guards"]["max_atr_pct"] = max_atr_pct
    cfg["guards"]["max_spread_vs_atr"] = max_spread_vs_atr
    cfg["guards"]["max_trades_per_day"] = max_trades_per_day

    cfg.setdefault("risk", cfg.get("risk", {}))
    cfg["risk"]["risk_per_trade_pct"] = risk_per_trade_pct
    cfg["risk"]["max_daily_loss_pct"] = max_daily_loss_pct
    cfg["risk"]["max_total_drawdown_pct"] = max_total_drawdown_pct

    cfg.setdefault("ai", cfg.get("ai", {}))
    cfg["ai"]["decision_threshold"] = decision_threshold

    cfg["max_hold_bars"] = max_hold_bars

    return cfg

def objective(stats, min_trades=200, dd_cap=35.0):
    """
    PF أولًا، مع عقوبات على DD العالي وovertrading، وفلتر قاسي تحت 0.9 PF.
    """
    if (stats is None) or (stats.get("trades", 0) < min_trades):
        return -1e9

    pf = stats.get("profit_factor")
    pf = (999.0 if isinstance(pf, str) and str(pf).lower()=="inf" else float(pf))
    dd = float(stats.get("max_dd", 0.0))
    trades = int(stats.get("trades", 0))
    win = float(stats.get("win_rate", 0.0))

    if pf < 0.9:
        return -1e6 + (pf - 1.0)

    score = pf
    if dd > dd_cap:
        score -= 0.07 * (dd - dd_cap)

    if trades > 1200:
        score -= 0.0005 * (trades - 1200)

    score += 0.002 * max(0.0, win - 42.0)
    return float(score)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100, help="عدد العينات العشوائية")
    parser.add_argument("--min-trades", type=int, default=200, help="حد أدنى لعدد الصفقات")
    parser.add_argument("--dd-cap", type=float, default=35.0, help="الحد المستهدف لأقصى سحب")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg_path = ROOT / "config" / "hybrid_ai.yaml"
    cfg_base = load_cfg(cfg_path)

    if not mt5.initialize():
        print("❌ mt5.initialize failed:", mt5.last_error())
        sys.exit(1)

    try:
        si = ensure_symbol_visible(cfg_base["symbol"])
        if not si:
            raise RuntimeError(f"Symbol not visible: {cfg_base['symbol']}")

        df = fetch_data(cfg_base["symbol"], cfg_base["timeframe"], cfg_base["days"])
        feats = build_features(df)

        results = []
        best = None
        best_score = -1e18

        for t in range(1, args.trials + 1):
            cfg_try = sample_space(cfg_base)
            stats, _ = backtest(df, feats, cfg_try, si)
            if stats is None:
                cur_score = -1e9
                row = {"trial": t, "score": cur_score, "note": "no_trades"}
            else:
                cur_score = objective(stats, min_trades=args.min_trades, dd_cap=args.dd_cap)
                row = {"trial": t, "score": round(cur_score, 6)}
                row.update(stats)
                row.update({
                    "min_signals": cfg_try["min_signals"],
                    "sl_atr_mult": cfg_try["sl_atr_mult"],
                    "tp_atr_mult": cfg_try["tp_atr_mult"],
                    "cooldown_bars": cfg_try["cooldown_bars"],
                    "use_trailing": cfg_try["use_trailing"],
                    "trail_atr_mult": cfg_try["trail_atr_mult"],
                    "breakeven_at_rr": cfg_try["breakeven_at_rr"],
                    "min_atr_pct": cfg_try["guards"]["min_atr_pct"],
                    "max_atr_pct": cfg_try["guards"]["max_atr_pct"],
                    "max_spread_vs_atr": cfg_try["guards"]["max_spread_vs_atr"],
                    "max_trades_per_day": cfg_try["guards"]["max_trades_per_day"],
                    "risk_per_trade_pct": cfg_try["risk"]["risk_per_trade_pct"],
                    "max_daily_loss_pct": cfg_try["risk"]["max_daily_loss_pct"],
                    "max_total_drawdown_pct": cfg_try["risk"]["max_total_drawdown_pct"],
                    "th_start": cfg_try["trading_hours"]["start"],
                    "th_end": cfg_try["trading_hours"]["end"],
                    "max_hold_bars": cfg_try["max_hold_bars"]
                })

            results.append(row)

            if cur_score > best_score:
                best_score = cur_score
                best = (cfg_try, stats)
                print(f"[{t}/{args.trials}] ⭐ NEW BEST score={best_score:.4f} | PF={stats.get('profit_factor') if stats else 'NA'} | DD={stats.get('max_dd') if stats else 'NA'} | Trades={stats.get('trades') if stats else 0}")

        dfres = pd.DataFrame(results)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = REPORTS_DIR / f"hybrid_ai_opt_{ts}.csv"
        dfres.to_csv(out_csv, index=False)
        print(f"\n📄 Saved optimization report: {out_csv}")

        if best and best[1] is not None:
            best_cfg, best_stats = best
            best_yaml = {
                "symbol": cfg_base["symbol"],
                "timeframe": cfg_base["timeframe"],
                "days": cfg_base["days"],
                "spread_points": cfg_base["spread_points"],
                "slippage_points": cfg_base["slippage_points"],
                "commission_per_lot_usd": cfg_base["commission_per_lot_usd"],
                "use_ema200_trend_filter": cfg_base["use_ema200_trend_filter"],
                "min_signals": best_cfg["min_signals"],
                "sl_atr_mult": best_cfg["sl_atr_mult"],
                "tp_atr_mult": best_cfg["tp_atr_mult"],
                "cooldown_bars": best_cfg["cooldown_bars"],
                "one_position": cfg_base.get("one_position", True),
                "use_trailing": best_cfg["use_trailing"],
                "trail_atr_mult": best_cfg["trail_atr_mult"],
                "breakeven_at_rr": best_cfg["breakeven_at_rr"],
                "trading_hours": {
                    "enabled": best_cfg["trading_hours"]["enabled"],
                    "start": best_cfg["trading_hours"]["start"],
                    "end": best_cfg["trading_hours"]["end"],
                },
                "guards": {
                    "min_atr_pct": best_cfg["guards"]["min_atr_pct"],
                    "max_atr_pct": best_cfg["guards"]["max_atr_pct"],
                    "max_spread_vs_atr": best_cfg["guards"]["max_spread_vs_atr"],
                    "max_trades_per_day": best_cfg["guards"]["max_trades_per_day"],
                },
                "risk": {
                    "risk_per_trade_pct": best_cfg["risk"]["risk_per_trade_pct"],
                    "max_daily_loss_pct": best_cfg["risk"]["max_daily_loss_pct"],
                    "max_total_drawdown_pct": best_cfg["risk"]["max_total_drawdown_pct"],
                },
                "max_hold_bars": best_cfg["max_hold_bars"],
                "ai": {
                    "enabled": cfg_base.get("ai", {}).get("enabled", False),
                    "model_path": cfg_base.get("ai", {}).get("model_path", "models/hybrid_clf.joblib"),
                    "decision_threshold": best_cfg["ai"]["decision_threshold"],
                },
                "_best_stats": best_stats
            }
            out_yaml = REPORTS_DIR / f"hybrid_ai_opt_best_{ts}.yaml"
            with open(out_yaml, "w", encoding="utf-8") as f:
                f.write(json.dumps(best_yaml, ensure_ascii=False, indent=2))
            print(f"🏁 Best config saved: {out_yaml}")
            print(f"👉 Best stats: PF={best_stats['profit_factor']}, Win%={best_stats['win_rate']}, DD={best_stats['max_dd']}%, Trades={best_stats['trades']}, Return={best_stats['return_pct']}%")

    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
