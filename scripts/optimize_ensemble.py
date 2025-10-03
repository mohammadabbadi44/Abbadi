# -*- coding: utf-8 -*-
"""
scripts/optimize_ensemble.py

Optimizer للـ Ensemble AI:
- يستورد الدوال من scripts/run_hybrid_ensemble_ai.py
- يسحب الداتا/الميزات مرة واحدة وبعدين يجرب عينات بارامترات كثيييرة
- الهدف: PF أولًا (فلتر قاسي تحت 1.0)، سقف DD، حد أدنى صفقات
- يحفظ تقرير CSV + أفضل إعدادات كـ "YAML-مكتوب-JSON" في reports/

تشغيل مقترح:
    python -m scripts.optimize_ensemble --trials 120 --min-trades 200 --dd-cap 25 --seed 42
"""

import os, sys, json, random, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.run_hybrid_ensemble_ai import (
    load_cfg, fetch_data, build_features, backtest
)

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ MetaTrader5 غير مثبت. ثبّت: pip install MetaTrader5")
    sys.exit(1)

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- فضاء البحث -----------------
def sample_cfg(base: dict) -> dict:
    """كوّن نسخة إعدادات مع عيّنة عشوائية من البارامترات."""
    cfg = json.loads(json.dumps(base))  # deep copy

    # ساعات التداول
    th_enabled = random.choice([True, False])
    if th_enabled:
        th_start = random.choice(["06:00","07:00"])
        th_end   = random.choice(["19:30","20:30","21:00"])
    else:
        th_start, th_end = "00:00", "23:59"

    cfg.setdefault("trading_hours", {})
    cfg["trading_hours"]["enabled"] = th_enabled
    cfg["trading_hours"]["start"]   = th_start
    cfg["trading_hours"]["end"]     = th_end

    # إدارة الصفقة
    cfg["sl_atr_mult"]     = round(random.uniform(3.0, 3.8), 2)
    cfg["tp_atr_mult"]     = round(random.uniform(3.8, 5.2), 2)
    cfg["breakeven_at_rr"] = round(random.uniform(0.9, 1.2), 2)
    cfg["use_trailing"]    = True
    cfg["trail_atr_mult"]  = round(random.uniform(0.8, 1.4), 2)
    cfg["max_hold_bars"]   = random.choice([8, 10, 12, 14, 16])

    # فلاتر/حراس
    cfg.setdefault("guards", cfg.get("guards", {}))
    cfg["guards"]["min_atr_pct"]       = round(random.uniform(0.0006, 0.0012), 6)
    cfg["guards"]["max_atr_pct"]       = round(random.uniform(0.025, 0.04), 3)
    cfg["guards"]["max_spread_vs_atr"] = round(random.uniform(0.20, 0.32), 2)
    cfg["guards"]["max_trades_per_day"]= random.choice([80, 100, 120])

    # Ensemble gates
    cfg["ensemble_mode"]      = random.choice(["weighted_mean", "majority_vote"])
    cfg["min_ensemble_votes"] = random.choice([2, 3])
    cfg["min_ensemble_conf"]  = round(random.uniform(0.60, 0.80), 2)
    cfg["use_ema200_trend_filter"] = random.choice([True, True, False])  # أغلب الوقت مفعّل

    # أوزان الاستراتيجيات
    cfg.setdefault("strategies", cfg.get("strategies", {}))
    for name, default in {
        "trend_following":  (1.1, 1.6),
        "momentum_macd":    (1.0, 1.4),
        "breakout_donchian":(0.8, 1.2),
        "volatility_ttm":   (0.7, 1.1),
        "volume_obv":       (0.6, 1.0),
        "mean_reversion":   (0.4, 0.8),
    }.items():
        if cfg["strategies"].get(name, {}).get("enabled", True):
            low, high = default
            cfg["strategies"][name]["weight"] = round(random.uniform(low, high), 2)
            cfg["strategies"][name]["enabled"] = True

    # مخاطرة
    cfg.setdefault("risk", cfg.get("risk", {}))
    cfg["risk"]["risk_per_trade_pct"]     = round(random.uniform(0.15, 0.35), 2)
    cfg["risk"]["max_daily_loss_pct"]     = random.choice([2.0, 2.5])
    cfg["risk"]["max_total_drawdown_pct"] = random.choice([25.0, 30.0, 35.0])

    # AI gate
    cfg.setdefault("ai", cfg.get("ai", {}))
    cfg["ai"]["enabled"] = True
    cfg["ai"]["decision_threshold"] = round(random.uniform(0.60, 0.72), 2)
    # train_if_missing يبقى حسب ملفك (الغالب True)

    return cfg

# ----------------- الهدف -----------------
def objective(stats: dict, min_trades=200, dd_cap=25.0) -> float:
    """نرجّح PF أولًا ونرفض تحت 1.0. نعاقب DD العالي وovertrading."""
    if not stats or stats.get("trades", 0) < min_trades:
        return -1e9

    pf = stats.get("profit_factor")
    if isinstance(pf, str):
        pf = 999.0 if pf.lower() == "inf" else float(pf)
    pf = float(pf)

    dd = float(stats.get("max_dd", 0.0))
    trades = int(stats.get("trades", 0))
    win = float(stats.get("win_rate", 0.0))
    ret = float(stats.get("return_pct", 0.0))

    # فلتر قاسي
    if pf < 1.0:
        return -1e6 + (pf - 1.0)

    score = pf
    # عقوبة DD
    if dd > dd_cap:
        score -= 0.08 * (dd - dd_cap)
    # عقوبة overtrading
    if trades > 1200:
        score -= 0.0006 * (trades - 1200)
    # مكافآت خفيفة
    score += 0.002 * max(0.0, win - 45.0)
    score += 0.0004 * ret
    return float(score)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--min-trades", type=int, default=200)
    ap.add_argument("--dd-cap", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg_path = ROOT / "config" / "ensemble_ai.yaml"
    base_cfg = load_cfg(cfg_path)

    if not mt5.initialize():
        print("❌ mt5.initialize failed:", mt5.last_error())
        sys.exit(1)

    try:
        # سحب بيانات/ميزات مرة واحدة
        df, si = fetch_data(base_cfg["symbol"], base_cfg["timeframe"], base_cfg["days"])
        feats   = build_features(df)

        results = []
        best = None
        best_score = -1e18

        for t in range(1, args.trials+1):
            cfg_try = sample_cfg(base_cfg)
            stats, _ = backtest(df, feats, cfg_try, si)
            if stats is None:
                cur = -1e9
                row = {"trial": t, "score": cur, "note": "no_trades"}
            else:
                cur = objective(stats, min_trades=args.min_trades, dd_cap=args.dd_cap)
                row = {"trial": t, "score": round(cur, 6)}
                row.update(stats)

                # سجّل أهم البارامترات
                row.update({
                    "ensemble_mode": cfg_try["ensemble_mode"],
                    "min_votes": cfg_try["min_ensemble_votes"],
                    "min_conf": cfg_try["min_ensemble_conf"],
                    "sl_atr_mult": cfg_try["sl_atr_mult"],
                    "tp_atr_mult": cfg_try["tp_atr_mult"],
                    "trail_atr_mult": cfg_try["trail_atr_mult"],
                    "breakeven_at_rr": cfg_try["breakeven_at_rr"],
                    "max_hold_bars": cfg_try["max_hold_bars"],
                    "min_atr_pct": cfg_try["guards"]["min_atr_pct"],
                    "max_atr_pct": cfg_try["guards"]["max_atr_pct"],
                    "max_spread_vs_atr": cfg_try["guards"]["max_spread_vs_atr"],
                    "max_trades_per_day": cfg_try["guards"]["max_trades_per_day"],
                    "risk_per_trade_pct": cfg_try["risk"]["risk_per_trade_pct"],
                    "max_daily_loss_pct": cfg_try["risk"]["max_daily_loss_pct"],
                    "max_total_drawdown_pct": cfg_try["risk"]["max_total_drawdown_pct"],
                    "ai_threshold": cfg_try["ai"]["decision_threshold"],
                    "th_enabled": cfg_try["trading_hours"]["enabled"],
                    "th_start": cfg_try["trading_hours"]["start"],
                    "th_end": cfg_try["trading_hours"]["end"],
                    # أوزان الاستراتيجيات
                    "w_trend": cfg_try["strategies"]["trend_following"]["weight"],
                    "w_mom": cfg_try["strategies"]["momentum_macd"]["weight"],
                    "w_brk": cfg_try["strategies"]["breakout_donchian"]["weight"],
                    "w_vol": cfg_try["strategies"]["volatility_ttm"]["weight"],
                    "w_obv": cfg_try["strategies"]["volume_obv"]["weight"],
                    "w_meanrev": cfg_try["strategies"]["mean_reversion"]["weight"],
                })

            results.append(row)

            if cur > best_score:
                best_score = cur
                best = (cfg_try, stats)
                pf_show = stats.get('profit_factor') if stats else 'NA'
                dd_show = stats.get('max_dd') if stats else 'NA'
                tr_show = stats.get('trades') if stats else 0
                print(f"[{t}/{args.trials}] ⭐ NEW BEST score={best_score:.4f} | PF={pf_show} | DD={dd_show} | Trades={tr_show}")

        # حفظ التقرير
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = REPORTS_DIR / f"ensemble_opt_{ts}.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"\n📄 Saved optimization report: {out_csv}")

        # أفضل إعدادات إلى ملف "yaml-json"
        if best and best[1] is not None:
            best_cfg, best_stats = best
            best_yaml = {
                "symbol": base_cfg["symbol"],
                "timeframe": base_cfg["timeframe"],
                "days": base_cfg["days"],
                "spread_points": base_cfg["spread_points"],
                "slippage_points": base_cfg["slippage_points"],
                "commission_per_lot_usd": base_cfg["commission_per_lot_usd"],
                # من الأوبتمايزر
                "sl_atr_mult": best_cfg["sl_atr_mult"],
                "tp_atr_mult": best_cfg["tp_atr_mult"],
                "breakeven_at_rr": best_cfg["breakeven_at_rr"],
                "use_trailing": best_cfg["use_trailing"],
                "trail_atr_mult": best_cfg["trail_atr_mult"],
                "max_hold_bars": best_cfg["max_hold_bars"],
                "use_ema200_trend_filter": best_cfg["use_ema200_trend_filter"],
                "min_ensemble_votes": best_cfg["min_ensemble_votes"],
                "min_ensemble_conf": best_cfg["min_ensemble_conf"],
                "cooldown_bars": base_cfg.get("cooldown_bars", 4),
                "one_position": base_cfg.get("one_position", True),
                "ensemble_mode": best_cfg["ensemble_mode"],
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
                "ai": {
                    "enabled": True,
                    "model_path": base_cfg.get("ai", {}).get("model_path", "models/ensemble_lr.joblib"),
                    "decision_threshold": best_cfg["ai"]["decision_threshold"],
                    "train_if_missing": base_cfg.get("ai", {}).get("train_if_missing", True),
                },
                "strategies": best_cfg["strategies"],
                "_best_stats": best_stats
            }
            out_yaml = REPORTS_DIR / f"ensemble_opt_best_{ts}.yaml"
            with open(out_yaml, "w", encoding="utf-8") as f:
                f.write(json.dumps(best_yaml, ensure_ascii=False, indent=2))
            print(f"🏁 Best config saved: {out_yaml}")
            print(f"👉 Best stats: PF={best_stats['profit_factor']}, Win%={best_stats['win_rate']}, DD={best_stats['max_dd']}%, Trades={best_stats['trades']}, Return={best_stats['return_pct']}%")

    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
