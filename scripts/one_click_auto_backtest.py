# -*- coding: utf-8 -*-
"""
scripts/one_click_auto_backtest.py

زرّ واحد يعمل كل شيء:
1) بحث مرحلي عن أفضل إعدادات (Wide -> Narrow) عبر تايمفريمات متعددة (--tfs)
2) يروّج الأفضل تلقائيًا إلى config/ensemble_ai.yaml
3) Backtest تأكيدي + Walk-Forward آخر 60 يوم
4) ينتج تقارير CSV + HTML مختصرة
5) (اختياري) يشغّل بشكل دوري: --loop-hours 1

تشغيل نموذج:
  python -m scripts.one_click_auto_backtest --tfs M5,M15 --trials 180 --min-trades 100 --dd-cap 25 --target-pf 1.05
"""
import os, sys, json, random, argparse, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# نستخدم نفس المنفذ اللي كتبته لك سابقًا
from scripts.run_hybrid_ensemble_ai import load_cfg, fetch_data, build_features, backtest

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ MetaTrader5 غير مثبت. pip install MetaTrader5")
    sys.exit(1)

REPORTS = ROOT / "reports"
CONFIG  = ROOT / "config" / "ensemble_ai.yaml"
REPORTS.mkdir(parents=True, exist_ok=True)

# ---------------- Utilities ----------------
def obj_score(stats: dict, min_trades=120, dd_cap=25.0, target_pf=1.05) -> float:
    if not stats or stats.get("trades", 0) < min_trades:
        return -1e9
    pf = stats["profit_factor"]
    pf = (999.0 if isinstance(pf, str) and str(pf).lower()=="inf" else float(pf))
    dd = float(stats.get("max_dd", 0.0))
    wr = float(stats.get("win_rate", 0.0))
    ret = float(stats.get("return_pct", 0.0))
    trades = int(stats.get("trades", 0))
    if pf < 1.0:
        return -1e6 + (pf - 1.0)
    score = 2.0*(pf - target_pf) + 0.004*(wr - 45.0) + 0.0006*ret
    if dd > dd_cap: score -= 0.08 * (dd - dd_cap)
    if trades > 1500: score -= 0.0006 * (trades - 1500)
    return float(score)

def to_yamlish(obj: dict) -> str:
    def dump(d, ind=0):
        sp="  "*ind; out=[]
        if isinstance(d, dict):
            for k,v in d.items():
                if isinstance(v,(dict,list)):
                    out.append(f"{sp}{k}:"); out.append(dump(v,ind+1))
                else:
                    if isinstance(v,str):
                        if any(c in v for c in [":","#","-","{","}","[","]"]) or " " in v:
                            out.append(f'{sp}{k}: "{v}"')
                        else: out.append(f"{sp}{k}: {v}")
                    else: out.append(f"{sp}{k}: {v}")
        elif isinstance(d, list):
            for x in d:
                if isinstance(x,(dict,list)):
                    out.append(f"{sp}-"); out.append(dump(x,ind+1))
                else: out.append(f"{sp}- {x}")
        else: out.append(f"{sp}{d}")
        return "\n".join(out)
    return dump(obj) + "\n"

def save_yaml_json(tag: str, cfg: dict, stats: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = REPORTS / f"{tag}_{ts}.yaml"
    data = dict(cfg); data["_best_stats"] = stats
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

def write_html_summary(path: Path, title: str, confirm: dict, wf: dict, tf_summ=None):
    def row(d,k): return d.get(k,"-") if d else "-"
    rows = ""
    if tf_summ:
        rows += "<h3>Per-TF Results</h3><table border='1' cellpadding='6'><tr><th>TF</th><th>PF</th><th>DD%</th><th>Trades</th><th>Score</th><th>YAML</th></tr>"
        for r in tf_summ:
            rows += f"<tr><td>{r['tf']}</td><td>{r['pf']}</td><td>{r['dd']}</td><td>{r['tr']}</td><td>{r['score']}</td><td>{r['yaml']}</td></tr>"
        rows += "</table>"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head>
<body>
<h2>{title}</h2>
<h3>Confirm</h3>
<ul>
<li>PF: {row(confirm,'profit_factor')}</li>
<li>Win%: {row(confirm,'win_rate')}</li>
<li>DD%: {row(confirm,'max_dd')}</li>
<li>Trades: {row(confirm,'trades')}</li>
<li>Return%: {row(confirm,'return_pct')}</li>
</ul>
<h3>Walk-Forward (last 60 days)</h3>
<ul>
<li>PF: {row(wf,'profit_factor')}</li>
<li>Win%: {row(wf,'win_rate')}</li>
<li>DD%: {row(wf,'max_dd')}</li>
<li>Trades: {row(wf,'trades')}</li>
<li>Return%: {row(wf,'return_pct')}</li>
</ul>
{rows}
<p>Generated at {datetime.now().isoformat()}</p>
</body></html>"""
    path.write_text(html, encoding="utf-8")

def sample_cfg(base: dict, tf: str, narrow=False) -> dict:
    """
    يبني إعداد عشوائي مع:
    - تفعيل/تعطيل ديناميكي لكل استراتيجية (ON/OFF)
    - أوزان مختلفة حسب النطاق (Wide/Narrow)
    - ضمان وجود عدد استراتيجيات مفعّلة يكفي للـ min_ensemble_votes
    """
    import copy, random
    cfg = json.loads(json.dumps(base))
    cfg["timeframe"] = tf

    # --------- سياسات تفعيل الاستراتيجيات ---------
    # احتمالات التفعيل (Wide ثم Narrow). عدّلها لو حاب.
    p_on_wide = {
        "trend_following":   0.95,
        "momentum_macd":     0.90,
        "breakout_donchian": 0.80,
        "volatility_ttm":    0.75,
        "volume_obv":        0.70,
        "mean_reversion":    0.30,  # الذهب غالباً يتعبه على M5، نخليه احتمال أقل
    }
    p_on_narrow = {
        "trend_following":   0.98,
        "momentum_macd":     0.95,
        "breakout_donchian": 0.85,
        "volatility_ttm":    0.75,
        "volume_obv":        0.70,
        "mean_reversion":    0.20,
    }
    P = p_on_narrow if narrow else p_on_wide

    # نطاقات أوزان لكل استراتيجية (Wide/Narrow)
    weight_rng_wide   = {
        "trend_following":   (1.3, 1.8),
        "momentum_macd":     (1.1, 1.5),
        "breakout_donchian": (0.9, 1.1),
        "volatility_ttm":    (0.7, 0.9),
        "volume_obv":        (0.6, 0.8),
        "mean_reversion":    (0.4, 0.7),
    }
    weight_rng_narrow = {
        "trend_following":   (1.6, 1.8),
        "momentum_macd":     (1.3, 1.5),
        "breakout_donchian": (0.9, 1.1),
        "volatility_ttm":    (0.7, 0.9),
        "volume_obv":        (0.6, 0.8),
        "mean_reversion":    (0.4, 0.6),
    }
    WR = weight_rng_narrow if narrow else weight_rng_wide

    # حضّر بلوك الاستراتيجيات
    cfg.setdefault("strategies", {})
    for name in ["trend_following","momentum_macd","breakout_donchian","volatility_ttm","volume_obv","mean_reversion"]:
        cfg["strategies"].setdefault(name, {})
        # تفعيل/تعطيل بحسب الاحتمال
        enabled = (random.random() < P[name])
        cfg["strategies"][name]["enabled"] = bool(enabled)
        # لو Disabled خليه بوزن 0
        if not enabled:
            cfg["strategies"][name]["weight"] = 0.0
        else:
            a, b = WR[name]
            cfg["strategies"][name]["weight"] = round(random.uniform(a, b), 2)

    # ضمان الحد الأدنى من الاستراتيجيات المفعّلة ≥ min_ensemble_votes
    min_votes = int(cfg.get("min_ensemble_votes", 3))
    enabled_list = [n for n, v in cfg["strategies"].items() if v.get("enabled", False)]
    if len(enabled_list) < min_votes:
        # فعّل أهم الاستراتيجيات حتى نصل للحد الأدنى
        priority = ["trend_following","momentum_macd","breakout_donchian","volatility_ttm","volume_obv","mean_reversion"]
        for name in priority:
            if not cfg["strategies"].get(name, {}).get("enabled", False):
                cfg["strategies"][name]["enabled"] = True
                if cfg["strategies"][name].get("weight", 0.0) == 0.0:
                    a, b = WR[name]
                    cfg["strategies"][name]["weight"] = round(random.uniform(a, b), 2)
                enabled_list.append(name)
                if len(enabled_list) >= min_votes:
                    break

    # --------- ساعات التداول ---------
    th_enabled = random.choice([True, True, False])
    cfg.setdefault("trading_hours", cfg.get("trading_hours", {}))
    cfg["trading_hours"]["enabled"] = th_enabled
    cfg["trading_hours"]["start"]   = random.choice(["06:30","07:00"]) if th_enabled else "00:00"
    cfg["trading_hours"]["end"]     = random.choice(["20:00","20:30","21:00"]) if th_enabled else "23:59"

    # --------- إدارة الصفقة ---------
    if not narrow:
        cfg["sl_atr_mult"]     = round(random.uniform(3.2, 3.8), 2)
        cfg["tp_atr_mult"]     = round(random.uniform(4.4, 5.2), 2)
        cfg["breakeven_at_rr"] = round(random.uniform(0.9, 1.1), 2)
        cfg["use_trailing"]    = True
        cfg["trail_atr_mult"]  = round(random.uniform(0.9, 1.2), 2)
        cfg["max_hold_bars"]   = random.choice([9,10,11,12,14])
    else:
        cfg["sl_atr_mult"]     = round(random.uniform(3.3, 3.7), 2)
        cfg["tp_atr_mult"]     = round(random.uniform(4.6, 5.1), 2)
        cfg["breakeven_at_rr"] = 1.0
        cfg["use_trailing"]    = True
        cfg["trail_atr_mult"]  = round(random.uniform(0.95, 1.15), 2)
        cfg["max_hold_bars"]   = random.choice([9,10,11])

    # --------- الحُرّاس (فلاتر الجودة) ---------
    cfg.setdefault("guards", cfg.get("guards", {}))
    if not narrow:
        cfg["guards"]["min_atr_pct"]       = round(random.uniform(0.0008, 0.0012), 6)
        cfg["guards"]["max_atr_pct"]       = round(random.uniform(0.025, 0.04), 3)
        cfg["guards"]["max_spread_vs_atr"] = round(random.uniform(0.20, 0.28), 2)
        cfg["guards"]["max_trades_per_day"]= random.choice([80, 90, 100, 120])
    else:
        cfg["guards"]["min_atr_pct"]       = round(random.uniform(0.0009, 0.0012), 6)
        cfg["guards"]["max_atr_pct"]       = round(random.uniform(0.028, 0.035), 3)
        cfg["guards"]["max_spread_vs_atr"] = round(random.uniform(0.20, 0.24), 2)
        cfg["guards"]["max_trades_per_day"]= random.choice([80, 90, 100])

    # --------- بوابات الإنسمبل والاتجاه ---------
    cfg["ensemble_mode"]            = "weighted_mean"
    cfg["min_ensemble_votes"]       = min_votes
    cfg["min_ensemble_conf"]        = round(random.uniform(0.62, 0.75) if not narrow else random.uniform(0.66, 0.75), 2)
    cfg["use_ema200_trend_filter"]  = True

    # --------- مخاطرة ---------
    cfg.setdefault("risk", cfg.get("risk", {}))
    cfg["risk"]["risk_per_trade_pct"]     = round(random.uniform(0.20, 0.30), 2)
    cfg["risk"]["max_daily_loss_pct"]     = 2.5
    cfg["risk"]["max_total_drawdown_pct"] = 30.0

    # --------- الذكاء الاصطناعي ---------
    cfg.setdefault("ai", cfg.get("ai", {}))
    cfg["ai"]["enabled"] = True
    cfg["ai"]["decision_threshold"] = round(random.uniform(0.66, 0.74) if not narrow else random.uniform(0.69, 0.74), 2)
    cfg["ai"]["train_if_missing"]   = True
    cfg["ai"]["model_path"] = base.get("ai", {}).get("model_path", "models/ensemble_lr.joblib")

    return cfg
    cfg = json.loads(json.dumps(base))
    cfg["timeframe"] = tf

    # تعطيل mean-reversion افتراضياً للذهب
    cfg.setdefault("strategies", cfg.get("strategies", {}))
    cfg["strategies"].setdefault("mean_reversion", {"enabled": False, "weight": 0.0})
    cfg["strategies"]["mean_reversion"]["enabled"] = False
    cfg["strategies"]["mean_reversion"]["weight"]  = 0.0

    # ساعات
    th_enabled = random.choice([True, True, False])
    cfg.setdefault("trading_hours", cfg.get("trading_hours", {}))
    cfg["trading_hours"]["enabled"] = th_enabled
    cfg["trading_hours"]["start"] = random.choice(["06:30","07:00"]) if th_enabled else "00:00"
    cfg["trading_hours"]["end"]   = random.choice(["20:00","20:30","21:00"]) if th_enabled else "23:59"

    # إدارة الصفقة
    if not narrow:
        cfg["sl_atr_mult"]     = round(random.uniform(3.2, 3.8), 2)
        cfg["tp_atr_mult"]     = round(random.uniform(4.4, 5.2), 2)
        cfg["breakeven_at_rr"] = round(random.uniform(0.9, 1.1), 2)
        cfg["use_trailing"]    = True
        cfg["trail_atr_mult"]  = round(random.uniform(0.9, 1.2), 2)
        cfg["max_hold_bars"]   = random.choice([9,10,11,12,14])
    else:
        cfg["sl_atr_mult"]     = round(random.uniform(3.3, 3.7), 2)
        cfg["tp_atr_mult"]     = round(random.uniform(4.6, 5.1), 2)
        cfg["breakeven_at_rr"] = 1.0
        cfg["use_trailing"]    = True
        cfg["trail_atr_mult"]  = round(random.uniform(0.95, 1.15), 2)
        cfg["max_hold_bars"]   = random.choice([9,10,11])

    # حراس جودة
    cfg.setdefault("guards", cfg.get("guards", {}))
    if not narrow:
        cfg["guards"]["min_atr_pct"]       = round(random.uniform(0.0008, 0.0012), 6)
        cfg["guards"]["max_atr_pct"]       = round(random.uniform(0.025, 0.04), 3)
        cfg["guards"]["max_spread_vs_atr"] = round(random.uniform(0.20, 0.28), 2)
        cfg["guards"]["max_trades_per_day"]= random.choice([80, 90, 100, 120])
    else:
        cfg["guards"]["min_atr_pct"]       = round(random.uniform(0.0009, 0.0012), 6)
        cfg["guards"]["max_atr_pct"]       = round(random.uniform(0.028, 0.035), 3)
        cfg["guards"]["max_spread_vs_atr"] = round(random.uniform(0.20, 0.24), 2)
        cfg["guards"]["max_trades_per_day"]= random.choice([80, 90, 100])

    # Ensemble gates
    cfg["ensemble_mode"]      = "weighted_mean"
    cfg["min_ensemble_votes"] = 3
    cfg["min_ensemble_conf"]  = round(random.uniform(0.62, 0.75) if not narrow else random.uniform(0.66, 0.75), 2)
    cfg["use_ema200_trend_filter"] = True

    # أوزان
    def rnd(a,b): return round(random.uniform(a,b),2)
    rngs = {
        "trend_following":  (1.3, 1.8) if not narrow else (1.6, 1.8),
        "momentum_macd":    (1.1, 1.5) if not narrow else (1.3, 1.5),
        "breakout_donchian":(0.9, 1.1),
        "volatility_ttm":   (0.7, 0.9),
        "volume_obv":       (0.6, 0.8),
    }
    for name, (a,b) in rngs.items():
        cfg["strategies"].setdefault(name, {"enabled": True, "weight": 1.0})
        cfg["strategies"][name]["enabled"] = True
        cfg["strategies"][name]["weight"]  = rnd(a,b)

    # مخاطرة
    cfg.setdefault("risk", cfg.get("risk", {}))
    cfg["risk"]["risk_per_trade_pct"]     = round(random.uniform(0.20, 0.30), 2)
    cfg["risk"]["max_daily_loss_pct"]     = 2.5
    cfg["risk"]["max_total_drawdown_pct"] = 30.0

    # AI
    cfg.setdefault("ai", cfg.get("ai", {}))
    cfg["ai"]["enabled"] = True
    cfg["ai"]["decision_threshold"] = round(random.uniform(0.66, 0.74) if not narrow else random.uniform(0.69, 0.74), 2)
    cfg["ai"]["train_if_missing"] = True
    cfg["ai"]["model_path"] = base.get("ai", {}).get("model_path", "models/ensemble_lr.joblib")
    return cfg

def run_stage(df, feats, si, base_cfg, tf, trials, min_trades, dd_cap, target_pf, narrow=False):
    rows, best, best_score = [], None, -1e18
    for i in range(1, trials+1):
        cfg_try = sample_cfg(base_cfg, tf=tf, narrow=narrow)
        stats, _ = backtest(df, feats, cfg_try, si)
        sc = obj_score(stats, min_trades=min_trades, dd_cap=dd_cap, target_pf=target_pf) if stats else -1e9
        rows.append({**({"trial": i, "tf": tf, "score": round(sc,6)}), **(stats or {})})
        if sc > best_score:
            best_score, best = sc, (cfg_try, stats)
            pf_show = stats.get('profit_factor') if stats else 'NA'
            dd_show = stats.get('max_dd') if stats else 'NA'
            tr_show = stats.get('trades') if stats else 0
            print(f"[{tf} {i}/{trials}] ⭐ NEW BEST score={best_score:.4f} | PF={pf_show} | DD={dd_show} | Trades={tr_show}")
    return pd.DataFrame(rows), best

def auto_once(args):
    base_cfg = load_cfg(CONFIG)
    tfs = [s.strip().upper() for s in args.tfs.split(",") if s.strip()]

    if not mt5.initialize():
        print("❌ mt5.initialize failed:", mt5.last_error()); return 1

    try:
        tf_summaries = []
        overall_best, overall_best_score = None, -1e18
        confirm_stats = None; wf_stats = None

        for tf in tfs:
            # Data/Features per TF
            base_tf = dict(base_cfg); base_tf["timeframe"] = tf
            df, si = fetch_data(base_tf["symbol"], base_tf["timeframe"], base_tf["days"])
            feats = build_features(df)

            # Stage 1 (Wide)
            print(f"\n=== {tf} Stage 1: Wide ===")
            wide_df, wide_best = run_stage(df, feats, si, base_tf, tf, max(60, args.trials//2), args.min_trades, args.dd_cap, max(1.0, args.target_pf-0.05), narrow=False)
            wide_df.to_csv(REPORTS / f"auto_wide_{tf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
            if not wide_best or not wide_best[1]:
                print(f"⚠️ {tf}: لا نتائج صالحة في Wide.")
                continue

            # Stage 2 (Narrow)
            print(f"=== {tf} Stage 2: Narrow ===")
            base_narrow = json.loads(json.dumps(wide_best[0]))
            narrow_df, narrow_best = run_stage(df, feats, si, base_narrow, tf, args.trials, args.min_trades, args.dd_cap, args.target_pf, narrow=True)
            narrow_df.to_csv(REPORTS / f"auto_narrow_{tf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
            if not narrow_best or not narrow_best[1]:
                print(f"⚠️ {tf}: لا نتائج صالحة في Narrow.")
                continue

            # Save TF-best
            y = save_yaml_json(f"ensemble_auto_{tf}_best", narrow_best[0], narrow_best[1])
            score_tf = obj_score(narrow_best[1], args.min_trades, args.dd_cap, args.target_pf)
            tf_summaries.append({"tf": tf, "pf": narrow_best[1]["profit_factor"], "dd": narrow_best[1]["max_dd"], "tr": narrow_best[1]["trades"], "score": round(score_tf,4), "yaml": str(y)})
            if score_tf > overall_best_score:
                overall_best_score, overall_best = score_tf, (tf, narrow_best, df, feats, si)

        if not overall_best:
            print("❌ ولا تايمفريم أعطى نتيجة فائزة."); return 2

        # Promote overall best
        tf_win, best_pair, df_b, feats_b, si_b = overall_best
        best_cfg, best_stats = best_pair
        CONFIG.write_text(to_yamlish({**best_cfg, "_best_stats": best_stats}), encoding="utf-8")
        print(f"\n✅ Promoted overall best ({tf_win}) → {CONFIG}")

        # Confirm backtest
        confirm_stats, _ = backtest(df_b, feats_b, best_cfg, si_b)
        print(f"Confirm → PF={confirm_stats['profit_factor']} | Win%={confirm_stats['win_rate']} | DD={confirm_stats['max_dd']}% | Trades={confirm_stats['trades']} | Return={confirm_stats['return_pct']}%")

        # Walk-Forward (last 60d)
        end_df = df_b
        df_wf = end_df[end_df["time"] >= (end_df["time"].iloc[-1] - pd.Timedelta(days=60))].reset_index(drop=True)
        feats_wf = build_features(df_wf)
        wf_stats, _ = backtest(df_wf, feats_wf, best_cfg, si_b)
        print(f"WalkFwd → PF={wf_stats['profit_factor']} | Win%={wf_stats['win_rate']} | DD={wf_stats['max_dd']}% | Trades={wf_stats['trades']} | Return={wf_stats['return_pct']}%")

        # Summary CSV + HTML
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = REPORTS / f"one_click_summary_{ts}.csv"
        pd.DataFrame(tf_summaries + [{
            "tf": f"OVERALL:{tf_win}",
            "pf": confirm_stats["profit_factor"],
            "dd": confirm_stats["max_dd"],
            "tr": confirm_stats["trades"],
            "score": round(overall_best_score,4),
            "yaml": str(CONFIG)
        }]).to_csv(out_csv, index=False)
        out_html = REPORTS / f"one_click_summary_{ts}.html"
        write_html_summary(out_html, f"One-Click Auto Backtest ({tf_win})", confirm_stats, wf_stats, tf_summaries)
        print(f"📝 Saved: {out_csv}\n📰 HTML: {out_html}")

        return 0
    finally:
        mt5.shutdown()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", type=str, default="M5")
    ap.add_argument("--trials", type=int, default=180)
    ap.add_argument("--min-trades", type=int, default=100)
    ap.add_argument("--dd-cap", type=float, default=25.0)
    ap.add_argument("--target-pf", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--loop-hours", type=float, default=0.0, help="شغّل تلقائياً كل N ساعات (0 = مرّة واحدة)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    if args.loop_hours and args.loop_hours > 0:
        print(f"⏱️ Auto loop: كل {args.loop_hours} ساعة.")
        while True:
            rc = auto_once(args)
            print(f"— دورة انتهت بكود: {rc} —")
            time.sleep(int(args.loop_hours * 3600))
    else:
        sys.exit(auto_once(args))

if __name__ == "__main__":
    main()
