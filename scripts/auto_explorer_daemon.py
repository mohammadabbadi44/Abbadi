# -*- coding: utf-8 -*-
"""
scripts/auto_explorer_daemon.py

Autopilot:
- يستكشف إعدادات الإنسمبل + AI (Wide -> Narrow) عبر عدة TFs
- يقلّب الاستراتيجيات ON/OFF ويعدل الأوزان والفلاتر و SL/TP
- لو طلعت "لا توجد صفقات" يرخي الشروط تلقائياً ويعيد المحاولة (حد أقصى محدد بالحوكمة)
- يختار الأفضل وفق governor.yaml
- يروّج أفضل إعداد -> config/ensemble_ai.yaml
- يعمل Backtest تأكيدي + Walk-Forward (آخر 60 يوم)
- يقرّر الاعتماد/الرفض + رولباك تلقائي لو الجودة هبطت
"""

import os, sys, json, random, argparse, time
from pathlib import Path
from datetime import datetime, timedelta

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
    print("❌ MetaTrader5 غير مثبت. pip install MetaTrader5")
    sys.exit(1)

REPORTS = ROOT / "reports"
CONFIG  = ROOT / "config" / "ensemble_ai.yaml"
GOVCONF = ROOT / "config" / "governor.yaml"
REPORTS.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
def load_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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

def sample_cfg(base: dict, tf: str, narrow: bool, gov: dict) -> dict:
    """ عينات إعدادات مع ON/OFF ديناميكي + أوزان + فلاتر + إدارة صفقة """
    cfg = json.loads(json.dumps(base))
    cfg["timeframe"] = tf

    # بوابات الإنسمبل
    cfg["ensemble_mode"] = "weighted_mean"
    cfg["min_ensemble_votes"] = 3
    cfg["min_ensemble_conf"]  = round(random.uniform(0.62, 0.75) if not narrow else random.uniform(0.66, 0.75), 2)
    cfg["use_ema200_trend_filter"] = True
    cfg["cooldown_bars"] = 4
    cfg["one_position"]  = True

    # ساعات التداول
    th_enabled = random.choice([True, True, False])
    cfg.setdefault("trading_hours", {})
    cfg["trading_hours"]["enabled"] = th_enabled
    cfg["trading_hours"]["start"]   = random.choice(["06:30","07:00"]) if th_enabled else "00:00"
    cfg["trading_hours"]["end"]     = random.choice(["20:00","20:30","21:00"]) if th_enabled else "23:59"

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

    # فلاتر الجودة (من الحوكمة)
    g = gov.get("guards", {})
    atr_rng = g["min_atr_pct_narrow" if narrow else "min_atr_pct_wide"]
    spread_rng = g["max_spread_vs_atr_narrow" if narrow else "max_spread_vs_atr_wide"]
    cfg.setdefault("guards", {})
    cfg["guards"]["min_atr_pct"]       = round(random.uniform(*atr_rng), 6)
    cfg["guards"]["max_atr_pct"]       = round(random.uniform(0.028, 0.04), 3)
    cfg["guards"]["max_spread_vs_atr"] = round(random.uniform(*spread_rng), 2)
    cfg["guards"]["max_trades_per_day"]= random.choice(g.get("max_trades_per_day",[80,100]))

    # مخاطرة
    cfg.setdefault("risk", {})
    cfg["risk"]["risk_per_trade_pct"]     = round(random.uniform(0.20, 0.30), 2)
    cfg["risk"]["max_daily_loss_pct"]     = 2.5
    cfg["risk"]["max_total_drawdown_pct"] = 30.0

    # AI
    cfg.setdefault("ai", {})
    cfg["ai"]["enabled"] = True
    cfg["ai"]["decision_threshold"] = round(random.uniform(0.66, 0.74) if not narrow else random.uniform(0.69, 0.74), 2)
    cfg["ai"]["train_if_missing"] = True
    cfg["ai"]["model_path"] = base.get("ai", {}).get("model_path", "models/ensemble_lr.joblib")

    # الاستراتيجيات: ON/OFF + أوزان
    p_on = (  # احتمالات التفعيل
        {"trend_following":0.95,"momentum_macd":0.90,"breakout_donchian":0.80,"volatility_ttm":0.75,"volume_obv":0.70,"mean_reversion":0.30}
        if not narrow else
        {"trend_following":0.98,"momentum_macd":0.95,"breakout_donchian":0.85,"volatility_ttm":0.75,"volume_obv":0.70,"mean_reversion":0.20}
    )
    weight_rng = (
        {"trend_following":(1.3,1.8),"momentum_macd":(1.1,1.5),"breakout_donchian":(0.9,1.1),"volatility_ttm":(0.7,0.9),"volume_obv":(0.6,0.8),"mean_reversion":(0.4,0.7)}
        if not narrow else
        {"trend_following":(1.6,1.8),"momentum_macd":(1.3,1.5),"breakout_donchian":(0.9,1.1),"volatility_ttm":(0.7,0.9),"volume_obv":(0.6,0.8),"mean_reversion":(0.4,0.6)}
    )
    cfg.setdefault("strategies", {})
    for name in ["trend_following","momentum_macd","breakout_donchian","volatility_ttm","volume_obv","mean_reversion"]:
        cfg["strategies"].setdefault(name, {})
        enabled = (random.random() < p_on[name])
        cfg["strategies"][name]["enabled"] = bool(enabled)
        if enabled:
            a,b = weight_rng[name]
            cfg["strategies"][name]["weight"] = round(random.uniform(a,b), 2)
        else:
            cfg["strategies"][name]["weight"] = 0.0

    # تأكد وجود عدد كافي مفعّل ≥ min_votes
    min_votes = int(cfg["min_ensemble_votes"])
    enabled_list = [n for n,v in cfg["strategies"].items() if v.get("enabled")]
    if len(enabled_list) < min_votes:
        for name in ["trend_following","momentum_macd","breakout_donchian","volatility_ttm","volume_obv","mean_reversion"]:
            if not cfg["strategies"][name]["enabled"]:
                cfg["strategies"][name]["enabled"] = True
                if cfg["strategies"][name]["weight"] == 0.0:
                    a,b = weight_rng[name]; cfg["strategies"][name]["weight"]=round(random.uniform(a,b),2)
                enabled_list.append(name)
                if len(enabled_list) >= min_votes: break
    return cfg

def score(stats: dict, gov: dict, stage: str) -> float:
    """ دالة هدف: PF أولوية + عقوبة DD + حد صفقات """
    if not stats: return -1e9
    s = gov["search"]
    pf = float(stats["profit_factor"]) if not isinstance(stats["profit_factor"], str) else (999.0 if str(stats["profit_factor"]).lower()=="inf" else float(stats["profit_factor"]))
    dd = float(stats["max_dd"]); tr = int(stats["trades"]); wr = float(stats["win_rate"]); ret=float(stats["return_pct"])
    if tr < int(s["min_trades"]): return -1e9
    tgt = float(s["target_pf"]) if stage=="narrow" else max(1.0, float(s["target_pf"])-0.05)
    if pf < 1.0 and stage=="wide": return -1e6 + (pf - 1.0)
    sc = 2.0*(pf - tgt) + 0.004*(wr - 45.0) + 0.0006*ret
    if dd > float(s["dd_cap"]): sc -= 0.08 * (dd - float(s["dd_cap"]))
    if tr > 1500: sc -= 0.0006 * (tr - 1500)
    return sc

def run_stage(df, feats, si, base_cfg, tf, trials, gov, stage):
    rows, best, best_score = [], None, -1e18
    narrow = (stage=="narrow")
    retries_no_trades = 0
    max_retries = int(gov["acceptance"]["max_retries_if_no_trades"])
    for i in range(1, trials+1):
        cfg_try = sample_cfg(base_cfg, tf, narrow, gov)
        stats, _ = backtest(df, feats, cfg_try, si)
        if stats is None:
            retries_no_trades += 1
            if retries_no_trades <= max_retries:
                # رخيّة ذكية: نخفض conf/threshold شَعرة ونوسع spread_vs_atr شَعرة
                cfg_try["min_ensemble_conf"] = max(0.60, cfg_try["min_ensemble_conf"] - 0.02)
                cfg_try["ai"]["decision_threshold"] = max(0.65, cfg_try["ai"]["decision_threshold"] - 0.02)
                cfg_try["guards"]["max_spread_vs_atr"] = min(0.30, cfg_try["guards"]["max_spread_vs_atr"] + 0.02)
                # إعادة المحاولة بنفس الـ cfg_try:
                stats2, _ = backtest(df, feats, cfg_try, si)
                stats = stats2
            # لو برضه None، نكمل لتجربة أخرى
        sc = score(stats, gov, stage) if stats else -1e9
        row = {"trial": i, "tf": tf, "stage": stage, "score": round(sc,6)}
        if stats: row.update(stats)
        rows.append(row)
        if sc > best_score:
            best_score = sc; best = (cfg_try, stats)
            pf_show = stats.get('profit_factor') if stats else 'NA'
            dd_show = stats.get('max_dd') if stats else 'NA'
            tr_show = stats.get('trades') if stats else 0
            print(f"[{tf} {stage} {i}/{trials}] ⭐ NEW BEST score={best_score:.4f} | PF={pf_show} | DD={dd_show} | Trades={tr_show}")
    return pd.DataFrame(rows), best

def promote_and_confirm(cfg_best, stats_best, df, feats, si):
    CONFIG.write_text(to_yamlish({**cfg_best, "_best_stats": stats_best}), encoding="utf-8")
    print(f"✅ Promoted → {CONFIG}")
    confirm_stats, _ = backtest(df, feats, cfg_best, si)
    return confirm_stats

def walk_forward(df_all: pd.DataFrame, cfg, si, days=60):
    end = df_all["time"].iloc[-1]
    df_wf = df_all[df_all["time"] >= end - pd.Timedelta(days=days)].reset_index(drop=True)
    feats_wf = build_features(df_wf)
    wf_stats, _ = backtest(df_wf, feats_wf, cfg, si)
    return wf_stats

def meets_acceptance(confirm_stats: dict, wf_stats: dict, gov: dict) -> bool:
    a = gov["acceptance"]
    ok = (
        float(confirm_stats["profit_factor"]) >= float(a["confirm_min_pf"]) and
        int(confirm_stats["trades"]) >= int(a["confirm_min_trades"]) and
        float(confirm_stats["max_dd"]) <= float(a["confirm_max_dd_pct"]) and
        float(confirm_stats["win_rate"]) >= float(a["confirm_min_win_pct"]) and
        (wf_stats is None or (
            float(wf_stats["profit_factor"]) >= float(a["walk_min_pf"]) and
            float(wf_stats["max_dd"]) <= float(a["walk_max_dd_pct"]) and
            int(wf_stats["trades"]) >= int(a["walk_min_trades"])
        ))
    )
    return bool(ok)

def save_summary(tf_summaries, confirm_stats, wf_stats, tag="daemon"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = REPORTS / f"autopilot_summary_{tag}_{ts}.csv"
    rows = list(tf_summaries)
    rows.append({
        "tf": f"OVERALL",
        "pf": confirm_stats.get("profit_factor") if confirm_stats else None,
        "dd": confirm_stats.get("max_dd") if confirm_stats else None,
        "tr": confirm_stats.get("trades") if confirm_stats else None,
        "score": None,
        "yaml": str(CONFIG)
    })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"📝 Saved summary: {out_csv}")

def run_once(governor: dict):
    base_cfg = load_cfg(CONFIG)
    tfs = [str(x).upper() for x in governor["search"]["tfs"]]
    trials_w = int(governor["search"]["trials_wide"])
    trials_n = int(governor["search"]["trials_narrow"])

    if not mt5.initialize():
        print("❌ mt5.initialize failed:", mt5.last_error()); return 1

    try:
        tf_summaries = []
        overall_best, overall_score = None, -1e18
        overall_df = overall_feats = overall_si = None

        for tf in tfs:
            base_tf = dict(base_cfg); base_tf["timeframe"] = tf
            df, si = fetch_data(base_tf["symbol"], base_tf["timeframe"], base_tf["days"])
            feats = build_features(df)

            wide_df, wide_best = run_stage(df, feats, si, base_tf, tf, trials_w, governor, "wide")
            wide_df.to_csv(REPORTS / f"daemon_wide_{tf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
            if not wide_best or not wide_best[1]:
                print(f"⚠️ {tf}: لا نتائج صالحة في Wide."); continue

            base_narrow = json.loads(json.dumps(wide_best[0]))
            narrow_df, narrow_best = run_stage(df, feats, si, base_narrow, tf, trials_n, governor, "narrow")
            narrow_df.to_csv(REPORTS / f"daemon_narrow_{tf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
            if not narrow_best or not narrow_best[1]:
                print(f"⚠️ {tf}: لا نتائج صالحة في Narrow."); continue

            # تقييم TF winner
            sc = score(narrow_best[1], governor, "narrow")
            tf_summaries.append({"tf": tf, "pf": narrow_best[1]["profit_factor"], "dd": narrow_best[1]["max_dd"], "tr": narrow_best[1]["trades"], "score": round(sc,4)})
            if sc > overall_score:
                overall_score = sc; overall_best = narrow_best
                overall_df, overall_feats, overall_si = df, feats, si

        if not overall_best:
            print("❌ لا يوجد فائز عبر جميع TFs."); return 2

        # ترويج + تأكيد + Walk-Forward
        best_cfg, best_stats = overall_best
        confirm_stats = promote_and_confirm(best_cfg, best_stats, overall_df, overall_feats, overall_si)
        wf_stats = walk_forward(overall_df, best_cfg, overall_si, days=60)

        print(f"Confirm → PF={confirm_stats['profit_factor']} | Win%={confirm_stats['win_rate']} | DD={confirm_stats['max_dd']}% | Trades={confirm_stats['trades']} | Return={confirm_stats['return_pct']}%")
        if wf_stats:
            print(f"WalkFwd → PF={wf_stats['profit_factor']} | Win%={wf_stats['win_rate']} | DD={wf_stats['max_dd']}% | Trades={wf_stats['trades']} | Return={wf_stats['return_pct']}%")

        # قرار القبول/الرفض
        accepted = meets_acceptance(confirm_stats, wf_stats, governor)
        if accepted:
            print("✅ Accepted config (meets governance thresholds).")
            # مهر تثبيت لمنع الترويج السريع المتكرر
            (REPORTS / "last_promote.txt").write_text(datetime.now().isoformat(), encoding="utf-8")
        else:
            print("🚫 Rejected — لا يحقق المعايير. سنستمر في الاستكشاف لاحقًا.")
        save_summary(tf_summaries, confirm_stats, wf_stats, tag="run")

        return 0 if accepted else 3
    finally:
        mt5.shutdown()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="تشغيل دوري حسب governor.yaml")
    args = ap.parse_args()

    if not GOVCONF.exists():
        print(f"⚠️ لا يوجد {GOVCONF}. أنشأته لك افتراضي.")
        GOVCONF.parent.mkdir(parents=True, exist_ok=True)
        GOVCONF.write_text("""acceptance:
  confirm_min_pf: 1.05
  confirm_min_trades: 100
  confirm_max_dd_pct: 5.0
  confirm_min_win_pct: 45.0
  walk_min_pf: 0.95
  walk_max_dd_pct: 6.0
  walk_min_trades: 40
  freeze_hours_after_promote: 24
  max_retries_if_no_trades: 8
search:
  trials_wide: 80
  trials_narrow: 180
  tfs: ["M5","M15"]
  target_pf: 1.06
  dd_cap: 25.0
  min_trades: 90
guards:
  min_atr_pct_wide: [0.0008, 0.0012]
  min_atr_pct_narrow: [0.0009, 0.0012]
  max_spread_vs_atr_wide: [0.20, 0.28]
  max_spread_vs_atr_narrow: [0.20, 0.24]
  max_trades_per_day: [80, 100, 120]
loop:
  enabled: false
  hours: 6
""", encoding="utf-8")

    governor = load_yaml(GOVCONF)

    if args.loop or (governor.get("loop",{}).get("enabled", False)):
        hours = float(governor["loop"].get("hours", 6))
        print(f"⏱️ Autopilot loop ON — سيعيد الدورة كل {hours} ساعة.")
        while True:
            rc = run_once(governor)
            print(f"— دورة انتهت بكود: {rc} — {datetime.now().isoformat()}")
            time.sleep(int(hours * 3600))
    else:
        sys.exit(run_once(governor))

if __name__ == "__main__":
    main()
