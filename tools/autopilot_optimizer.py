# tools/autopilot_optimizer.py
# -*- coding: utf-8 -*-
import os, sys, json, copy, random, argparse, time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Fix Windows console encoding (no emojis anyway) ===
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from backtester.backtest_full import run_backtest_entrypoint as run_bt
except Exception as e:
    print("[ERR] import backtester.backtest_full:", e); raise

try:
    import yaml
except Exception:
    yaml = None

def deep_merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a

def read_yaml(path):
    if yaml is None or not os.path.isfile(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            d = yaml.safe_load(f) or {}
        except Exception:
            d = {}
    return d if isinstance(d, dict) else {}

def write_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if yaml:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def make_space(goal):
    space = {
        "min_signals": [2,3,4,5,6],
        "use_ema200_trend_filter": [True, False],
        "rr_ratio": (1.1, 2.2),
        "sl_atr_mult": (1.6, 3.2),
        "tp_atr_mult": (2.0, 4.4),
        "use_trailing": [True, False],
        "trail_atr_mult": (0.9, 1.4),
        "trading_hours": [
            {"enabled": False},
            {"enabled": True, "start": "07:00", "end": "19:00"},
            {"enabled": True, "start": "08:00", "end": "17:00"},
            {"enabled": True, "start": "12:00", "end": "18:00"}
        ],
        "signal_strength_thresholds": [
            {"weak": 0.000, "medium": 0.010, "strong": 0.020, "very_strong": 0.040},
            {"weak": 0.005, "medium": 0.015, "strong": 0.030, "very_strong": 0.050},
            {"weak": 0.005, "medium": 0.015, "strong": 0.030, "very_strong": 0.060}
        ]
    }
    if goal == "high_profit":
        space["min_signals"] = [3,4,5]
        space["rr_ratio"] = (1.3, 2.4)
        space["sl_atr_mult"] = (1.8, 2.8)
        space["tp_atr_mult"] = (2.6, 4.4)
    elif goal == "low_dd":
        space["min_signals"] = [5,6]
        space["use_ema200_trend_filter"] = [True]
        space["rr_ratio"] = (1.1, 1.6)
        space["sl_atr_mult"] = (2.4, 3.2)
        space["tp_atr_mult"] = (2.6, 3.6)
        space["trading_hours"] = [
            {"enabled": True, "start": "07:00", "end": "17:00"},
            {"enabled": True, "start": "08:00", "end": "16:00"}
        ]
    elif goal == "more_trades":
        space["min_signals"] = [2,3,4]
        space["rr_ratio"] = (1.2, 1.7)
        space["sl_atr_mult"] = (1.6, 2.4)
        space["tp_atr_mult"] = (2.2, 3.2)
        space["trading_hours"] = [
            {"enabled": False},
            {"enabled": True, "start": "07:00", "end": "20:00"}
        ]
    return space

def sample_candidate(space):
    def r(lo, hi): return lo + random.random()*(hi-lo)
    return {
        "strategy": {
            "min_signals": random.choice(space["min_signals"]),
            "use_ema200_trend_filter": random.choice(space["use_ema200_trend_filter"]),
            "rr_ratio": round(r(*space["rr_ratio"]), 4),
            "sl_atr_mult": round(r(*space["sl_atr_mult"]), 4),
            "tp_atr_mult": round(r(*space["tp_atr_mult"]), 4),
            "use_trailing": random.choice(space["use_trailing"]),
            "trail_atr_mult": round(r(*space["trail_atr_mult"]), 4),
            "trading_hours": random.choice(space["trading_hours"]),
            "signal_strength_thresholds": random.choice(space["signal_strength_thresholds"]),
            "allowed_strategies": [],
            "ensemble": {"weights": {}}
        },
        "backtest": {"commission_perc": 0.0002, "slippage_perc": 0.0001},
        "imports": {"dirs": ["strategy","strategies","indicators","indicator","signals","modules","utils"]},
        "guards": {"max_dd_stop_pct": 22.0, "daily_stop_pct": 2.5, "daily_take_profit_pct": 3.0}
    }

# Targets tuned for 5m by default
TARGET_TR_MIN, TARGET_TR_MAX = 200, 700

def trade_range_bonus(n):
    if n is None: return -1.0
    if n < TARGET_TR_MIN: return -0.002*(TARGET_TR_MIN-n)
    if n <= TARGET_TR_MAX: return 0.0015*(n-TARGET_TR_MIN)
    return -0.003*(n-TARGET_TR_MAX)

def objective_score(m, goal):
    if not m or "error" in m: return -1e9
    # robust pulls with defaults
    pf = float(m.get("profit_factor") or 0.0)
    wr = float(m.get("win_rate") or 0.0)/100.0
    dd = abs(float(m.get("max_dd_pct") or 100.0))
    n  = int(m.get("trades") or 0)
    hit3 = float(m.get("daily_hit_3pct_rate") or 0.0) / 100.0

    if pf < 1.10 or dd > 25 or n < 150:
        return -1e9

    pf_n = min(pf/2.0, 1.5)
    dd_pen = (dd/20.0)**2
    tr_b = trade_range_bonus(n)

    if goal == "high_profit":
        w = (0.35,0.15,0.30,0.05,0.35)  # pf, wr, hit3, tr, dd_pen
    elif goal == "low_dd":
        w = (0.30,0.20,0.25,0.05,0.45)
    elif goal == "more_trades":
        w = (0.30,0.15,0.20,0.25,0.35)
    else:  # balanced
        w = (0.32,0.18,0.30,0.10,0.38)

    score = w[0]*pf_n + w[1]*wr + w[2]*hit3 + w[3]*tr_b - w[4]*dd_pen
    return score

def evaluate_candidate(overrides, goal):
    base = copy.deepcopy(overrides) or {}
    guards = base.get("guards", {})
    if "daily_take_profit_pct" not in guards:
        guards["daily_take_profit_pct"] = 3.0
        base["guards"] = guards
    try:
        m = run_bt(base)
    except Exception as e:
        m = {"error": f"{e}"}
    s = objective_score(m, goal)
    return s, base, m

def evaluate_batch(cands, goal, workers):
    out = []
    print(f"[AP] Evaluating batch of {len(cands)} candidates with workers={workers}...", flush=True)
    start = time.perf_counter()
    logs_dir = os.path.join(PROJECT_ROOT, "logs"); os.makedirs(logs_dir, exist_ok=True)
    hb_path = os.path.join(logs_dir, "autopilot_progress.log")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(evaluate_candidate, c, goal): c for c in cands}
        done = 0; last_hb = 0
        for fut in as_completed(futs):
            try:
                s, c, m = fut.result()
            except Exception as e:
                s, c, m = -1e9, futs[fut], {"error": str(e)}
            out.append((s, c, m)); done += 1
            elapsed = time.perf_counter() - start
            eta = (elapsed/done)*(len(cands)-done) if done else 0
            best_so_far = max([x[0] for x in out]) if out else float("-inf")
            msg = f"[AP] Progress {done}/{len(cands)} | elapsed={elapsed:.1f}s | eta={eta:.1f}s | best={best_so_far:.4f}"
            print(msg, flush=True)
            now = time.perf_counter()
            if now - last_hb > 3:
                with open(hb_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                last_hb = now
    return out

def neighbor_candidates(base, k=8):
    out = []
    for _ in range(k):
        c = copy.deepcopy(base)
        s = c["strategy"]
        def j(x,p=0.12,lo=None,hi=None):
            v = x*(1+random.uniform(-p,p))
            if lo is not None: v = max(lo, v)
            if hi is not None: v = min(hi, v)
            return round(v,4)
        s["rr_ratio"] = j(s["rr_ratio"],0.10,1.05,2.6)
        s["sl_atr_mult"] = j(s["sl_atr_mult"],0.12,1.4,3.6)
        s["tp_atr_mult"] = j(s["tp_atr_mult"],0.12,2.0,4.8)
        s["trail_atr_mult"] = j(s["trail_atr_mult"],0.12,0.8,1.5)
        if random.random()<0.35: s["use_trailing"] = not s.get("use_trailing",True)
        if random.random()<0.30: s["use_ema200_trend_filter"] = not s.get("use_ema200_trend_filter",True)
        if random.random()<0.30: s["min_signals"] = max(1,min(7,int(s.get("min_signals",3)+random.choice([-1,1]))))
        out.append(c)
    return out

def optimize(symbol, timeframe, goal, coarse_n, topk, refine_each, workers):
    random.seed(42)
    space = make_space(goal)
    logs_dir = os.path.join(PROJECT_ROOT, "logs"); os.makedirs(logs_dir, exist_ok=True)
    runs_path = os.path.join(logs_dir, "autopilot_runs.csv")

    base = {
        "data": {"symbol": symbol, "timeframe": timeframe},
        "strategy": {"profile": "fast_stable"},
        "imports": {"dirs": ["strategy","strategies","indicators","indicator","signals","modules","utils"]},
        "guards": {"daily_take_profit_pct": 3.0, "daily_stop_pct": 2.5, "max_dd_stop_pct": 22.0}
    }

    coarse = []
    for _ in range(coarse_n):
        c = copy.deepcopy(base); deep_merge(c, sample_candidate(space)); coarse.append(c)

    print(f"[AP] Submitting COARSE {len(coarse)} candidates...", flush=True)
    res_coarse = evaluate_batch(coarse, goal, workers)
    with open(runs_path,"a",encoding="utf-8") as f:
        for s,c,m in res_coarse:
            f.write(json.dumps({"phase":"coarse","score":s,"overrides":c,"metrics":m,"ts":now_iso()},ensure_ascii=False)+"\n")

    elites = sorted(res_coarse, key=lambda x:x[0], reverse=True)[:max(1,min(topk,len(res_coarse)))]
    print(f"[AP] COARSE done. Top-{len(elites)} selected. Generating REFINE...", flush=True)

    refine = []
    for s,c,m in elites:
        refine += neighbor_candidates(c, k=refine_each)

    print(f"[AP] Submitting REFINE {len(refine)} candidates...", flush=True)
    res_refine = evaluate_batch(refine, goal, workers)
    with open(runs_path,"a",encoding="utf-8") as f:
        for s,c,m in res_refine:
            f.write(json.dumps({"phase":"refine","score":s,"overrides":c,"metrics":m,"ts":now_iso()},ensure_ascii=False)+"\n")

    all_res = res_coarse + res_refine
    best = max(all_res, key=lambda x:x[0])
    best_score, best_over, best_metrics = best

    best_payload = {
        "goal": goal,
        "score": round(float(best_score),4),
        "symbol": symbol,
        "timeframe": timeframe,
        "metrics": best_metrics,
        "overrides": best_over,
        "ts": now_iso()
    }
    write_json(os.path.join(logs_dir,"best_profile.json"), best_payload)

    settings_path = os.path.join(PROJECT_ROOT,"config","settings.yaml")
    curr = read_yaml(settings_path)
    if "strategy" not in curr: curr["strategy"]={}
    if "backtest" not in curr: curr["backtest"]={}
    if "guards" not in curr: curr["guards"]={}
    write_yaml(settings_path, deep_merge(curr, copy.deepcopy(best_over)))

    # Plain ASCII prints to avoid cp1252 issues
    print("Finished. Applied best overrides to settings.yaml")
    print("Best score =", round(float(best_score),4))
    print("Best overrides:")
    print(json.dumps(best_over, ensure_ascii=False, indent=2))
    return best_payload

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--timeframe", default="5m")  # default to 5m
    p.add_argument("--goal", default="balanced", choices=["balanced","high_profit","low_dd","more_trades"])
    p.add_argument("--coarse", type=int, default=28)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--refine_each", type=int, default=6)
    p.add_argument("--workers", type=int, default=6)
    return p.parse_args()

def main():
    a = parse_args()
    optimize(a.symbol, a.timeframe, a.goal, a.coarse, a.topk, a.refine_each, a.workers)

if __name__ == "__main__":
    main()
