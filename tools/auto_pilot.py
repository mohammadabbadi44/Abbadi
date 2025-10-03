# tools/auto_pilot.py
# -*- coding: utf-8 -*-
import os, sys, json, random, time, subprocess, math, copy
from datetime import datetime

try:
    import yaml
except Exception:
    yaml = None  # هنحاول نشتغل حتى لو YAML مش متوفّر

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKTEST_PY = os.path.join(PROJECT_ROOT, "backtester", "backtest_full.py")
SETTINGS_YAML = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
METRICS_JSON = os.path.join(PROJECT_ROOT, "logs", "metrics_auto.json")
SPACE_YAML = os.path.join(PROJECT_ROOT, "config", "auto_space.yaml")

# ======== Search Space ========
DEFAULT_SPACE = {
    "strategy": {
        "min_signals": {"type": "int", "low": 2, "high": 5},
        "use_ema200_trend_filter": {"type": "choice", "choices": [True, False]},
        "rr_ratio": {"type": "float", "low": 1.2, "high": 2.5},
        "sl_atr_mult": {"type": "float", "low": 1.0, "high": 3.0},
        "tp_atr_mult": {"type": "float", "low": 1.2, "high": 4.0},
        "use_trailing": {"type": "choice", "choices": [True, False]},
        "trail_atr_mult": {"type": "float", "low": 0.5, "high": 2.0},
        "trading_hours": {  # 09:00-18:00 / 24h
            "type": "choice",
            "choices": [
                {"enabled": False},
                {"enabled": True, "start": "09:00", "end": "18:00"}
            ]
        },
        "signal_strength_thresholds": {  # تحويل قوة الإشارة إلى مخاطرة
            "type": "grid",
            "values": [
                {"weak": 0.01, "medium": 0.02, "strong": 0.03, "very_strong": 0.04},
                {"weak": 0.00, "medium": 0.02, "strong": 0.03, "very_strong": 0.05},
                {"weak": 0.00, "medium": 0.015, "strong": 0.025, "very_strong": 0.04},
            ]
        },
        "ensemble": {
            "type": "soft",
            "weights": {
                # لو عندك أسماء استراتيجيات، أضفها هنا لتعديل أوزان التصويت
                # مثال:
                # "classic_combo": {"type": "float", "low": 0.5, "high": 2.0},
                # "donchian_fisher_keltner": {"type": "float", "low": 0.5, "high": 2.0},
            }
        }
    }
}

def load_yaml(path):
    if yaml is None:
        return None
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if yaml is None:
        # Fallback: خزّن JSON بدل YAML
        path_json = path.replace(".yaml", ".json")
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path_json
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path

def sample_param(spec):
    t = spec.get("type")
    if t == "int":
        return random.randint(spec["low"], spec["high"])
    if t == "float":
        return round(random.uniform(spec["low"], spec["high"]), 4)
    if t == "choice":
        return random.choice(spec["choices"])
    if t == "grid":
        return random.choice(spec["values"])
    if t == "soft":
        # weights dict
        wspec = spec.get("weights", {})
        weights = {}
        for k, vs in wspec.items():
            weights[k] = round(random.uniform(vs["low"], vs["high"]), 4)
        return {"weights": weights}
    raise ValueError(f"Unknown param type: {t}")

def sample_candidate(space):
    s = {}
    for k, v in space["strategy"].items():
        s[k] = sample_param(v)
    return {"strategy": s}

def mutate_candidate(cand, space, p=0.35):
    newc = copy.deepcopy(cand)
    for k, v in space["strategy"].items():
        if random.random() < p:
            newc["strategy"][k] = sample_param(v)
    return newc

def objective_score(metrics):
    """
    هدف متوازن: PF و WinRate و FinalCapital مع عقوبة للـ Drawdown.
    عدّل الأوزان براحتك.
    """
    if not metrics or "error" in metrics:
        return -1e9
    pf = (metrics.get("profit_factor") or 0) * 1.0
    wr = (metrics.get("win_rate") or 0) * 1.0  # كنسبة مئوية
    fc = (metrics.get("final_capital") or 0) * 1.0
    dd = abs(metrics.get("max_drawdown") or 0)

    # تطبيع بسيط
    wr_n = wr / 100.0
    pf_n = min(pf / 2.0, 1.5)  # سقف 3.0 → 1.5
    dd_penalty = min(dd / 10000.0, 1.0)  # عدّل حسب رأس المال

    score = 0.45 * (wr_n) + 0.45 * (pf_n) + 0.10 * (fc / 10000.0) - 0.50 * dd_penalty
    return score

def run_backtest_with_overrides(overrides_dict):
    overrides_str = json.dumps(overrides_dict, ensure_ascii=False)
    cmd = [sys.executable, BACKTEST_PY, "--overrides_json", overrides_str, "--out_json", METRICS_JSON]
    try:
        out = subprocess.check_output(cmd, cwd=PROJECT_ROOT, stderr=subprocess.STDOUT, timeout=1800)
        # stdout يُطبع فيه المِتريكس JSON (من الباتش)
        text = out.decode("utf-8", errors="ignore").strip()
        try:
            metrics = json.loads(text.splitlines()[-1])
        except Exception:
            # fallback: اقرأ من الملف
            with open(METRICS_JSON, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        return metrics
    except subprocess.CalledProcessError as e:
        return {"error": f"Backtest failed: {e.output.decode('utf-8', errors='ignore')}"}
    except Exception as e:
        return {"error": str(e)}

def backup_settings():
    if os.path.isfile(SETTINGS_YAML):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = SETTINGS_YAML + f".bak.{ts}"
        with open(SETTINGS_YAML, "rb") as src, open(bak, "wb") as dst:
            dst.write(src.read())
        return bak
    return None

def apply_best_to_settings(best_overrides):
    base = load_yaml(SETTINGS_YAML) or {}
    def deep_merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                deep_merge(a[k], v)
            else:
                a[k] = v
        return a
    merged = deep_merge(base, best_overrides)
    save_yaml(SETTINGS_YAML, merged)

def main():
    print("🚀 Auto-Pilot started")
    random.seed(42)

    space = load_yaml(SPACE_YAML) or DEFAULT_SPACE
    pop_size = 18
    generations = 10
    elite_k = 4
    hill_steps = 3

    population = [sample_candidate(space) for _ in range(pop_size)]
    history = []

    best = None
    best_score = -1e9

    for gen in range(generations):
        print(f"\n=== Generation {gen+1}/{generations} ===")
        scored = []
        for idx, cand in enumerate(population):
            metrics = run_backtest_with_overrides(cand)
            score = objective_score(metrics)
            scored.append((score, cand, metrics))
            history.append({"gen": gen+1, "score": score, "cand": cand, "metrics": metrics})
            msg = f"[G{gen+1} {idx+1}/{len(population)}] score={round(score,4)} | metrics={metrics}"
            print(msg)

            if score > best_score:
                best_score = score
                best = copy.deepcopy(cand)

        # اختيار النخبة + طفرات + تزاوج بسيط
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [copy.deepcopy(c) for _, c, _ in scored[:elite_k]]

        # Hill-climbing حول الأفضل
        local_best = copy.deepcopy(elites[0])
        local_best_score = scored[0][0]
        for _ in range(hill_steps):
            neighbor = mutate_candidate(local_best, space, p=0.5)
            m = run_backtest_with_overrides(neighbor)
            s = objective_score(m)
            history.append({"gen": gen+1, "score": s, "cand": neighbor, "metrics": m})
            print(f"  ↳ hill step score={round(s,4)}")
            if s > local_best_score:
                local_best, local_best_score = neighbor, s

        # جيل جديد
        new_pop = elites + [local_best]
        while len(new_pop) < pop_size:
            parent = random.choice(elites)
            child = mutate_candidate(parent, space, p=0.35)
            new_pop.append(child)
        population = new_pop[:pop_size]

    # طبّق الأفضل على settings.yaml
    print("\n🏁 Finished. Applying best overrides to settings.yaml")
    backup_settings()
    apply_best_to_settings(best)

    # حفظ التاريخ
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "logs", "auto_pilot_history.json"), "w", encoding="utf-8") as f:
        json.dump({"best": best, "best_score": best_score, "history": history}, f, ensure_ascii=False, indent=2)

    print(f"✅ Best score = {round(best_score,4)}")
    print(f"✅ Best overrides:\n{json.dumps(best, ensure_ascii=False, indent=2)}")
    print("تم ✨")

if __name__ == "__main__":
    main()
