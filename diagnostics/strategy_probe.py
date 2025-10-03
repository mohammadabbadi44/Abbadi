# diagnostics/strategy_probe.py
# -*- coding: utf-8 -*-

import os
import sys
import glob
import importlib.util
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

# ========= إعدادات عامة =========
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .. من داخل diagnostics/
sys.path.append(str(PROJECT_ROOT))

# اسم ورقة البيانات
SYMBOL = os.environ.get("PROBE_SYMBOL", "XAUUSD")
TIMEFRAME = os.environ.get("PROBE_TF", "30m")
CSV_PATH = os.environ.get("PROBE_CSV", str(PROJECT_ROOT / f"data/historical/{SYMBOL}_{TIMEFRAME}.csv"))

# مسار الاستراتيجيات (اسمح بتغييره بمتغير بيئة)
DEFAULT_STRAT_DIR = PROJECT_ROOT / "strategy" / "forex"
STRAT_DIR = Path(os.environ.get("STRAT_DIR", str(DEFAULT_STRAT_DIR))).resolve()

OUT_DIR = PROJECT_ROOT / "logs" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _load_df():
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    if "time" not in df.columns and "timestamp" in df.columns:
        df.rename(columns={"timestamp": "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"])
    for c in ["open","high","low","close","volume"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    return df

def _discover_strategy_files():
    # ابحث Recursively عن .py بدون الملفات اللي تبدأ بـ _
    patterns = [str(STRAT_DIR / "**" / "*.py")]
    files = []
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            fn = os.path.basename(p)
            if fn.startswith("_"):
                continue
            files.append(Path(p))
    return sorted(files)

def _import_module_from_path(mod_name, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _normalize_signal(x):
    if isinstance(x, (np.generic,)): x = x.item()
    if isinstance(x, (list, tuple)) and len(x): x = x[0]
    if isinstance(x, dict):
        for k in ("signal","sig","direction","dir","value"):
            if k in x: x = x[k]; break
    if isinstance(x, (bool, np.bool_)): return "Buy" if x else "Hold"
    if isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x):
        if x > 0: return "Buy"
        if x < 0: return "Sell"
        return "Hold"
    if isinstance(x, str):
        s = str(x).strip().lower()
        if s in {"buy","long","bull","bullish","up","+","1","true","yes"} or "buy" in s or "long" in s or "bull" in s:
            return "Buy"
        if s in {"sell","short","bear","bearish","down","-","-1","false","no"} or "sell" in s or "short" in s or "bear" in s:
            return "Sell"
        if s in {"hold","wait","neutral","none","flat","0",""}:
            return "Hold"
        if s.startswith(("b","+")): return "Buy"
        if s.startswith(("s","-")): return "Sell"
    return "Hold"

def _series_like_to_series(sig_like, n):
    if isinstance(sig_like, pd.Series):
        s = sig_like.copy()
        if len(s) != n:
            s = s.reindex(range(n))
    elif hasattr(sig_like, "__iter__"):
        arr = list(sig_like)
        if len(arr) < n: arr += [np.nan] * (n - len(arr))
        arr = arr[:n]
        s = pd.Series(arr)
    else:
        s = pd.Series([np.nan] * n)
    return s

def _auto_threshold_suggestion(num_series):
    s = pd.to_numeric(num_series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty: return None, None
    med = s.median()
    mad = (s - med).abs().median()
    robust_sigma = 1.4826 * mad if pd.notna(mad) and mad > 0 else s.std()
    if robust_sigma and robust_sigma > 0:
        eps = 0.5 * robust_sigma
        pos = (s > eps).mean()
        neg = (s < -eps).mean()
        zer = ((s >= -eps) & (s <= eps)).mean()
        return eps, {"p_pos(>|eps|)": round(float(pos)*100,2),
                     "p_neg(<-|eps|)": round(float(neg)*100,2),
                     "p_mid(|x|<=eps)": round(float(zer)*100,2)}
    return None, None

def _summarize(signals_raw, df, module_name):
    n = len(df)
    s = _series_like_to_series(signals_raw, n)
    s_norm = s.map(_normalize_signal)

    buy = int((s_norm == "Buy").sum())
    sell = int((s_norm == "Sell").sum())
    hold = int((s_norm == "Hold").sum())
    coverage = {"Buy": buy, "Sell": sell, "Hold": hold, "non-Hold%": round(100*(buy+sell)/max(1,n),2)}

    dtype = str(s.dtype)
    num_stats = {}
    if pd.api.types.is_numeric_dtype(s):
        eps, parts = _auto_threshold_suggestion(s)
        if eps is not None:
            num_stats["suggest_eps"] = float(eps)
            if parts: num_stats.update(parts)
        num_stats["pos% (x>0)"] = round(100 * (s > 0).mean(), 2)
        num_stats["neg% (x<0)"] = round(100 * (s < 0).mean(), 2)
        num_stats["zero%"]      = round(100 * (s == 0).mean(), 2)
        num_stats["nan%"]       = round(100 * s.isna().mean(), 2)
        num_stats["std"]        = float(s.std(skipna=True)) if s.notna().any() else np.nan

    return s_norm, coverage, dtype, num_stats

def main():
    print(f"[DBG] CWD: {os.getcwd()}")
    print(f"[DBG] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[DBG] STRAT_DIR: {STRAT_DIR}")
    print(f"[DBG] CSV_PATH: {CSV_PATH}")

    df = _load_df()
    files = _discover_strategy_files()
    print(f"🔎 Found {len(files)} .py files under: {STRAT_DIR}")
    for p in files[:10]:
        print(f"   - {p.relative_to(PROJECT_ROOT)}")
    if not files:
        print("❌ لا يوجد ملفات .py في هذا المسار. تأكد من المسار ومن أنك تشغّل من جذر المشروع.")
        print('   غيّر المسار ببيئة STRAT_DIR مثلًا:  set STRAT_DIR="D:\\trading-bot new\\strategy\\forex"')
        print("   أو حرّك السكربت لتشغله من الجذر.")
        out = pd.DataFrame([{"strategy": "(none found)"}])
        out.to_csv(OUT_DIR / f"probe_{SYMBOL}_{TIMEFRAME}.csv", index=False)
        return

    rows = []
    for path in files:
        mod_name = path.stem
        try:
            mod = _import_module_from_path(mod_name, path)
            if mod is None:
                rows.append({"strategy": mod_name, "error": "import_failed"}); continue
            if not hasattr(mod, "generate_signal"):
                rows.append({"strategy": mod_name, "error": "no_generate_signal"}); continue

            try:
                sig = mod.generate_signal(df)
            except TypeError:
                try:
                    sig = mod.generate_signal(df, debug=True)
                except Exception:
                    sig = mod.generate_signal(df)

            s_norm, cov, dtype, num_stats = _summarize(sig, df, mod_name)
            row = {
                "strategy": mod_name,
                "dtype": dtype,
                "buy": cov["Buy"], "sell": cov["Sell"], "hold": cov["Hold"],
                "non_hold_%": cov["non-Hold%"],
                "suggest_eps": num_stats.get("suggest_eps", np.nan),
                "pos%_x>0": num_stats.get("pos% (x>0)", np.nan),
                "neg%_x<0": num_stats.get("neg% (x<0)", np.nan),
                "zero%": num_stats.get("zero%", np.nan),
                "nan%": num_stats.get("nan%", np.nan),
                "std": num_stats.get("std", np.nan),
            }
            rows.append(row)
            print(f"— {mod_name}: non-hold {row['non_hold_%']}% | type={dtype} | suggest_eps={row['suggest_eps']}")
        except Exception:
            rows.append({"strategy": mod_name, "error": traceback.format_exc()})
            print(f"💥 {mod_name} failed to probe. See CSV for traceback.")

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / f"probe_{SYMBOL}_{TIMEFRAME}.csv"
    out.to_csv(out_path, index=False)
    print(f"\n✅ Report saved: {out_path}")

if __name__ == "__main__":
    main()
