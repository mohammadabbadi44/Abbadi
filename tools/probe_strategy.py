# tools/probe_strategy.py
# -*- coding: utf-8 -*-

# === ensure project root on sys.path ===
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# =======================================

import argparse
import importlib.util
import inspect
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable

import pandas as pd
import numpy as np


# ----------------------- IO helpers -----------------------

def load_module_from_file(file_path: Path, module_name: Optional[str] = None):
    """حمّل موديول بايثون من ملف .py"""
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))
    if module_name is None:
        module_name = f"_probe_{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Can't load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def pick_signal_fn(mod):
    """اختيار دالة الإشارات بترتيب أولوية مع fallback محترم."""
    for name in ("get_signals", "generate_signal_series", "generate_signal"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    # fallback: أول دالة معرفة داخل الموديول نفسه
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if obj.__module__ == mod.__name__:
            return obj
    return None


# ----------------------- Data helpers -----------------------

CANON = ["time", "open", "high", "low", "close", "volume"]
ALT_TIME = ["time", "timestamp", "ts", "date", "datetime"]

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    mapping: Dict[str, str] = {}
    lower = {c.lower(): c for c in df.columns}

    # time
    for t in ALT_TIME:
        if t in lower:
            mapping[lower[t]] = "time"
            break

    # prices + volume
    for nm in ["open", "high", "low", "close", "volume"]:
        if nm in lower:
            mapping[lower[nm]] = nm
        else:
            for alt in (nm.capitalize(), nm.upper(), "vol" if nm == "volume" else nm):
                if alt in df.columns:
                    mapping[alt] = nm
                    break

    out = df.rename(columns=mapping).copy()
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out.get(c), errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out.get("volume"), errors="coerce").fillna(0.0)

    if "time" not in out.columns:
        out["time"] = np.arange(len(out))

    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def parse_params(kvs: Optional[Iterable[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kv in (kvs or []):
        if "=" not in kv: 
            continue
        k, v = kv.split("=", 1)
        vv = v.strip()
        if vv.lower() in ("true", "false"):
            out[k] = (vv.lower() == "true")
            continue
        try:
            if "." in vv or "e" in vv.lower():
                out[k] = float(vv)
            else:
                out[k] = int(vv)
        except Exception:
            out[k] = vv
    return out


def as_series(sig, index) -> pd.Series:
    if isinstance(sig, (list, tuple, np.ndarray)):
        sig = pd.Series(list(sig), index=index, dtype=object)
    elif not isinstance(sig, pd.Series):
        sig = pd.Series([str(sig)] * len(index), index=index, dtype=object)
    # تطبيع النص
    sig = sig.astype(str).str.strip().str.lower().map({
        "buy": "Buy", "long": "Buy",
        "sell": "Sell", "short": "Sell",
        "hold": "Hold", "": "Hold", "nan": "Hold"
    }).fillna("Hold")
    return sig


def signal_stats(sig: pd.Series) -> Dict[str, Any]:
    n = int(len(sig))
    vc = sig.value_counts()
    b = int(vc.get("Buy", 0)); s = int(vc.get("Sell", 0)); h = int(vc.get("Hold", 0))
    pct = lambda x: (x/n*100.0) if n else 0.0
    return {
        "total": n,
        "buy": b, "sell": s, "hold": h,
        "non_hold_%": round(pct(b+s), 2),
        "buy_%": round(pct(b), 2),
        "sell_%": round(pct(s), 2),
    }


# ----------------------- Runner -----------------------

DEFAULT_SUBDIRS = [
    "breakout",
    "mean_reversion",
    "momentum",
    "scalping",
    "trend_following",
    "volatility",
    "volume",
]

def run_one(file_path: Path, df: pd.DataFrame, params: Dict[str, Any], save_samples_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    try:
        # تجاهل init
        if file_path.name == "__init__.py":
            return None

        mod = load_module_from_file(file_path)
        fn  = pick_signal_fn(mod)
        if fn is None:
            print(f"[⚠️] لا توجد دالة إشارات في: {file_path}")
            return None

        # استدعاء مرن
        try:
            sig = fn(df=df, params=params)
        except TypeError:
            try:
                sig = fn(df, params=params)
            except TypeError:
                try:
                    sig = fn(df)
                except TypeError:
                    sig = fn(df=df)

        sig = as_series(sig, df.index)
        st  = signal_stats(sig)

        st["strategy"] = file_path.stem
        st["path"] = str(file_path.as_posix())

        # حفظ عينة آخر 200 شمعة
        if save_samples_dir is not None:
            save_samples_dir.mkdir(parents=True, exist_ok=True)
            smp = df.tail(200).copy()
            smp["signal"] = sig.tail(200).values
            (save_samples_dir / f"{file_path.stem}_sample.csv").write_text(smp.to_csv(index=False), encoding="utf-8")

        return st

    except Exception as e:
        print(f"[❌] {file_path}: {e}")
        traceback.print_exc()
        return None


def collect_files(root: Path, subdirs: List[str], glob: str) -> List[Path]:
    files: List[Path] = []
    for sd in subdirs:
        base = root / sd
        if base.exists():
            files += [p for p in base.rglob(glob) if p.is_file()]
    # فلتر بسيط: تجنّب ملفات cache
    files = [f for f in files if "__pycache__" not in f.as_posix()]
    return sorted(files)


# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser(description="افحص كل استراتيجيات الفوركس تلقائيًا ويطلعك تقرير واحد.")
    ap.add_argument("--csv", required=True, help="مسار CSV (OHLCV)")
    ap.add_argument("--root", default="strategy/forex", help="جذر الاستراتيجيات (افتراضي strategy/forex)")
    ap.add_argument("--dirs", nargs="*", default=DEFAULT_SUBDIRS, help="قائمة المجلدات الفرعية المراد فحصها")
    ap.add_argument("--glob", default="*.py", help="نمط الملفات (افتراضي *.py)")
    ap.add_argument("--limit", type=int, default=5000, help="عدد الشموع من الذيل (افتراضي 5000)")
    ap.add_argument("--param", action="append", help="باراميتر بشكل key=value (كرر الخيار)")
    ap.add_argument("--out", default="logs/diagnostics/probe_results.csv", help="ملف نتائج CSV")
    ap.add_argument("--samples_dir", default="logs/diagnostics/probe_samples", help="مجلد عينات الإشارات")
    args = ap.parse_args()

    # حمل البيانات
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")
    df_raw = pd.read_csv(csv_path)
    df = normalize_ohlcv(df_raw).tail(args.limit).reset_index(drop=True)

    params = parse_params(args.param)
    root = Path(args.root)
    files = collect_files(root, args.dirs, args.glob)

    if not files:
        print(f"[⚠️] لا يوجد ملفات مطابقة تحت {root} في {args.dirs}")
        return

    print(f"🧭 Root: {root} | Targets: {len(files)} files")
    rows: List[Dict[str, Any]] = []
    samples_dir = Path(args.samples_dir)

    for fp in files:
        print(f"🔍 {fp}")
        st = run_one(fp, df, params, samples_dir)
        if st:
            rows.append(st)

    if not rows:
        print("⚠️ لم يتم توليد أي نتائج.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["non_hold_%", "strategy"], ascending=[False, True]).to_csv(out_path, index=False)
    print(f"\n✅ تم حفظ النتائج في: {out_path}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
