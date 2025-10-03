# diagnostics/smoke_strategies.py
# -*- coding: utf-8 -*-

import os, sys, argparse
import numpy as np
import pandas as pd

# ====== اضبط مسار المشروع عشان يلاقي strategy/* ======
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from strategy.load_forex_strategies import load_all_forex_strategies  # noqa: E402


def _read_csv_any(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # حاول نطابق عمود الوقت
    time_col = None
    for c in ("time", "timestamp", "date", "datetime"):
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError("No time column found. Expected one of: time, timestamp, date, datetime")

    # Parse dates
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if df[time_col].isna().all():
        raise ValueError(f"Failed to parse datetime column: {time_col}")

    df = df.dropna(subset=[time_col]).reset_index(drop=True).set_index(time_col)

    # تأكد من الأعمدة الأساسية
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # املأ volume لو ناقص
    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df


def _coverage(series: pd.Series) -> dict:
    n = len(series)
    hold = int((series == "Hold").sum())
    buy = int((series == "Buy").sum())
    sell = int((series == "Sell").sum())
    non_hold_pct = 100.0 * (n - hold) / max(1, n)
    return {
        "buy": buy,
        "sell": sell,
        "hold": hold,
        "non_hold_%": round(non_hold_pct, 2),
        "dtype": str(series.dtype),
        "zero%": round(100.0 * (series == "Hold").mean(), 2),
    }


def print_table(results: dict):
    # results: {name: Series}
    rows = []
    for name, s in results.items():
        try:
            cov = _coverage(s.astype("string"))
        except Exception:
            cov = {"buy": "-", "sell": "-", "hold": "-", "non_hold_%": "-", "dtype": "ERR", "zero%": "-"}
        rows.append({
            "strategy": name,
            "dtype": cov["dtype"],
            "buy": cov["buy"],
            "sell": cov["sell"],
            "hold": cov["hold"],
            "non_hold_%": cov["non_hold_%"],
            "zero%": cov["zero%"],
        })
    df = pd.DataFrame(rows).sort_values(by=["non_hold_%", "strategy"], ascending=[False, True])
    # طباعة نظيفة
    colw = {c: max(len(c), df[c].astype(str).map(len).max() if not df.empty else len(c)) for c in df.columns}
    header = "  ".join(c.ljust(colw[c]) for c in df.columns)
    print("\n=== Strategy Signals Coverage ===")
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        print("  ".join(str(r[c]).ljust(colw[c]) for c in df.columns))


def main():
    ap = argparse.ArgumentParser(description="Smoke test for strategy signals (no backtest).")
    ap.add_argument("--csv", required=False, default=os.path.join(PROJECT_ROOT, "data", "historical", "XAUUSD_30m.csv"),
                    help="Path to CSV with columns: time/timestamp + open,high,low,close[,volume]")
    args = ap.parse_args()

    print(f"🔍 Scanning strategy folder: {os.path.join(PROJECT_ROOT, 'strategy', 'forex')}")
    df = _read_csv_any(args.csv)

    # تحميل الاستراتيجيات كلها (اللودر عندك ممكن يستخدم multiprocessing داخليًا)
    signals = load_all_forex_strategies(df)

    # اطبع أي ستراتجي ما إلها دالة أو رجّعت كلّها Hold
    missing = [k for k, s in signals.items() if isinstance(s, pd.Series) and s.eq("Hold").all()]
    if missing:
        print("\n⚠️ Strategies returning ALL 'Hold' (needs thresholds or generate_signal):")
        for k in missing:
            print(f"   - {k}")

    print_table(signals)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # مهم لو اللودر يستخدم multiprocessing على Windows
    main()
