# tools/hit_api.py
import argparse
import json
import sys
from typing import List, Dict

import pandas as pd
import requests


def detect_time_column(df: pd.DataFrame) -> str:
    """يحاول يكتشف عمود الوقت: time / timestamp / ts"""
    for c in ["time", "timestamp", "ts"]:
        if c in df.columns:
            return c
    raise ValueError("لم يتم العثور على عمود وقت: جرّب time أو timestamp أو ts")


def build_payload(df: pd.DataFrame, limit: int) -> Dict[str, List[Dict]]:
    """يبني JSON جاهز للإرسال لـ /predict"""
    df = df.tail(limit).reset_index(drop=True)

    # تأكد الأعمدة المطلوبة موجودة
    req_cols = {"open", "high", "low", "close", "volume"}
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"أعمدة ناقصة في CSV: {missing}. لازم يكون عندك: {sorted(req_cols)}")

    tcol = detect_time_column(df)

    candles = []
    for _, row in df.iterrows():
        candles.append(
            {
                "time": str(row[tcol]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]) if not pd.isna(row["volume"]) else 0.0,
            }
        )

    return {"candles": candles}


def main():
    ap = argparse.ArgumentParser(description="Send CSV candles to /predict endpoint.")
    ap.add_argument("--csv", required=True, help="مسار ملف CSV (لازم يحتوي: time/timestamp/ts + open,high,low,close,volume)")
    ap.add_argument("--url", default="http://127.0.0.1:8000/predict", help="رابط /predict (افتراضي محلي)")
    ap.add_argument("--limit", type=int, default=220, help="عدد الشموع الأخيرة المرسلة (افتراضي 220)")
    ap.add_argument("--min-window", type=int, default=60, help="قيمة min_window للـendpoint (اختياري)")
    ap.add_argument("--timeout", type=int, default=15, help="مهلة HTTP بالثواني")
    ap.add_argument("--pretty", action="store_true", help="طباعة جميلة للنتيجة")
    ap.add_argument("--save", default="", help="مسار ملف لحفظ الاستجابة JSON (اختياري)")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"❌ فشل قراءة CSV: {e}")
        sys.exit(1)

    try:
        payload = build_payload(df, args.limit)
    except Exception as e:
        print(f"❌ خطأ في تجهيز البودّي: {e}")
        sys.exit(1)

    try:
        # مرر min_window عبر QueryString لو موجود
        url = args.url
        if args.min_window:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}min_window={args.min_window}"

        r = requests.post(url, json=payload, timeout=args.timeout)
    except Exception as e:
        print(f"❌ فشل الاتصال بالسيرفر: {e}")
        sys.exit(1)

    print(f"HTTP {r.status_code}")
    content = None
    try:
        content = r.json()
    except Exception:
        # لو مو JSON اطبع نص أول 1000 حرف
        print(r.text[:1000])
        sys.exit(0)

    if args.pretty:
        print(json.dumps(content, ensure_ascii=False, indent=2))
    else:
        print(content)

    if args.save:
        try:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
            print(f"💾 حفظت الاستجابة في: {args.save}")
        except Exception as e:
            print(f"⚠️ فشل حفظ الاستجابة: {e}")


if __name__ == "__main__":
    main()
