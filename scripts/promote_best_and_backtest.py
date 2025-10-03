# -*- coding: utf-8 -*-
"""
scripts/promote_best_and_backtest.py
- يلتقط آخر ملف best من reports/hybrid_ai_opt_best_*.yaml
- ينسخه إلى config/hybrid_ai.yaml
- يشغّل باكتيست run_hybrid_backtest.py للتأكيد
"""

import json, glob, os, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CONFIG  = ROOT / "config" / "hybrid_ai.yaml"

def latest_best():
    files = sorted(glob.glob(str(REPORTS / "hybrid_ai_opt_best_*.yaml")))
    if not files:
        print("❌ لا يوجد ملفات best. شغّل الأوبتمايزر أولاً.")
        sys.exit(1)
    return Path(files[-1])

def main():
    best_file = latest_best()
    print(f"⭐ Using best: {best_file}")

    # الملف مكتوب JSON (امتداد yaml لسهولة)، نحافظ على المحتوى كما هو
    data = best_file.read_text(encoding="utf-8")
    try:
        # تحقق سريع إن JSON صالح
        obj = json.loads(data)
    except Exception as e:
        print("❌ ملف best ليس JSON صالح:", e)
        sys.exit(1)

    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(data, encoding="utf-8")
    print(f"✅ Promoted to: {CONFIG}")

    # شغّل الباكتيست التأكيدي
    print("▶️ Running confirm backtest...")
    code = subprocess.call([sys.executable, str(ROOT / "scripts" / "run_hybrid_backtest.py")])
    sys.exit(code)

if __name__ == "__main__":
    main()
