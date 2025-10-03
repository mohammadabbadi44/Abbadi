import pandas as pd
from pathlib import Path

SRC = Path("logs/diagnostics/probe_results.csv")
OUT = Path("strategy/forex/weights.py")

def base_weight(p):
    if 30 <= p <= 55: return 2.0
    if 20 <= p < 30 or 55 < p <= 70: return 1.5
    return 1.0

def main():
    df = pd.read_csv(SRC)
    rows = []
    for _, r in df.iterrows():
        w = base_weight(r["non_hold_%"])
        bias = abs(float(r["buy_%"]) - float(r["sell_%"]))
        if bias >= 10: w -= 0.25
        w = max(0.5, round(w, 2))
        rows.append((r["strategy"], w))
    text = "# generated from probe_results.csv\nWEIGHTS = {\n"
    for name, w in rows:
        text += f'    "{name}": {w},\n'
    text += "}\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"✅ wrote {OUT}")

if __name__ == "__main__":
    main()
