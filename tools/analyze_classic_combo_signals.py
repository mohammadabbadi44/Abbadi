import pandas as pd
import os
import sys
import importlib.util

# ✅ تحميل ملف الاستراتيجية من المسار الصحيح
strategy_path = os.path.abspath("strategy/classic_combo.py")
spec = importlib.util.spec_from_file_location("classic_combo", strategy_path)
strategy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy_module)
generate_signal = strategy_module.generate_signal

# 🟡 إعدادات البيانات
symbol = "XAUUSD"
timeframe = "15m"
csv_path = f"data/historical/{symbol}_{timeframe}.csv"

print("📊 Loading historical data...")
df = pd.read_csv(csv_path)
df['time'] = pd.to_datetime(df['time'])

# 🔁 توليد الإشارات لكل شمعة
signals = []
for i in range(50, len(df)):
    window = df.iloc[:i+1]
    signal = generate_signal(window)
    signals.append({
        "time": df.iloc[i]['time'],
        "signal": signal
    })

result_df = pd.DataFrame(signals)

# 📈 تلخيص عدد الإشارات
summary = result_df['signal'].value_counts().to_frame().rename(columns={"signal": "count"})
summary["percentage"] = (summary["count"] / len(result_df) * 100).round(2)

# 💾 حفظ النتائج في ملفات CSV
os.makedirs("logs/debug_signals", exist_ok=True)
result_df.to_csv("logs/debug_signals/classic_combo_signals.csv", index=False)
summary.to_csv("logs/debug_signals/classic_combo_summary.csv")

print("\n✅ Signal Summary:")
print(summary)
