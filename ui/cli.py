# cli.py

import argparse
import pandas as pd
from strategy.signal_generator import predict_signal
from strategy.confirmation_rules import evaluate_signal_strength
from risk_management.position_sizing import calculate_position_size
from execution.execute_trade_mt5 import execute_trade_mt5
from indicators.atr import calculate_atr

def load_data_from_csv(path: str):
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return df

def main():
    parser = argparse.ArgumentParser(description="📊 بوت التداول CLI")

    parser.add_argument("--data", type=str, required=True, help="🗂️ مسار ملف CSV لبيانات الشموع")
    parser.add_argument("--symbol", type=str, required=True, help="🔤 رمز التداول (مثل XAUUSD)")
    parser.add_argument("--balance", type=float, default=1000, help="💰 رصيد الحساب")
    parser.add_argument("--risk", type=float, default=0.02, help="📉 نسبة المخاطرة (مثلاً 0.02 = 2%)")
    parser.add_argument("--execute", action="store_true", help="🚀 نفّذ الصفقة مباشرة في MT5")

    args = parser.parse_args()

    df = load_data_from_csv(args.data)
    signal_result = predict_signal(df)

    signal = signal_result["signal"]
    strength = signal_result["strength"]
    details = signal_result["details"]

    print(f"\n✅ Signal: {signal}")
    print(f"💪 Strength: {strength}")
    print(f"📊 Indicators: {details}")

    if signal != "Hold":
        price = details.get("price", df.iloc[-1]["close"])
        lot = calculate_position_size(args.balance, price, args.risk)
        stop_loss_pips = calculate_atr(df)

        print(f"\n📏 Lot: {lot:.2f} | SL (ATR): {stop_loss_pips:.2f} pips")

        if args.execute:
            execute_trade_mt5(args.symbol, signal, lot, stop_loss_pips)

if __name__ == "__main__":
    main()
