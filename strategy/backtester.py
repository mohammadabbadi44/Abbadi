# backtest_all_strategies.py

import os
import sys
import pandas as pd

# إضافة المسار الرئيسي
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy.load_forex_strategies import load_all_forex_strategies
from logs.logger import log_trade, log_error
from risk_management.position_sizing import calculate_lot_size

def run_backtest_for_all_strategies(csv_path, symbol="XAUUSD", balance=10000):
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])

    strategies = load_all_forex_strategies(df)

    if not strategies:
        print("❌ لم يتم تحميل أي استراتيجية.")
        return

    results = {}

    for strategy_name, signals in strategies.items():
        equity = balance
        wins, losses = 0, 0
        print(f"\n🔁 Backtest لـ [{strategy_name}] على {symbol} | {len(df)} شموع")

        for i in range(len(signals) - 1):
            try:
                signal = signals[i]
                if signal not in ["Buy", "Sell"]:
                    continue

                current_price = df.at[i, 'close']
                next_price = df.at[i + 1, 'close'] if i + 1 < len(df) else current_price

                lot, risk_percent = calculate_lot_size(equity, signal, strength="Medium")

                # SL/TP وهمي: 0.5% SL, 0.75% TP
                sl = current_price * (1 - 0.005) if signal == "Buy" else current_price * (1 + 0.005)
                tp = current_price * (1 + 0.0075) if signal == "Buy" else current_price * (1 - 0.0075)

                result = "win" if (signal == "Buy" and next_price >= tp) or (signal == "Sell" and next_price <= tp) else "loss"
                equity += (risk_percent / 100 * balance) if result == "win" else -(risk_percent / 100 * balance)

                wins += 1 if result == "win" else 0
                losses += 1 if result == "loss" else 0

                log_trade(symbol, signal, lot, current_price, sl, tp, result)

            except Exception as e:
                log_error(f"❌ خطأ في استراتيجية {strategy_name} على شمعة {i}: {e}")

        results[strategy_name] = {
            "final_equity": round(equity, 2),
            "wins": wins,
            "losses": losses,
            "win_rate": round(100 * wins / (wins + losses + 1e-9), 2)
        }

    print("\n📊 ملخص النتائج لكل استراتيجية:")
    for name, res in results.items():
        print(f"📈 {name}: Equity=${res['final_equity']} | ✅ {res['wins']} | ❌ {res['losses']} | 🎯 WinRate={res['win_rate']}%")

if __name__ == "__main__":
    csv_path = "data/historical/XAUUSD_5m.csv"
    run_backtest_for_all_strategies(csv_path)
