import sys
import os
import json
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# 🛠️ إضافة المسار الجذر لمسارات الاستيراد
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.time_utils import format_timestamp

# 🔄 مسارات الملفات
LIVE_SIGNAL_PATH = "logs/live_signal.json"
TRADE_LOG_PATH = "logs/trade_log.json"

console = Console()

def load_json_file(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def show_live_signal():
    signal_data = load_json_file(LIVE_SIGNAL_PATH)
    if not signal_data:
        console.print("[yellow]🚫 لا توجد إشارة حالية[/yellow]")
        return

    table = Table(title="📡 الإشارة الحالية", show_lines=True)
    table.add_column("Symbol", style="cyan", justify="center")
    table.add_column("Signal", style="green", justify="center")
    table.add_column("Strength", style="magenta", justify="center")
    table.add_column("Entry", justify="center")
    table.add_column("Timestamp", justify="center")

    table.add_row(
        signal_data.get("symbol", "N/A"),
        signal_data.get("signal", "N/A"),
        signal_data.get("strength", "N/A"),
        str(signal_data.get("entry_price", "N/A")),
        format_timestamp(signal_data.get("timestamp", ""))
    )

    console.print(table)

def show_trade_log():
    trades = load_json_file(TRADE_LOG_PATH)
    if not trades:
        console.print("[yellow]📭 لا يوجد صفقات مسجلة بعد[/yellow]")
        return

    table = Table(title="📊 سجل الصفقات", show_lines=False)
    table.add_column("Time", style="dim")
    table.add_column("Symbol", style="cyan")
    table.add_column("Signal", style="green")
    table.add_column("Lot", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("SL", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("Result", style="bold", justify="center")

    for trade in trades[-10:]:  # آخر 10 صفقات
        table.add_row(
            format_timestamp(trade.get("timestamp", "")),
            trade.get("symbol", ""),
            trade.get("signal", ""),
            str(trade.get("lot", "")),
            str(trade.get("entry_price", "")),
            str(trade.get("sl", "")),
            str(trade.get("tp", "")),
            str(trade.get("result", ""))
        )

    console.print(table)

def run_dashboard(refresh_rate=10):
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            console.print(Panel("🧠 [bold cyan]AI Trading Dashboard[/bold cyan] ⏱️", expand=False))
            show_live_signal()
            console.print("")
            show_trade_log()
            time.sleep(refresh_rate)
    except KeyboardInterrupt:
        console.print("\n[red]📴 تم إغلاق الداشبورد[/red]")

if __name__ == "__main__":
    run_dashboard()
