# execution/trade_manager.py
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Optional

# إعدادات افتراضية قابلة للتعديل
DEFAULTS = {
    "daily_stop_pct": -0.06,      # قاطع يومي: لو PnL اليوم <= -6% وقف التداول لباقي اليوم
    "cooldown_minutes": 30,       # مدة الاستراحة بعد خسارة كبيرة
    "big_loss_pct": -0.05,        # خسارة تُعد كبيرة (تفَعِّل الكولداون)
    "max_trades_per_side": 2,     # حد أقصى لصفقات نفس الاتجاه على نفس الرمز قبل كولداون
    "state_path": "logs/risk_state.json",
    "timezone_offset_hours": 3,   # عمان +03:00
}

def now_local(offset_hours: int = 3) -> datetime:
    return datetime.utcnow() + timedelta(hours=offset_hours)

class TradeRiskGuard:
    """
    يدير:
      - القاطع اليومي (Daily Stop)
      - الاستراحة الإجبارية (Cooldown)
      - عدّاد الصفقات لكل رمز/اتجاه
      - تخزين الحالة JSON للاستمرارية بعد إعادة التشغيل
    """
    def __init__(self, cfg: Dict = None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.state = self._load_state()
        self._ensure_today_bucket()

    # === واجهة عامة ===
    def can_trade(self, symbol: str, side: str) -> (bool, str):
        self._roll_day_if_needed()

        if self.state["trading_disabled_today"]:
            return False, "daily_stop_triggered"

        cd_until = self._cooldown_until()
        if cd_until and now_local(self.cfg["timezone_offset_hours"]) < cd_until:
            return False, f"cooldown_active_until_{cd_until.strftime('%H:%M:%S')}"

        key = self._side_key(symbol, side)
        if self.state["counts"].get(key, 0) >= self.cfg["max_trades_per_side"]:
            return False, "max_trades_per_side_reached"

        return True, "ok"

    def on_trade_open(self, symbol: str, side: str):
        self._roll_day_if_needed()
        key = self._side_key(symbol, side)
        self.state["counts"][key] = self.state["counts"].get(key, 0) + 1
        self._save_state()

    def on_trade_close(self, pnl_pct: float):
        """
        pnl_pct: نسبة من الرصيد (مثال -0.05 = -5%, +0.012 = +1.2%)
        """
        self._roll_day_if_needed()
        self.state["today_pnl_pct"] += pnl_pct

        if pnl_pct <= self.cfg["big_loss_pct"]:
            self._start_cooldown(minutes=self.cfg["cooldown_minutes"])

        if self.state["today_pnl_pct"] <= self.cfg["daily_stop_pct"]:
            self.state["trading_disabled_today"] = True

        self._save_state()

    def force_cooldown(self, minutes: int):
        self._start_cooldown(minutes=minutes)
        self._save_state()

    def reset_day(self):
        self._new_day_reset()
        self._save_state()

    # === داخلي ===
    def _side_key(self, symbol: str, side: str) -> str:
        return f"{symbol.upper()}_{side.upper()}"

    def _cooldown_until(self) -> Optional[datetime]:
        ts = self.state.get("cooldown_until")
        if not ts:
            return None
        return datetime.fromisoformat(ts)

    def _start_cooldown(self, minutes: int):
        until = now_local(self.cfg["timezone_offset_hours"]) + timedelta(minutes=minutes)
        self.state["cooldown_until"] = until.isoformat()

    def _roll_day_if_needed(self):
        local_today = now_local(self.cfg["timezone_offset_hours"]).date()
        if self.state.get("date") != str(local_today):
            self._new_day_reset()

    def _new_day_reset(self):
        self.state["date"] = str(now_local(self.cfg["timezone_offset_hours"]).date())
        self.state["today_pnl_pct"] = 0.0
        self.state["trading_disabled_today"] = False
        self.state["cooldown_until"] = None
        self.state["counts"] = {}

    def _ensure_today_bucket(self):
        if "date" not in self.state:
            self._new_day_reset()

    # === تخزين الحالة ===
    def _load_state(self) -> Dict:
        path = self.cfg["state_path"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "date": None,
            "today_pnl_pct": 0.0,
            "trading_disabled_today": False,
            "cooldown_until": None,
            "counts": {}
        }

    def _save_state(self):
        path = self.cfg["state_path"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
