# execution/execute_trade_mt5.py

import os
from datetime import datetime
import MetaTrader5 as mt5

from logs.logger import log_error
from indicators.atr import calculate_atr
from execution.trade_manager import TradeRiskGuard
from risk_management.position_sizing import lot_from_score_and_atr

# ========= إعدادات عامة =========
ATR_UNIT = "price"     # "price" إذا calculate_atr يرجّع بوحدة السعر | "points" إذا يرجّع بالنقاط
DEFAULT_DEVIATION = 20 # انزياح التنفيذ

# خريطة المخاطرة بناءً على قوة الإشارة (score)
RISK_MAP = {
    0.60: 0.05,   # Very Strong  → 5%
    0.45: 0.03,   # Strong       → 3%
    0.30: 0.02,   # Medium       → 2%
    0.20: 0.01,   # Light        → 1%
    0.00: 0.005,  # Weak         → 0.5%
}

# ========= أنشئ الـ Guard مرة واحدة =========
GUARD = TradeRiskGuard({
    "daily_stop_pct": -0.06,   # قاطع يومي: أوقف التداول إذا إجمالي اليوم ≤ -6%
    "cooldown_minutes": 30,    # الاستراحة بعد خسارة كبيرة
    "big_loss_pct": -0.05,     # خسارة كبيرة تعتبر 5% (متماشية مع Very Strong=5%)
    "max_trades_per_side": 2,  # أقصى صفقات بنفس الاتجاه/الرمز قبل كولداون
    "state_path": os.path.join("logs", "risk_state.json"),
    "timezone_offset_hours": 3 # عمّان
})

def _calc_sl_tp(signal: str,
                price: float,
                symbol_point: float,
                take_profit_ratio: float,
                atr_value: float,
                atr_multiplier: float,
                digits: int) -> (float, float):
    """
    يحسب SL/TP وفقًا لوحدة ATR:
      - ATR_UNIT="price": atr_value بوحدة السعر (مثال 1.2$)
      - ATR_UNIT="points": atr_value بالنقاط → نحوله للسعر عبر symbol_point
    """
    if ATR_UNIT == "points":
        stop_loss_price_units = (atr_value * atr_multiplier) * symbol_point
    else:
        stop_loss_price_units = atr_value * atr_multiplier

    if signal == "Buy":
        sl = price - stop_loss_price_units
        tp = price + stop_loss_price_units * take_profit_ratio
    else:
        sl = price + stop_loss_price_units
        tp = price - stop_loss_price_units * take_profit_ratio

    return round(sl, digits), round(tp, digits)


def execute_trade_mt5(
    symbol: str,
    signal: str,             # "Buy" أو "Sell"
    df,                      # DataFrame لحساب ATR
    score: float,            # -1..+1 (نستخدم القيمة المطلقة للتسعير)
    take_profit_ratio: float = 1.8,
    atr_multiplier: float = 1.5,
    use_dynamic_sizing: bool = True,
    fixed_lot: float = None
):
    """
    تنفيذ Market Order على MT5 مع:
      - SL/TP مبنية على ATR
      - بوابة Daily Stop & Cooldown قبل التنفيذ
      - Position Sizing ديناميكي من score + ATR + equity (أو استخدام lot ثابت)

    ملاحظات:
      - لو بدك ATR بالنقاط غيّر ATR_UNIT="points"
      - لو use_dynamic_sizing=False، لازم تمرّر fixed_lot
    """
    try:
        # فلتر الإشارة
        if signal not in ["Buy", "Sell"]:
            print("⏹️ لا يوجد تنفيذ لأن الإشارة Hold أو غير معروفة.")
            return

        # 🧱 بوابة الحماية (قاطع يومي + كولداون + حد الصفقات لكل اتجاه)
        can, reason = GUARD.can_trade(symbol=symbol, side=signal)
        if not can:
            print(f"⏹️ التنفيذ مرفوض: {reason}")
            return

        if not mt5.initialize():
            log_error("❌ فشل في تهيئة الاتصال مع MT5")
            return

        # الرمز
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            log_error(f"❌ لم يتم العثور على الرمز: {symbol}")
            return
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            log_error("❌ فشل في جلب Tick")
            return

        price  = tick.ask if signal == "Buy" else tick.bid
        point  = symbol_info.point
        digits = symbol_info.digits

        # بيانات الحساب (Equity)
        acc = mt5.account_info()
        if acc is None:
            log_error("❌ فشل في جلب معلومات الحساب")
            return
        equity_usd = float(acc.equity)

        # 🔥 ATR
        atr_value = calculate_atr(df)
        if not atr_value or atr_value == 0:
            log_error("❌ فشل في حساب ATR: القيمة صفر")
            return

        # SL / TP
        sl, tp = _calc_sl_tp(
            signal=signal,
            price=price,
            symbol_point=point,
            take_profit_ratio=take_profit_ratio,
            atr_value=atr_value,
            atr_multiplier=atr_multiplier,
            digits=digits
        )

        # اللوت
        if use_dynamic_sizing:
            lot = lot_from_score_and_atr(
                symbol=symbol,
                equity_usd=equity_usd,
                entry_price=price,
                atr_value=atr_value,
                atr_multiplier=atr_multiplier,
                score=score,
                atr_unit=ATR_UNIT,
                risk_map=RISK_MAP
            )
        else:
            lot = float(fixed_lot or 0.0)

        if lot <= 0:
            print("⏹️ تم إلغاء التنفيذ: حجم اللوت صفر/غير صالح.")
            return

        order_type = mt5.ORDER_TYPE_BUY if signal == "Buy" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": DEFAULT_DEVIATION,
            "magic": 123456,
            "comment": f"AI_Trade_{signal}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)
        if result is None:
            log_error("🚫 order_send أعاد None")
            return

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log_error(f"🚫 فشل تنفيذ الصفقة: {result.comment}")
        else:
            # ✅ حدّث عدّاد الصفقات عند الفتح
            GUARD.on_trade_open(symbol=symbol, side=signal)
            print(
                f"✅ [{datetime.now().strftime('%H:%M:%S')}] تنفيذ {signal} لـ {symbol} | "
                f"Lot: {lot} | SL: {sl} | TP: {tp} | Equity: {equity_usd:.2f} | Score: {score:.3f}"
            )

    except Exception as e:
        log_error(f"[❌] خطأ أثناء تنفيذ الصفقة: {e}")

    finally:
        mt5.shutdown()
