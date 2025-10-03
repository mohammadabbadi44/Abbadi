# risk_management/position_sizing.py
"""
حساب حجم الصفقة (Lot) على MT5 بناءً على:
- نسبة المخاطرة من رصيد الحساب (risk%)
- مسافة وقف الخسارة (SL) من ATR أو من سعر محدد
- خصائص الرمز: min/max/step, tick_value, tick_size/point

المزايا:
- إدارة جلسة MT5 بدون shutdown في كل نداء (اختياري)
- كوابح مخاطرة وحدود منطقية لمسافة الستوب
- خارطة score→risk مرنة + كاب للمخاطرة
- تأثير قوة الإشارة (Weak/Medium/Strong/Very Strong) على المخاطرة
- كاب إجمالي للمخاطرة + حد أقصى لعدد الصفقات المفتوحة
- دالة تشخيص لإرجاع تفاصيل الحسابات
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import math
import MetaTrader5 as mt5

# ===== إعدادات افتراضية قابلة للتعديل =====
ATR_UNIT = "price"   # "price" لو ATR بالسعر | "points" لو ATR بالنقاط

# ✅ طلبك: 1% مخاطرة للصفقة الواحدة، 3 صفقات كحد أقصى ⇒ 3% كاب إجمالي
PER_TRADE_RISK_PCT = 0.01   # 1% لكل صفقة
TOTAL_RISK_CAP_PCT = 0.03   # 3% إجمالي
MAX_CONCURRENT_POS = 3      # أقصى صفقات مفتوحة بالتزامن

# كوابع عامة (ما عاد نحتاج 5% كسقف؛ نخليها = 1%)
MAX_RISK_PCT = PER_TRADE_RISK_PCT  # لا تتجاوز 1% إطلاقًا
MIN_RISK_PCT = 0.0
MIN_TICKS_SL = 2  # أقل مسافة ستوب = 2 تكّات

DEFAULT_RISK_MAP: Dict[float, float] = {
    # |score| >= threshold : risk_pct  (سيُقصّ تلقائيًا إلى 1%)
    0.80: 0.04,
    0.60: 0.03,
    0.45: 0.02,
    0.30: 0.01,
    0.00: 0.00,
}

# تأثير قوة الإشارة على المخاطرة (يضرب الـ risk_pct قبل الكوابح)
STRENGTH_FACTOR = {
    "Weak":         0.60,
    "Medium":       1.00,
    "Strong":       1.30,
    "Very Strong":  1.60,
}

@dataclass
class SizingInputs:
    symbol: str
    equity_usd: float                 # رصيد/إكويتي الحساب بالدولار
    entry_price: float                # سعر الدخول
    sl_price: Optional[float] = None  # سعر الستوب (لو محدد يَغلِب ATR)
    atr_value: Optional[float] = None # قيمة ATR
    atr_multiplier: float = 1.5
    risk_pct: Optional[float] = None  # نسبة مخاطرة مباشرة (سيتم قصّها إلى 1%)
    score: Optional[float] = None     # -1..+1 (نستخدم |score|)
    strength: Optional[str] = None    # "Weak" | "Medium" | "Strong" | "Very Strong"
    risk_map: Optional[Dict[float, float]] = None
    atr_unit: str = ATR_UNIT          # "price" أو "points"
    manage_mt5_session: bool = True   # لو True: نهيّئ MT5 إن لزم ولا نطفّيه هنا

# ========= أدوات مساعدة =========

def _ensure_mt5_ready(manage_session: bool) -> None:
    if manage_session:
        if not mt5.initialize():
            acc = mt5.account_info()
            if acc is None:
                raise RuntimeError("❌ فشل تهيئة MT5: لا يوجد اتصال بحساب.")

def _open_positions_count() -> int:
    """عدد الصفقات المفتوحة على الحساب (كل الرموز)."""
    poss = mt5.positions_get()
    return len(poss) if poss is not None else 0

def _normalize_lot(lot: float, symbol_info) -> float:
    """قص اللوت ضمن حدود الرمز واحترام خطوة اللوت."""
    min_lot = float(symbol_info.volume_min or 0.0)
    max_lot = float(symbol_info.volume_max or float("inf"))
    step = float(symbol_info.volume_step or 0.01)
    if lot <= 0:
        return 0.0
    lot = max(min_lot, min(lot, max_lot))
    steps = math.floor(lot / step + 1e-9)  # قص لأسفل على step
    return round(steps * step, 4)

def _distance_price_units(entry: float, sl: float) -> float:
    return abs(float(entry) - float(sl))

def _score_to_risk(score: float, risk_map: Dict[float, float]) -> float:
    s = abs(score)
    for thr in sorted(risk_map.keys(), reverse=True):
        if s >= thr:
            return risk_map[thr]
    return 0.0

def _apply_strength_factor(risk_pct: float, strength: Optional[str]) -> float:
    if not strength:
        return risk_pct
    factor = STRENGTH_FACTOR.get(str(strength), 1.0)
    return risk_pct * factor

def _calc_stop_distance_price_units(
    atr_value: Optional[float],
    atr_multiplier: float,
    entry_price: float,
    sl_price: Optional[float],
    symbol_point: float,
    tick_size: float,
    atr_unit: str,
) -> float:
    """يعيد مسافة الستوب بوحدة السعر. يضمن ≥ MIN_TICKS_SL * tick_size."""
    if sl_price is not None:
        dist = _distance_price_units(entry_price, sl_price)
    else:
        if not atr_value or atr_value <= 0:
            raise ValueError("ATR value is required if sl_price is not provided.")
        if atr_unit == "points":
            dist = (atr_value * atr_multiplier) * float(symbol_point)
        else:
            dist = float(atr_value) * float(atr_multiplier)

    # أقل مسافة منطقية = على الأقل عدد تِكّات محدد
    min_dist = max(MIN_TICKS_SL * float(tick_size), float(symbol_point))
    return max(dist, min_dist)

# ========= الحساب الأساسي =========

def _compute_lot_core(
    equity_usd: float,
    risk_pct: float,
    stop_distance_price: float,
    tick_size: float,
    tick_value: float,
) -> float:
    """
    المخاطرة بالدولار = lot * stop_distance_price * value_per_1_price_per_lot
    value_per_1_price_per_lot = (1 / tick_size) * tick_value
    """
    risk_usd = equity_usd * risk_pct
    if risk_usd <= 0:
        return 0.0
    if stop_distance_price <= 0 or tick_size <= 0 or tick_value <= 0:
        return 0.0

    value_per_1_price_per_lot = (1.0 / tick_size) * tick_value
    denom = stop_distance_price * value_per_1_price_per_lot
    if denom <= 0:
        return 0.0
    raw_lot = risk_usd / denom
    if not math.isfinite(raw_lot) or raw_lot > 1e6:
        return 0.0
    return float(raw_lot)

def calculate_position_size(inputs: SizingInputs) -> float:
    """
    يرجع اللوت فقط (float). لإخراج تفاصيل إضافية استخدم size_with_diagnostics.
    """
    lot, _diag = size_with_diagnostics(inputs)
    return lot

def size_with_diagnostics(inputs: SizingInputs) -> Tuple[float, Dict]:
    """
    يرجع (lot, diagnostics) لغايات الفحص والتتبّع.
    مطبّق:
      - 1% مخاطرة للصفقة الواحدة
      - 3 صفقات كحد أقصى
      - كاب إجمالي مخاطرة = 3%
      - قصّ المخاطرة إن كانت الميزانية المتبقية أقل من 1%
    """
    _ensure_mt5_ready(inputs.manage_mt5_session)

    # كوابح عدد الصفقات + ميزانية المخاطرة
    open_cnt = _open_positions_count()
    diagnostics: Dict = {"risk_policy": {
        "per_trade_risk_pct": PER_TRADE_RISK_PCT,
        "total_risk_cap_pct": TOTAL_RISK_CAP_PCT,
        "max_concurrent_positions": MAX_CONCURRENT_POS,
        "open_positions_now": open_cnt,
    }}

    if open_cnt >= MAX_CONCURRENT_POS:
        diagnostics["blocked_reason"] = "max_concurrent_positions_reached"
        return 0.0, diagnostics

    # المتبقي من الميزانية الإجمالية (مثلاً 3% - (صفقتين مفتوحتين × 1%) = 1%)
    remaining_budget_pct = TOTAL_RISK_CAP_PCT - open_cnt * PER_TRADE_RISK_PCT
    if remaining_budget_pct <= 0:
        diagnostics["blocked_reason"] = "total_risk_cap_exhausted"
        return 0.0, diagnostics

    info = mt5.symbol_info(inputs.symbol)
    if info is None:
        raise ValueError(f"❌ لم يتم العثور على الرمز: {inputs.symbol}")
    if not info.visible:
        mt5.symbol_select(inputs.symbol, True)

    # خصائص الرمز
    point = float(info.point or 0.0)
    tick_size = float(info.trade_tick_size or info.point or 0.0)
    tick_value = float(info.trade_tick_value or 0.0)

    if tick_size <= 0:
        raise ValueError("❌ tick_size غير صالح (0).")
    if tick_value <= 0:
        raise ValueError("❌ tick_value = 0 — لا يمكن حساب اللوت بدقة لهذا الرمز.")

    # تحديد risk_pct المدخلة/المستنتجة
    risk_map = inputs.risk_map or DEFAULT_RISK_MAP
    if inputs.risk_pct is not None:
        risk_pct = float(inputs.risk_pct)
    elif inputs.score is not None:
        risk_pct = _score_to_risk(float(inputs.score), risk_map)
    else:
        risk_pct = PER_TRADE_RISK_PCT

    # تأثير قوة الإشارة + القص على 1% + قص حسب الميزانية المتبقية
    risk_pct = _apply_strength_factor(risk_pct, inputs.strength)
    risk_pct = max(MIN_RISK_PCT, min(MAX_RISK_PCT, risk_pct))      # ≤ 1%
    risk_pct = min(risk_pct, remaining_budget_pct)                  # ≤ المتاح من الإجمالي

    # لو لا يوجد متاح كفاية لفتح صفقة جديدة، امنع
    if risk_pct <= 0:
        diagnostics["blocked_reason"] = "no_remaining_risk_budget"
        diagnostics["remaining_budget_pct"] = remaining_budget_pct
        return 0.0, diagnostics

    # مسافة الستوب
    stop_distance_price = _calc_stop_distance_price_units(
        atr_value=inputs.atr_value,
        atr_multiplier=inputs.atr_multiplier,
        entry_price=inputs.entry_price,
        sl_price=inputs.sl_price,
        symbol_point=point,
        tick_size=tick_size,
        atr_unit=inputs.atr_unit,
    )

    raw_lot = _compute_lot_core(
        equity_usd=float(inputs.equity_usd),
        risk_pct=risk_pct,
        stop_distance_price=stop_distance_price,
        tick_size=tick_size,
        tick_value=tick_value,
    )

    lot = _normalize_lot(raw_lot, info)

    # مخرجات تشخيصية
    diagnostics.update({
        "symbol": inputs.symbol,
        "equity_usd": float(inputs.equity_usd),
        "entry_price": float(inputs.entry_price),
        "risk_pct_used": risk_pct,
        "remaining_budget_pct": remaining_budget_pct,
        "stop_distance_price": stop_distance_price,
        "tick_size": tick_size,
        "tick_value": tick_value,
        "value_per_1_price_per_lot": (1.0 / tick_size) * tick_value if tick_size > 0 else None,
        "raw_lot": raw_lot,
        "normalized_lot": lot,
        "symbol_limits": {
            "min_lot": info.volume_min,
            "max_lot": info.volume_max,
            "lot_step": info.volume_step,
        },
        "params": {
            "atr_value": inputs.atr_value,
            "atr_multiplier": inputs.atr_multiplier,
            "sl_price": inputs.sl_price,
            "atr_unit": inputs.atr_unit,
            "score": inputs.score,
            "strength": inputs.strength,
        },
    })

    return lot, diagnostics

# ========= واجهات مختصرة =========

def lot_from_score_and_atr(
    symbol: str,
    equity_usd: float,
    entry_price: float,
    atr_value: float,
    atr_multiplier: float,
    score: float,
    strength: Optional[str] = None,
    atr_unit: str = ATR_UNIT,
    risk_map: Optional[Dict[float, float]] = None,
    manage_mt5_session: bool = True,
) -> float:
    inputs = SizingInputs(
        symbol=symbol,
        equity_usd=equity_usd,
        entry_price=entry_price,
        atr_value=atr_value,
        atr_multiplier=atr_multiplier,
        score=score,
        strength=strength,
        risk_map=risk_map or DEFAULT_RISK_MAP,
        atr_unit=atr_unit,
        manage_mt5_session=manage_mt5_session,
    )
    return calculate_position_size(inputs)

def lot_from_fixed_risk_and_prices(
    symbol: str,
    equity_usd: float,
    entry_price: float,
    sl_price: float,
    risk_pct: float,
    strength: Optional[str] = None,
    manage_mt5_session: bool = True,
) -> float:
    # يمكن تطبيق عامل القوة حتى مع مخاطرة ثابتة (ثم يُقصّ إلى 1% وكاب الإجمالي)
    adj_risk_pct = _apply_strength_factor(float(risk_pct), strength) if strength else risk_pct
    inputs = SizingInputs(
        symbol=symbol,
        equity_usd=equity_usd,
        entry_price=entry_price,
        sl_price=sl_price,
        risk_pct=adj_risk_pct,
        manage_mt5_session=manage_mt5_session,
    )
    return calculate_position_size(inputs)
