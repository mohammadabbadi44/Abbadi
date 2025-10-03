# strategy/signal_generator.py
# -*- coding: utf-8 -*-

import os
import sys
import importlib
import importlib.util
import inspect
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# ==== تأكد أن جذر المشروع على sys.path ====
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# === مؤشرات محلية ===
from indicators.ema import ema
from indicators.rsi import rsi
from indicators.macd import macd
from indicators.bollinger_bands import bollinger_bands
from indicators.volume_analysis import analyze_volume
from indicators.calculate_adx import calculate_adx
from indicators.calculate_atr import calculate_atr

# === SMC ===
from smc.order_blocks import detect_order_block
from smc.bos import detect_bos
from smc.liquidity_grabs import detect_liquidity_grab
from smc.support_resistance import identify_support_resistance

# === فيبوناتشي ===
from strategy.sk_fibonacci import detect_fibonacci_signal

# === عقل الـAI (مدمج محليًا)
from ai.predict import predict_from_df

# === اللوج ===
from logs.logger import log_error


# --------------------------
# إعدادات عامة
# --------------------------
REQUIRED_COLS = ["time", "open", "high", "low", "close", "volume"]
MAX_WINDOW = 300

# وزن تصويت المؤشرات الكلاسيكية (لكل صوت فردي)
W_CLASSIC = 1.0

# وزن تصويت المجمّع (دفعة الاستراتيجيات)
FOREX_ENSEMBLE_MIN_VOTES = 1.0  # أقل وزن لاعتبار قرار المجمّع
AI_WEIGHT = 2.0                 # وزن تصويت الـAI إن قال Buy/Sell

# --------------------------
# أدوات مساعدة للبيانات
# --------------------------
def _ensure_cols(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["open", "high", "low", "close"])

def _safe_last(x, default=np.nan):
    try:
        return x[-1]
    except Exception:
        return default

def _vote_strength(total_votes: float, ai_score: float, final_signal: str) -> str:
    """تقدير قوة الإشارة اعتمادًا على مجموع الأوزان + درجة الـAI."""
    if final_signal == "Hold":
        return "Weak"
    # حنحط سُلّم بسيط حسب الوزن الإجمالي
    # ملاحظة: الأوزان الآن قد تكون عشرية، فالتقسيم تقديري
    base = 0
    if total_votes <= 2.0:
        base = 0
    elif total_votes <= 3.5:
        base = 1
    elif total_votes <= 5.0:
        base = 2
    else:
        base = 3
    # بوست لو الـAI score عالي
    boost = 1 if ai_score >= 0.8 else 0
    scale = ["Weak", "Medium", "Strong", "Very Strong"]
    idx = min(len(scale) - 1, base + boost)
    return scale[idx]


# --------------------------
# تحميل أوزان واستراتيجيات الفوركس
# --------------------------
def _load_weights() -> Dict[str, float]:
    """
    يقرأ strategy/forex/weights.py إن وُجد.
    المفاتيح يجب أن تكون أسماء الملفات بدون .py
    """
    weights_path = os.path.join(os.path.dirname(__file__), "forex", "weights.py")
    if not os.path.exists(weights_path):
        return {}
    spec = importlib.util.spec_from_file_location("_forex_weights_", weights_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return getattr(mod, "WEIGHTS", {}) or {}

def _discover_forex_files() -> List[str]:
    """
    يجمع كل ملفات .py داخل strategy/forex/ باستثناء __init__.py و weights.py و ensemble.py
    """
    base_dir = os.path.join(os.path.dirname(__file__), "forex")
    found: List[str] = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            if f in ("__init__.py", "weights.py", "ensemble.py"):
                continue
            found.append(os.path.join(root, f))
    return found

def _import_module_from_file(file_path: str):
    name = "_fx_" + os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Can't load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def _pick_signal_fn(mod):
    """
    يختار دالة الإشارات:
    أولوية: get_signals -> generate_signal_series -> generate_signal
    وإلا أول دالة معرفة داخل الموديول.
    يجب أن تُرجع Series/ list/ أو قيمة نصية per-bar (أو إشارة واحدة fallback).
    """
    for name in ("get_signals", "generate_signal_series", "generate_signal"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if obj.__module__ == mod.__name__:
            return obj
    return None

def _run_strategy(fn, df: pd.DataFrame) -> pd.Series:
    """استدعاء متسامح وتوحيد الإخراج إلى Series بالقيم Buy/Sell/Hold."""
    try:
        sig = fn(df=df)
    except TypeError:
        try:
            sig = fn(df)
        except TypeError:
            sig = fn(df=df)

    if isinstance(sig, (list, tuple, np.ndarray)):
        sig = pd.Series(list(sig), index=df.index, dtype=object)
    elif not isinstance(sig, pd.Series):
        sig = pd.Series([str(sig)] * len(df), index=df.index, dtype=object)

    sig = (
        sig.astype(str)
        .str.strip()
        .str.lower()
        .map({"buy": "Buy", "long": "Buy", "sell": "Sell", "short": "Sell", "hold": "Hold"})
        .fillna("Hold")
    )
    return sig


# --------------------------
# الإشارة النهائية (تصويت موزون)
# --------------------------
def predict_signal(df: pd.DataFrame) -> Dict[str, Any]:
    """
    يدمج:
      - مؤشرات تقليدية (EMA/RSI/MACD/Bollinger/ADX/ATR/حجم)
      - SMC (OB/BOS/Liquidity + SR)
      - Fibonacci
      - استراتيجيات فوركس متعددة (موزونة عبر weights.py)
      - عقل AI داخلي (predict_from_df)
    ويخرج: signal, strength, score, details
    """
    try:
        if df is None or len(df) < 60:
            return {"signal": "Hold", "strength": "Weak", "score": 0.0, "details": {}}

        _ensure_cols(df)
        df = _to_numeric(df.tail(MAX_WINDOW)).reset_index(drop=True)
        if len(df) < 60:
            return {"signal": "Hold", "strength": "Weak", "score": 0.0, "details": {}}

        last = df.iloc[-1]
        price = float(last["close"])
        closes = df["close"].values

        details: Dict[str, Any] = {}

        # ===== مؤشرات أساسية =====
        ema20 = ema(closes, 20); ema50 = ema(closes, 50); ema200 = ema(closes, 200)
        e20 = float(_safe_last(ema20, price))
        e50 = float(_safe_last(ema50, price))
        e200 = float(_safe_last(ema200, price))
        details["ema20"] = e20; details["ema50"] = e50; details["ema200"] = e200

        # تصويت EMA بسيط
        details["ema"] = "Buy" if price > e20 else "Sell"

        # RSI
        rsi_arr = rsi(closes, 14)
        rsi_val = float(_safe_last(rsi_arr, 50.0))
        details["rsi_value"] = rsi_val
        if rsi_val < 30:   details["rsi"] = "Buy"
        elif rsi_val > 70: details["rsi"] = "Sell"
        else:              details["rsi"] = "Hold"

        # MACD
        try:
            macd_line, signal_line = macd(closes)
            details["macd"] = "Buy" if _safe_last(macd_line, 0) > _safe_last(signal_line, 0) else "Sell"
        except Exception:
            details["macd"] = "Hold"

        # Bollinger
        try:
            upper, middle, lower = bollinger_bands(closes)
            up, lo = float(_safe_last(upper, price)), float(_safe_last(lower, price))
            if price > up:      details["bollinger"] = "Sell"
            elif price < lo:    details["bollinger"] = "Buy"
            else:               details["bollinger"] = "Hold"
        except Exception:
            details["bollinger"] = "Hold"

        # حجم/ADX/ATR
        try:
            details["volume"] = analyze_volume(df)
        except Exception:
            details["volume"] = "Neutral"

        try:
            adx_val = float(calculate_adx(df))
        except Exception:
            adx_val = 0.0
        details["adx"] = adx_val

        try:
            atr_val = float(calculate_atr(df))
        except Exception:
            atr_val = 0.0
        details["atr"] = atr_val

        # ===== أدوات SMC =====
        try:
            details["order_block"] = detect_order_block(df)  # "Bullish"/"Bearish"/...
        except Exception:
            details["order_block"] = "None"

        try:
            details["bos"] = detect_bos(df)  # "Bullish BOS"/"Bearish BOS"/...
        except Exception:
            details["bos"] = "None"

        try:
            details["liquidity"] = detect_liquidity_grab(df)  # "Buy Trap"/"Sell Trap"/...
        except Exception:
            details["liquidity"] = "None"

        # دعم/مقاومة
        try:
            sr = identify_support_resistance(df)
        except Exception:
            sr = {}
        details["support_levels"]    = sr.get("support", [])
        details["resistance_levels"] = sr.get("resistance", [])
        details["last_support"]      = sr.get("last_support")
        details["last_resistance"]   = sr.get("last_resistance")

        # فيبو
        try:
            fib_signals = detect_fibonacci_signal(df)
            fib_last = fib_signals[-1] if isinstance(fib_signals, (list, pd.Series)) and len(fib_signals) else fib_signals
            details["fibonacci"] = fib_last if isinstance(fib_last, str) else "Hold"
        except Exception:
            details["fibonacci"] = "Hold"

        # ===== تصويت المؤشرات الكلاسيكية (موزونًا بالـ W_CLASSIC) =====
        vote_buy = 0.0
        vote_sell = 0.0
        core_votes = [details["ema"], details["rsi"], details["macd"], details["bollinger"]]
        vote_buy += W_CLASSIC * core_votes.count("Buy")
        vote_sell += W_CLASSIC * core_votes.count("Sell")

        # SMC وزن خفيف (0.5 لكل إشارة لصالح اتجاهها)
        if details["order_block"] == "Bullish":       vote_buy  += 0.5
        if details["bos"] == "Bullish BOS":           vote_buy  += 0.5
        if details["liquidity"] == "Sell Trap":       vote_buy  += 0.5

        if details["order_block"] == "Bearish":       vote_sell += 0.5
        if details["bos"] == "Bearish BOS":           vote_sell += 0.5
        if details["liquidity"] == "Buy Trap":        vote_sell += 0.5

        # قرب السعر من SR (0.3%)
        if details["last_support"] and abs(price - details["last_support"]) / price < 0.003:
            vote_buy += 0.5
        if details["last_resistance"] and abs(price - details["last_resistance"]) / price < 0.003:
            vote_sell += 0.5

        # فيبو
        if details["fibonacci"] == "Buy":   vote_buy  += 0.5
        if details["fibonacci"] == "Sell":  vote_sell += 0.5

        # ✳️ فلتر ADX: لو ترند قوي (>20) نعزّز الاتجاه الغالب +0.5
        if adx_val > 20:
            if vote_buy > vote_sell:   vote_buy  += 0.5
            elif vote_sell > vote_buy: vote_sell += 0.5

        # ✳️ انحياز الترند EMA200: خصم 0.5 من الاتجاه المعاكس
        trend_bias = "Bullish" if price >= e200 else "Bearish"
        details["trend_bias"] = trend_bias
        if trend_bias == "Bullish" and vote_sell > vote_buy:
            vote_sell = max(0.0, vote_sell - 0.5)
        elif trend_bias == "Bearish" and vote_buy > vote_sell:
            vote_buy = max(0.0, vote_buy - 0.5)

        # ===== دمج استراتيجيات الفوركس (تصويت موزون عبر weights.py) =====
        weights = _load_weights()
        forex_breakdown: Dict[str, Dict[str, Any]] = {}
        forex_files = _discover_forex_files()

        fx_buy = 0.0
        fx_sell = 0.0

        for fpath in forex_files:
            try:
                mod = _import_module_from_file(fpath)
                fn = _pick_signal_fn(mod)
                if fn is None:
                    continue
                sig_series = _run_strategy(fn, df)
                last_sig = str(sig_series.iloc[-1])
                # اسم الاستراتيجية بدون امتداد
                name = os.path.splitext(os.path.basename(fpath))[0]
                w = float(weights.get(name, 1.0))

                # طبّق الوزن
                if last_sig == "Buy":
                    fx_buy += w
                elif last_sig == "Sell":
                    fx_sell += w

                forex_breakdown[name] = {"signal": last_sig, "weight": w, "path": fpath}
            except Exception as ee:
                log_error(f"❌ Strategy load/run error ({fpath}): {ee}")

        details["forex_strategies"] = forex_breakdown
        details["forex_votes"] = {"buy": round(fx_buy, 3), "sell": round(fx_sell, 3), "total": round(max(fx_buy, fx_sell), 3)}

        # أدخل تصويت الاستراتيجيات ضمن الإجمالي
        # نعتبر أن هذا "كتلة" مستقلة؛ لا نطبّق حدًا أدنى، لكن ممكن تحط FOREX_ENSEMBLE_MIN_VOTES لاحقًا
        vote_buy  += fx_buy
        vote_sell += fx_sell

        # ===== دمج عقل الـAI (وزن ثابت AI_WEIGHT) =====
        ai_out = {}
        try:
            ai_out = predict_from_df(df) or {}
        except Exception as e:
            log_error(f"AI predict error: {e}")
            ai_out = {}

        ai_signal = ai_out.get("signal", "Hold")
        ai_score  = float(ai_out.get("score", 0.0))
        ai_strength = ai_out.get("strength", "Weak")

        details["ai_signal"]   = ai_signal
        details["ai_score"]    = ai_score
        details["ai_strength"] = ai_strength

        if ai_signal == "Buy":
            vote_buy  += AI_WEIGHT
        elif ai_signal == "Sell":
            vote_sell += AI_WEIGHT

        # ===== الإشارة النهائية =====
        # حط عتبة بسيطة للفارق، لتجنّب قرارات هشّة عند التعادل
        diff = vote_buy - vote_sell
        if diff > 0.75:
            final_signal = "Buy"
        elif diff < -0.75:
            final_signal = "Sell"
        else:
            final_signal = "Hold"

        total_votes = max(vote_buy, vote_sell)
        strength = _vote_strength(total_votes, ai_score, final_signal)

        # حزمة تفاصيل التصويت
        details["votes"] = {
            "classic_buy": round(W_CLASSIC * core_votes.count("Buy"), 3),
            "classic_sell": round(W_CLASSIC * core_votes.count("Sell"), 3),
            "fx_buy": round(fx_buy, 3),
            "fx_sell": round(fx_sell, 3),
            "ai_weight": AI_WEIGHT if ai_signal in ("Buy", "Sell") else 0.0,
            "total_buy": round(vote_buy, 3),
            "total_sell": round(vote_sell, 3),
            "edge": round(diff, 3),
        }

        return {
            "signal": final_signal,
            "strength": strength,
            "score": round(ai_score, 4),
            "details": details
        }

    except Exception as e:
        log_error(f"Signal Generation Error: {e}")
        return {"signal": "Hold", "strength": "Weak", "score": 0.0, "details": {"error": str(e)}}


# ===== اختبار يدوي =====
if __name__ == "__main__":
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "test_data.csv")
        df = pd.read_csv(csv_path)
        print(predict_signal(df))
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
