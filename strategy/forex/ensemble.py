# strategy/forex/ensemble.py
"""
Ensemble لمجموعة استراتيجيات الفوركس داخل strategy/forex
- يكتشف كل الموديولات التي فيها generate_signal(df)
- يشغّلها بوقت أقصى لكل وحدة (timeout)
- يطبع الإشارات (Buy/Sell/Hold) ويجمعها بأوزان configurable
- يرجّع إشارة موحّدة + تفاصيل

الاستخدام السريع:
    from strategy.forex.ensemble import ensemble_signal
    out = ensemble_signal(df, min_votes=2)
    print(out["signal"], out["votes"])

ملاحظات:
- لو عندك ملف weights.py بجانب هذا الملف وفيه dict WEIGHTS = {"rsi_trend": 2, ...}
  رح تُقرأ الأوزان تلقائيًا. الافتراضي 1 لكل استراتيجية.
"""

import os
import importlib
import pkgutil
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# إعدادات قابلة للتعديل
TIMEOUT_SEC   = 0.8   # حد أقصى لكل استراتيجية
AI_WEIGHT     = 0     # هذا المجمّع مستقل عن AI؛ خلّيه 0 (الـAI يتعالج في signal_generator)
DEFAULT_WEIGHT = 1
MIN_VOTES_DEFAULT = 2  # حد أدنى للأصوات لتبنّي Buy/Sell

# كاش للأسماء/الدوال
_DISCOVERY_CACHE: Optional[List[Tuple[str, Any]]] = None
_WEIGHTS_CACHE: Optional[Dict[str, float]] = None

def _load_weights() -> Dict[str, float]:
    """يحاول يقرأ WEIGHTS من weights.py وإلا يرجّع dict فاضي (وزن 1 افتراضي)."""
    global _WEIGHTS_CACHE
    if _WEIGHTS_CACHE is not None:
        return _WEIGHTS_CACHE

    weights: Dict[str, float] = {}
    try:
        # ابحث ملف weights.py في نفس المجلد
        module_path = __name__.rsplit(".", 1)[0] + ".weights"  # strategy.forex.weights
        mod = importlib.import_module(module_path)
        if hasattr(mod, "WEIGHTS") and isinstance(mod.WEIGHTS, dict):
            for k, v in mod.WEIGHTS.items():
                try:
                    weights[str(k)] = float(v)
                except Exception:
                    pass
    except Exception:
        pass

    _WEIGHTS_CACHE = weights
    return weights

def discover_strategies() -> List[Tuple[str, Any]]:
    """يكتشف كل الاستراتيجيات داخل strategy/forex اللي عندها generate_signal(df)."""
    global _DISCOVERY_CACHE
    if _DISCOVERY_CACHE is not None:
        return _DISCOVERY_CACHE

    strategies: List[Tuple[str, Any]] = []
    base_pkg = "strategy.forex"
    base_dir = os.path.dirname(__file__)

    for _, name, ispkg in pkgutil.walk_packages([base_dir], prefix=base_pkg + "."):
        if ispkg:
            continue
        try:
            module = importlib.import_module(name)
            fn = getattr(module, "generate_signal", None)
            if callable(fn):
                # اسم قصير للاستراتيجية
                short = name.split(".")[-1]
                strategies.append((short, fn))
        except Exception:
            continue

    _DISCOVERY_CACHE = strategies
    return strategies

def _run_one(fn, df):
    """شغّل استراتيجية وأرجع نص الإشارة فقط، أو None عند الفشل."""
    try:
        out = fn(df)
        if isinstance(out, str):
            return out  # "Buy"/"Sell"/"Hold"
        if isinstance(out, dict):
            # لو رجّعت dict وفيه signal
            sig = out.get("signal")
            if isinstance(sig, str):
                return sig
        return None
    except Exception:
        return None

def _norm_signal(sig: Optional[str]) -> str:
    s = (sig or "").strip().lower()
    if s in ("buy", "long"):
        return "Buy"
    if s in ("sell", "short"):
        return "Sell"
    return "Hold"

def _vote_to_strength(total: float) -> str:
    # تحويل عدد الأصوات لقوة تقريبية مستقلة عن AI (للإرجاع فقط)
    if total <= 1:
        return "Weak"
    if total <= 3:
        return "Medium"
    if total <= 5:
        return "Strong"
    return "Very Strong"

def ensemble_signal(
    df,
    min_votes: int = MIN_VOTES_DEFAULT,
    timeout_sec: float = TIMEOUT_SEC,
    include_details: bool = True,
) -> Dict[str, Any]:
    """
    يشغّل كل استراتيجيات forex المعرّفة، ويعيد:
    {
      "signal": "Buy|Sell|Hold",
      "votes": {"buy": x, "sell": y, "total": max(x,y)},
      "strength": "...",
      "details": { "per_strategy": {name: "Buy"/"Sell"/"Hold"}, "weights": {...} }
    }
    """
    strategies = discover_strategies()
    weights = _load_weights()

    per_strategy: Dict[str, str] = {}
    buy, sell = 0.0, 0.0

    if not strategies:
        return {
            "signal": "Hold",
            "votes": {"buy": 0, "sell": 0, "total": 0},
            "strength": "Weak",
            "details": {"per_strategy": {}, "weights": {}},
        }

    # شغّل بالتوازي مع مهلة لكل دالة
    with ThreadPoolExecutor(max_workers=min(8, max(2, len(strategies)))) as ex:
        futures = {}
        for name, fn in strategies:
            futures[ex.submit(_run_one, fn, df)] = name

        for fut in as_completed(futures):
            name = futures[fut]
            try:
                sig = _norm_signal(fut.result(timeout=timeout_sec))
            except Exception:
                sig = "Hold"

            per_strategy[name] = sig
            w = float(weights.get(name, DEFAULT_WEIGHT))

            if sig == "Buy":
                buy += w
            elif sig == "Sell":
                sell += w
            # Hold لا تضيف شيء

    total = max(buy, sell)
    if total < 0:
        total = 0

    # قرار مبدئي
    if buy > sell and total >= min_votes:
        final = "Buy"
    elif sell > buy and total >= min_votes:
        final = "Sell"
    else:
        final = "Hold"

    strength = _vote_to_strength(total)

    out: Dict[str, Any] = {
        "signal": final,
        "votes": {"buy": round(buy, 3), "sell": round(sell, 3), "total": round(total, 3)},
        "strength": strength,
    }
    if include_details:
        out["details"] = {"per_strategy": per_strategy, "weights": weights}

    return out

# واجهة متوافقة مع اللودر القديم (لو حاب تضيف الملف كاستراتيجية)
def generate_signal(df):
    return ensemble_signal(df)["signal"]
