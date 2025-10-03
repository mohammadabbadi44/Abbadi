# strategy/voting_core.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable

# ===== إعدادات قابلة للتعديل بسرعة =====
WARMUP_BARS = 220                 # انتظار قبل السماح بدخول (EMA200 + هامش)
USE_EMA_TREND_FILTER = False      # عطّله حالياً لفتح صفقات
EMA_TREND_COL = "ema_200"

MIN_STRAT_VOTES = 1               # الحد الأدنى لأصوات الاستراتيجيات لدخول "عادي"
MIN_SMC_VOTES_SMC_ONLY = 2        # الحد الأدنى لدخول "SMC-only" عندما الاستراتيجيات ساكتة
MIN_SCORE = 2                     # مجموع نقاط أدنى للدخول

# نقاط التصويت
POINTS = {
    "strategy_buy": 2,
    "strategy_sell": 2,
    "smc_buy": 1,
    "smc_sell": 1,
}

# تحويل مجموع النقاط إلى قوة الإشارة
def score_to_strength(score: int) -> str:
    if score >= 6:   return "Very Strong"
    if score >= 4:   return "Strong"
    if score >= 3:   return "Medium"
    return "Weak"

def _safe_sig(x: str) -> str:
    if not isinstance(x, str):
        return "Hold"
    x = x.strip().title()
    return x if x in ("Buy", "Sell", "Hold") else "Hold"

def ensure_ema200(df: pd.DataFrame):
    if EMA_TREND_COL not in df.columns:
        df[EMA_TREND_COL] = df["close"].ewm(span=200, adjust=False).mean()

def compute_strategy_votes(
    i: int,
    df: pd.DataFrame,
    strategy_funcs: Dict[str, Callable[[pd.DataFrame, int], str]],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """
    يرجّع عدّ خام للإشارات من كل استراتيجية + المجموع العام.
    """
    per_strat_counts = {}
    total = {"Buy": 0, "Sell": 0, "Hold": 0}
    for name, func in strategy_funcs.items():
        try:
            sig = _safe_sig(func(df, i))
        except Exception:
            sig = "Hold"
        per_strat_counts[name] = {"Buy": 0, "Sell": 0, "Hold": 0}
        per_strat_counts[name][sig] += 1
        total[sig] += 1
    return per_strat_counts, total

def compute_smc_votes(
    i: int,
    smc_signals: Dict[str, str],  # مثل {"ob":"Buy","bos":"Buy","liq":"Hold","mss":"Sell",...}
) -> Dict[str, int]:
    counts = {"Buy": 0, "Sell": 0}
    for k, v in (smc_signals or {}).items():
        sig = _safe_sig(v)
        if sig == "Buy":  counts["Buy"]  += 1
        elif sig == "Sell": counts["Sell"] += 1
    return counts

def combined_signal(
    i: int,
    df: pd.DataFrame,
    strategy_funcs: Dict[str, Callable[[pd.DataFrame, int], str]],
    smc_signals: Dict[str, str],
    debug_log: List[str] = None
) -> Tuple[str, str, Dict]:
    """
    يرجّع (signal, strength, debug_info)
    """
    if i < WARMUP_BARS:
        return "Hold", "Weak", {"reason": "warmup"}

    ensure_ema200(df)

    # 1) عدّ الاستراتيجيات (خام)
    per_strat_counts, strat_total = compute_strategy_votes(i, df, strategy_funcs)
    strat_buy  = strat_total["Buy"]
    strat_sell = strat_total["Sell"]
    strat_non_hold = strat_buy + strat_sell

    # 2) عدّ SMC/PA
    smc_counts = compute_smc_votes(i, smc_signals)
    smc_buy  = smc_counts["Buy"]
    smc_sell = smc_counts["Sell"]
    smc_non_hold = smc_buy + smc_sell

    # 3) EMA Trend (اختياري حالياً)
    ema_ok = True
    if USE_EMA_TREND_FILTER:
        price = df.iloc[i]["close"]
        ema200 = df.iloc[i][EMA_TREND_COL]
        if np.isnan(ema200):
            ema_ok = False

    # 4) حسبة النقاط والعتبات
    score_buy  = strat_buy  * POINTS["strategy_buy"]  + smc_buy  * POINTS["smc_buy"]
    score_sell = strat_sell * POINTS["strategy_sell"] + smc_sell * POINTS["smc_sell"]

    reason = []
    if strat_non_hold >= MIN_STRAT_VOTES:
        # دخول “عادي” لو تعدّت النقاط العتبة
        if score_buy >= max(MIN_SCORE, score_sell + 0):  # تعادل = لا دخول
            if ema_ok:
                return "Buy", score_to_strength(score_buy), {
                    "mode":"normal", "strat_buy":strat_buy, "smc_buy":smc_buy,
                    "score_buy":score_buy, "score_sell":score_sell
                }
            else:
                reason.append("ema_filter_blocked_buy")
        if score_sell >= max(MIN_SCORE, score_buy + 0):
            if ema_ok:
                return "Sell", score_to_strength(score_sell), {
                    "mode":"normal", "strat_sell":strat_sell, "smc_sell":smc_sell,
                    "score_buy":score_buy, "score_sell":score_sell
                }
            else:
                reason.append("ema_filter_blocked_sell")
    else:
        # SMC-only gate
        if smc_buy >= MIN_SMC_VOTES_SMC_ONLY and score_buy >= MIN_SCORE:
            return "Buy", score_to_strength(score_buy), {
                "mode":"smc_only", "smc_buy":smc_buy, "score_buy":score_buy
            }
        if smc_sell >= MIN_SMC_VOTES_SMC_ONLY and score_sell >= MIN_SCORE:
            return "Sell", score_to_strength(score_sell), {
                "mode":"smc_only", "smc_sell":smc_sell, "score_sell":score_sell
            }
        reason.append("no_strat_votes_and_smc_below_threshold")

    dbg = {
        "mode":"none",
        "strat_non_hold": strat_non_hold,
        "smc_non_hold": smc_non_hold,
        "score_buy": score_buy,
        "score_sell": score_sell,
        "ema_ok": ema_ok,
        "reason": ";".join(reason) if reason else "no_edge",
        "per_strat_sample": {k: v for k, v in list(per_strat_counts.items())[:5]}  # أول 5 للاستعراض
    }
    if debug_log is not None:
        debug_log.append(str(dbg))
    return "Hold", "Weak", dbg
