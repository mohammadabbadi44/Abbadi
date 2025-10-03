# ai/server.py  (نسخة مرنة)
from fastapi import FastAPI, Query
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np

from .predict import predict_from_df

app = FastAPI(title="Trading AI Brain", version="1.1.0")

# ===== طلب مرن: نقبل أي حقول إضافية ونتكفّل بالتحويل داخليًا =====
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # تجاهل الحقول الزايدة
    candles: List[Dict[str, Any]]

class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    features: List[float]
    label: int  # 0=Hold,1=Buy,2=Sell

# ===== أدوات مساعدة =====
_TIME_ALIASES = ("time", "timestamp", "ts")
_REQ_NUMERIC = ("open", "high", "low", "close")  # volume اختياري

def _pick_time(d: Dict[str, Any]) -> Optional[str]:
    for k in _TIME_ALIASES:
        if k in d:
            v = d[k]
            # نحول أي رقم/float لنص
            try:
                if isinstance(v, (int, float, np.integer, np.floating)):
                    # لو كان بالميلي ثانية خفّضه
                    if float(v) > 1e12:
                        v = float(v) / 1000.0
                    return str(v)
                return str(v)
            except Exception:
                return None
    return None

def _row_to_record(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t = _pick_time(d)
    try:
        rec = {
            "time": t,
            "open": float(d["open"]),
            "high": float(d["high"]),
            "low":  float(d["low"]),
            "close":float(d["close"]),
            "volume": float(d.get("volume", 0.0)),
        }
    except Exception:
        return None
    # استبعد أي صف ناقص time أو فيه NaN
    if t is None or any([pd.isna(rec[k]) for k in ("open","high","low","close")]):
        return None
    return rec

def _build_df(candles: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for c in candles:
        rec = _row_to_record(c)
        if rec is not None:
            rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])
    df = pd.DataFrame(rows)
    # فلترة أخيرة لأي قيم غير صالحة
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open","high","low","close"])
    return df

# ===== Endpoints =====
@app.get("/health")
def health():
    return {"ok": True, "version": app.version}

@app.post("/predict")
def predict(
    req: PredictRequest,
    min_window: int = Query(60, ge=1, description="أقل عدد شموع للمعالجة (رجع Hold لو أقل)")
):
    """
    يقبل time كنص/رقم/ts/timestamp، volume اختياري، ويطنّش أي حقول إضافية.
    """
    df = _build_df(req.candles)
    if len(df) < min_window:
        # لا نرمي 422؛ نرجع Hold بشكل رحيم
        return {"signal": "Hold", "strength": "Weak", "score": 0.0}

    out = predict_from_df(df)
    # تأكد من المفاتيح
    signal = str(out.get("signal", "Hold"))
    strength = str(out.get("strength", "Weak"))
    score = float(out.get("score", 0.0))
    return {"signal": signal, "strength": strength, "score": score}

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    # حافظنا على نفس السكيمة الأصلية
    try:
        feats = np.array(req.features, dtype=float).reshape(1, -1)
        label = int(req.label)
    except Exception as e:
        return {"ok": False, "reason": f"bad payload: {e}"}

    # إذا بدك نربطها فعلاً بالتعلم الجزئي، نقدر نوصلها بـ model_store هنا.
    # حالياً خليه بسيط لأن عندك نسخة online_update من داخل البوت.
    return {"ok": True}
