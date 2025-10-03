# ai/predict.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from .features import make_features
from .model_store import load_model, save_model
from .model import CLASS_NAMES

def _softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def predict_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Input: raw OHLCV dataframe (>= 210 شمعة أفضل)
    Output format: {"signal": "Buy|Sell|Hold", "strength": "Weak|Medium|Strong|Very Strong", "score": float}
    """
    feats = make_features(df)
    if len(feats) == 0:
        return {"signal":"Hold","strength":"Weak","score":0.0}

    X_last = feats.iloc[[-1]].to_numpy()

    clf, scaler = load_model()
    # لو أول مرة لسه ما اتدرّب، رجّع Hold
    try:
        Xs = scaler.transform(X_last)
    except Exception:
        # scaler not fitted yet
        return {"signal":"Hold","strength":"Weak","score":0.0}

    # decision function → probabilities
    try:
        logits = clf.decision_function(Xs)
        if logits.ndim == 1:  # binary edge-case (shouldn't happen after initial fit)
            logits = np.column_stack([np.zeros_like(logits), logits])
        probs = _softmax(logits)
    except Exception:
        # fallback predict_proba if available
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(Xs)
        else:
            return {"signal":"Hold","strength":"Weak","score":0.0}

    idx = int(np.argmax(probs[0]))
    signal = CLASS_NAMES[idx]

    # score = confidence of non-Hold classes; if Hold picked, use 1 - P(Hold)
    p_hold = float(probs[0, 0]) if probs.shape[1] >= 1 else 0.0
    p_buy  = float(probs[0, 1]) if probs.shape[1] >= 2 else 0.0
    p_sell = float(probs[0, 2]) if probs.shape[1] >= 3 else 0.0
    conf = {"Hold": 1.0 - p_hold, "Buy": p_buy, "Sell": p_sell}.get(signal, 0.0)

    # strength buckets
    if conf >= 0.8:   strength = "Very Strong"
    elif conf >= 0.6: strength = "Strong"
    elif conf >= 0.4: strength = "Medium"
    else:             strength = "Weak"

    return {"signal": str(signal), "strength": strength, "score": float(round(conf, 4))}
