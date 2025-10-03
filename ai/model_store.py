# ai/model_store.py
import os
import joblib
from sklearn.preprocessing import StandardScaler
from .model import make_model

DEFAULT_DIR  = "ai/store"
MODEL_PATH   = os.path.join(DEFAULT_DIR, "clf.pkl")
SCALER_PATH  = os.path.join(DEFAULT_DIR, "scaler.pkl")

def ensure_store():
    os.makedirs(DEFAULT_DIR, exist_ok=True)

def save_model(clf, scaler):
    ensure_store()
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

def load_model():
    ensure_store()
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        clf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        clf = make_model()
        scaler = StandardScaler(with_mean=True, with_std=True)
        # model will be fit in train_offline.py first time
    return clf, scaler
