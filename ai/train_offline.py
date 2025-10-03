# ai/train_offline.py
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from .features import make_features
from .model_store import save_model
from .model import make_model, initial_partial_fit, CLASS_NAMES

def label_from_future_return(df: pd.DataFrame, horizon: int = 5, buy_th=0.0015, sell_th=-0.0015):
    """
    يولّد تصنيف (0=Hold,1=Buy,2=Sell) حسب العائد المستقبلي على horizon شموع
    """
    close = df['close'].astype(float)
    future = close.shift(-horizon)
    future_ret = (future - close) / (close + 1e-9)
    labels = np.zeros(len(df), dtype=int)
    labels[future_ret >  buy_th] = 1
    labels[future_ret < sell_th] = 2
    return labels

def main(csv_path: str, horizon: int = 5):
    raw = pd.read_csv(csv_path)
    # required cols: time, open, high, low, close, volume
    feats = make_features(raw)
    # align labels to feats index
    labels = label_from_future_return(raw.loc[feats.index], horizon=horizon)

    # remove any remaining NaNs just in case
    mask = np.isfinite(feats.to_numpy()).all(axis=1)
    X = feats.to_numpy()[mask]
    y = labels[mask]

    # balance slighty (optional): keep as-is for now
    X, y = shuffle(X, y, random_state=42)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = make_model()
    clf = initial_partial_fit(clf, Xs, y)

    save_model(clf, scaler)
    print(f"[OK] Trained & saved model. Samples={len(y)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="path to historical CSV")
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()
    main(args.csv, args.horizon)
