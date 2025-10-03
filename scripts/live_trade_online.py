# -*- coding: utf-8 -*-
"""
scripts/live_trade_online.py

لايف + تدريب مستمر (Online Learning):
- يقرأ ensemble_ai.yaml + live.yaml
- يفتح صفقات لايف بناءً على إنسمبل + AI
- يسجّل الإدخالات بميزات لحظة الدخول (features snapshot)
- يراقب إغلاقات الصفقات ويستخرج الربح/الخسارة
- يدرّب SGDClassifier (logistic loss) بشكل دوري عبر partial_fit
- يعاير decision threshold لتعظيم EV المتوقع
- يحفظ الموديل models/ensemble_sgd.joblib ويعمل Hot-Reload بالمحرك

ملاحظة:
- التدريب يعتمد على لوج الصفقات (open/close) مع ميزاتها لحظة الدخول.
- البداية تكون بالموديل الموجود (joblib). لو مافيه، يبدأ بدون AI (ثقة 0.5) إلى أن يتشكل Dataset كافي.
"""

import os, sys, time, math, json, threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.run_hybrid_ensemble_ai import (
    load_cfg as load_ensemble_cfg,
    fetch_data, build_features, ensure_symbol_visible, atr, ema, rsi
)

try:
    import yaml
    import MetaTrader5 as mt5
    from sklearn.linear_model import SGDClassifier
    from sklearn.utils import shuffle
    import joblib
except ImportError as e:
    print("❌ مفقود:", e, "\nثبت: pip install scikit-learn joblib pyyaml MetaTrader5")
    sys.exit(1)

CONFIG_ENSEMBLE = ROOT / "config" / "ensemble_ai.yaml"
CONFIG_LIVE     = ROOT / "config" / "live.yaml"
MODELS_DIR      = ROOT / "models"
LOG_DIR         = ROOT / "reports" / "live_logs"
DATASET_CSV     = LOG_DIR / "online_dataset.csv"   # ميزات لحظة الدخول + نتيجة عند الإغلاق
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH      = MODELS_DIR / "ensemble_sgd.joblib"
THRESH_PATH     = MODELS_DIR / "ensemble_sgd_threshold.json"

# ---------------- Utils ----------------
def load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def within_hours(cfg_live):
    h = cfg_live.get("filters", {}).get("trading_hours", {})
    if not h.get("enabled", True): return True
    now = datetime.now().time()
    st = datetime.strptime(h.get("start","00:00"), "%H:%M").time()
    en = datetime.strptime(h.get("end","23:59"), "%H:%M").time()
    return (st <= now <= en)

def get_account_info():
    acc = mt5.account_info()
    if not acc: return None
    return {"balance": acc.balance, "equity": acc.equity, "login": acc.login}

def daily_pnl():
    day = datetime.now().strftime("%Y%m%d")
    p = LOG_DIR / f"pnl_{day}.csv"
    if not p.exists(): return 0.0
    df = pd.read_csv(p)
    return float(df["pnl_usd"].sum()) if not df.empty else 0.0

def circuit_breaker_hit(cfg_live, eq0):
    max_daily_loss_pct = float(cfg_live["trade"]["max_daily_loss_pct"])
    pnl = daily_pnl()
    return pnl <= -(eq0 * max_daily_loss_pct / 100.0)

def append_pnl_log(row: dict):
    day = datetime.now().strftime("%Y%m%d")
    p = LOG_DIR / f"pnl_{day}.csv"
    df = pd.DataFrame([row])
    if p.exists():
        df0 = pd.read_csv(p); df = pd.concat([df0, df], ignore_index=True)
    df.to_csv(p, index=False)

def ensure_live_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- AI Model (Hot-Reload) ----------------
class AIModule:
    def __init__(self):
        self.model = None
        self.threshold = 0.7
        self.mtime = None
        self.load_if_updated()

    def load_if_updated(self):
        if MODEL_PATH.exists():
            mtime = MODEL_PATH.stat().st_mtime
            if self.mtime is None or mtime != self.mtime:
                try:
                    self.model = joblib.load(MODEL_PATH)
                    self.mtime = mtime
                    if THRESH_PATH.exists():
                        self.threshold = json.loads(THRESH_PATH.read_text())["threshold"]
                    print(f"🧠 AI model loaded (thr={self.threshold})")
                except Exception as e:
                    print("⚠️ Failed loading model:", e)

    def predict_conf(self, features_dict: dict):
        if self.model is None:
            return 0.5
        keys = ["rsi14","atr14","ema20","ema50","ema200","score"]
        x = np.array([[float(features_dict.get(k,0.0)) for k in keys]], dtype=float)
        try:
            proba = self.model.predict_proba(x)[0,1]
            return float(proba)
        except Exception:
            return 0.5

AI = AIModule()

# ---------------- Dataset Helpers ----------------
FEATURE_KEYS = ["time","side","rsi14","atr14","ema20","ema50","ema200","score","sl_atr_mult","tp_atr_mult","result"]  # result=1/0 بعد الإغلاق

def log_entry_features(time_iso, side, feats_row, cfg_ens):
    row = {
        "time": time_iso,
        "side": side,
        "rsi14": float(feats_row["rsi14"]),
        "atr14": float(feats_row["atr14"]),
        "ema20": float(feats_row["ema20"]),
        "ema50": float(feats_row["ema50"]),
        "ema200": float(feats_row["ema200"]),
        "score": float(feats_row["score"]),
        "sl_atr_mult": float(cfg_ens.get("sl_atr_mult", 3.5)),
        "tp_atr_mult": float(cfg_ens.get("tp_atr_mult", 4.8)),
        "result": np.nan,  # نعبيها عند الإغلاق
    }
    df = pd.DataFrame([row])
    if DATASET_CSV.exists():
        df0 = pd.read_csv(DATASET_CSV)
        df = pd.concat([df0, df], ignore_index=True)
    df.to_csv(DATASET_CSV, index=False)

def update_result_on_close(close_time_iso, pnl_usd):
    # نلصق أقرب إدخال مفتوح (أحدث صفّ فيه result NaN) ونعيّن 1 لو pnl>0 وإلا 0
    if not DATASET_CSV.exists(): return
    df = pd.read_csv(DATASET_CSV)
    mask = df["result"].isna()
    if not mask.any(): return
    idx = df[mask].index[-1]
    df.loc[idx, "result"] = 1 if float(pnl_usd) > 0 else 0
    df.to_csv(DATASET_CSV, index=False)

# ---------------- Online Trainer Thread ----------------
class OnlineTrainer(threading.Thread):
    def __init__(self, interval_minutes=30, min_samples=200, max_daily_drawdown_pause_pct=2.5):
        super().__init__(daemon=True)
        self.interval = interval_minutes * 60
        self.min_samples = min_samples
        self.pause_dd = max_daily_drawdown_pause_pct
        self.running = True

    def run(self):
        while self.running:
            try:
                eq = get_account_info()["equity"]
                # لو اليوم سيء جدًا، لا تحدّث نموذج (تجنب overfit على سلوك يوم سيء)
                if circuit_breaker_hit(load_yaml(CONFIG_LIVE), eq):
                    time.sleep(self.interval); continue
                self.train_once()
            except Exception as e:
                print("⚠️ Trainer error:", e)
            time.sleep(self.interval)

    def train_once(self):
        if not DATASET_CSV.exists():
            return
        df = pd.read_csv(DATASET_CSV)
        df = df.dropna(subset=["result"])
        if len(df) < self.min_samples:
            print(f"⏳ Waiting for samples: {len(df)}/{self.min_samples}")
            return
        # تحضير X,y
        X_cols = ["rsi14","atr14","ema20","ema50","ema200","score","sl_atr_mult","tp_atr_mult"]
        X = df[X_cols].values.astype(float)
        y = df["result"].values.astype(int)
        X, y = shuffle(X, y, random_state=42)

        # SGDClassifier مع log loss (لوجيستي)
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            if not isinstance(model, SGDClassifier):
                model = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1, learning_rate="optimal", tol=None)
        else:
            model = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1, learning_rate="optimal", tol=None)

        # partial_fit يحتاج classes
        classes = np.array([0,1], dtype=int)
        # نعمل عدة epochs صغيرة
        epochs = 6
        for _ in range(epochs):
            model.partial_fit(X, y, classes=classes)

        # معايرة العتبة لتعظيم EV بناءً على R:R (تقريبي)
        rr = (df["tp_atr_mult"] / df["sl_atr_mult"]).median() if "tp_atr_mult" in df and "sl_atr_mult" in df else 1.5
        proba = _predict_proba_safe(model, X)
        thr = _best_threshold_for_ev(proba, y, rr)
        joblib.dump(model, MODEL_PATH)
        THRESH_PATH.write_text(json.dumps({"threshold": float(thr), "updated_at": datetime.now().isoformat()}), encoding="utf-8")
        print(f"🧪 Online train done: samples={len(df)} | thr={thr:.3f}")
        # سيقرأها Trading loop تلقائيًا عبر AI.load_if_updated()

def _predict_proba_safe(model, X):
    try:
        return model.predict_proba(X)[:,1]
    except Exception:
        # لو ما فيه predict_proba (بعض إعدادات SGD)، نقرّب via decision_function
        d = model.decision_function(X)
        # sigmoid
        return 1.0/(1.0+np.exp(-d))

def _best_threshold_for_ev(proba, y, rr=1.5):
    # EV(p,thr) ~ p_win*RR - (1-p_win)
    # p_win ≈ mean(y_hat|score>=thr)
    best_thr = 0.7; best_ev = -1e9
    for thr in np.linspace(0.55, 0.85, 31):
        sel = proba >= thr
        if sel.sum() < 20:
            continue
        p_win = y[sel].mean()
        ev = p_win*rr - (1 - p_win)
        if ev > best_ev:
            best_ev = ev; best_thr = thr
    return best_thr

# ---------------- Trading Logic ----------------
def tf_to_sec(tf):
    tf = str(tf).upper()
    return {"M1":60,"M5":300,"M15":900,"M30":1800,"H1":3600}.get(tf,300)

def position_size(si, cfg_live, atr_val, sl_mult):
    point = si.point or 0.01
    tick_value = getattr(si, "trade_tick_value", 1.0) or 1.0
    vol_min  = getattr(si, "volume_min", 0.01) or 0.01
    vol_step = getattr(si, "volume_step", 0.01) or 0.01
    eq = get_account_info()["equity"]
    risk_usd = eq * (cfg_live["trade"]["risk_per_trade_pct"]/100.0)
    atr_points = atr_val / point
    sl_points  = max(1.0, sl_mult * atr_points)
    lots_raw   = risk_usd / (sl_points * point * tick_value)
    lots      = max(vol_min, math.floor(lots_raw/vol_step)*vol_step)
    return float(lots)

def ai_confidence_from_feats(feats_row):
    d = {
        "rsi14": float(feats_row["rsi14"]),
        "atr14": float(feats_row["atr14"]),
        "ema20": float(feats_row["ema20"]),
        "ema50": float(feats_row["ema50"]),
        "ema200": float(feats_row["ema200"]),
        "score": float(feats_row["score"]),
    }
    AI.load_if_updated()
    return AI.predict_conf(d), AI.threshold

def ensemble_signal(df, feats, cfg_ens, si):
    i = len(df) - 2
    if i < 250:
        return None, None, None
    above = df["close"].iloc[i] > feats["ema200"].iloc[i]
    cond_buy  = (feats["ema20"].iloc[i] > feats["ema50"].iloc[i]) and (feats["rsi14"].iloc[i] > 55) and above
    cond_sell = (feats["ema20"].iloc[i] < feats["ema50"].iloc[i]) and (feats["rsi14"].iloc[i] < 45) and (not above)

    # AI gate
    conf, thr = ai_confidence_from_feats(feats.iloc[i])
    min_conf = float(cfg_ens.get("min_ensemble_conf", 0.7))
    ai_gate = (conf >= max(min_conf, thr))

    side = None
    if cond_buy and ai_gate: side = "BUY"
    if cond_sell and ai_gate: side = "SELL"
    return side, i, conf

def place_order(side, df, feats, i, si, cfg_ens, cfg_live):
    point = si.point or 0.01
    spread = cfg_live["trade"]["spread_points"] * point
    price  = df["close"].iloc[i]
    atr_v  = float(feats["atr14"].iloc[i])
    sl_m   = float(cfg_ens.get("sl_atr_mult", 3.5))
    tp_m   = float(cfg_ens.get("tp_atr_mult", 4.8))

    lots = position_size(si, cfg_live, atr_v, sl_m)
    if lots <= 0:
        return False, "Lot calc <= 0"

    if side=="BUY":
        price_in = price + spread
        sl = price - sl_m*atr_v
        tp = price + tp_m*atr_v
        req_type = mt5.ORDER_TYPE_BUY
    else:
        price_in = price - spread
        sl = price + sl_m*atr_v
        tp = price - tp_m*atr_v
        req_type = mt5.ORDER_TYPE_SELL

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": cfg_live["trade"]["symbol"],
        "volume": lots,
        "type": req_type,
        "price": price_in,
        "sl": sl, "tp": tp,
        "deviation": int(cfg_live["trade"]["slippage_points"]),
        "magic": int(cfg_live["trade"]["magic"]),
        "comment": "LiveOnlineAI",
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"retcode={getattr(res,'retcode',None)} {getattr(res,'comment','send_fail')}"
    # لوج دخول + لقطة ميزات للتدريب لاحقاً
    log_entry_features(datetime.now().isoformat(), side, feats.iloc[i], cfg_ens)
    append_pnl_log({"time": datetime.now().isoformat(), "side": side, "lots": lots, "pnl_usd": 0.0, "event": "OPEN"})
    return True, f"Opened {side} {lots} @ {price_in:.2f}"

def poll_closed_deals(symbol):
    """
    نجلب من تاريخ اليوم صفقات مغلقة ونحدّث الـdataset بالنتيجة.
    (تبسيط: نعتمد profit deal>0 = 1 وإلا 0)
    """
    utc_from = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    utc_to   = datetime.now()
    deals = mt5.history_deals_get(utc_from, utc_to, group="*")
    if deals is None:
        return
    for d in deals:
        if d.symbol != symbol: continue
        if d.entry == mt5.DEAL_ENTRY_OUT:
            pnl = d.profit
            update_result_on_close(datetime.fromtimestamp(d.time).isoformat(), pnl)
            append_pnl_log({"time": datetime.fromtimestamp(d.time).isoformat(), "side": "NA", "lots": d.volume, "pnl_usd": pnl, "event": "CLOSE"})

def main():
    ensure_live_dirs()
    cfg_live = load_yaml(CONFIG_LIVE)
    cfg_ens  = load_ensemble_cfg(CONFIG_ENSEMBLE)

    symbol = cfg_live["trade"]["symbol"]
    timeframe = cfg_live["trade"]["timeframe"]
    days = max(10, cfg_ens.get("days", 180))

    if not mt5.initialize():
        print("❌ mt5.initialize failed:", mt5.last_error()); sys.exit(1)

    si = ensure_symbol_visible(symbol)
    if not si:
        print(f"❌ Symbol not visible: {symbol}"); sys.exit(1)

    acc = get_account_info()
    if not acc:
        print("❌ لا يمكن قراءة معلومات الحساب من MT5."); sys.exit(1)
    eq0 = acc["equity"]
    print(f"▶️ Live+Online on {symbol} {timeframe} | acct={acc['login']} | eq={eq0:.2f}")

    # أطلق المدرّب بالخلفية
    trainer = OnlineTrainer(interval_minutes=30, min_samples=200, max_daily_drawdown_pause_pct=cfg_live["trade"]["max_daily_loss_pct"])
    trainer.start()

    last_bar_time = None
    bar_sec = tf_to_sec(timeframe)

    try:
        while True:
            if not within_hours(cfg_live):
                time.sleep(5); poll_closed_deals(symbol); continue

            if circuit_breaker_hit(cfg_live, eq0):
                print("⛔ Circuit Breaker hit. Pausing new entries.")
                time.sleep(30); poll_closed_deals(symbol); continue

            # سحب بيانات + ميزات
            try:
                df = fetch_data(symbol, timeframe, days)
            except Exception as e:
                print("⚠️ fetch_data:", e); time.sleep(5); continue

            feats = build_features(df)
            cur_last = df["time"].iloc[-1]
            if last_bar_time is None:
                last_bar_time = cur_last

            # إشارة عند اكتمال شمعة
            if cur_last != last_bar_time:
                side, i, conf = ensemble_signal(df, feats, cfg_ens, si)
                if side:
                    ok, msg = place_order(side, df, feats, i, si, cfg_ens, cfg_live)
                    print(datetime.now().strftime("%H:%M:%S"), f"{msg} | conf={conf:.2f}")
                last_bar_time = cur_last

            # التحقق من صفقات أُغلقت لتحديث الداتاسِت
            poll_closed_deals(symbol)

            # Hot reload للـAI/Config كل بضع دورات
            AI.load_if_updated()
            # بإمكانك أيضاً إعادة تحميل ensemble_ai.yaml دورياً لو الأوتوبايلِت يُحدّثه
            # cfg_ens = load_ensemble_cfg(CONFIG_ENSEMBLE)

            time.sleep(2)
    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    finally:
        trainer.running = False
        mt5.shutdown()

if __name__ == "__main__":
    main()
