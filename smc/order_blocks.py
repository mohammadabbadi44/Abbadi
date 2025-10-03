import pandas as pd
from logs.logger import log_error

def detect_order_blocks(df: pd.DataFrame, threshold_ratio: float = 2.0) -> pd.Series:
    """
    يرجّع سلسلة إشارات Order Block:
    - Bullish: بعد شمعة هابطة تأتي شمعة صعود قوية.
    - Bearish: بعد شمعة صاعدة تأتي شمعة هبوط قوية.
    - None: لا يوجد نمط واضح.
    """
    try:
        signals = ["None"] * len(df)
        df = df.copy()
        df["body_size"] = abs(df["close"] - df["open"])
        avg_body = df["body_size"].rolling(window=20).mean()

        for i in range(1, len(df) - 1):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]
            next_ = df.iloc[i + 1]
            avg = avg_body.iloc[i]

            if pd.isna(avg):
                continue

            # Bullish Order Block
            if curr["close"] < curr["open"] and next_["close"] > next_["open"]:
                next_body = abs(next_["close"] - next_["open"])
                if next_body > threshold_ratio * avg:
                    signals[i + 1] = "Bullish"

            # Bearish Order Block
            if curr["close"] > curr["open"] and next_["close"] < next_["open"]:
                next_body = abs(next_["close"] - next_["open"])
                if next_body > threshold_ratio * avg:
                    signals[i + 1] = "Bearish"

        return pd.Series(signals, index=df.index)

    except Exception as e:
        log_error(f"Order Block Detection Error: {e}")
        return pd.Series(["None"] * len(df), index=df.index)
