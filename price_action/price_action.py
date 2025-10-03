import pandas as pd

def detect_price_action_patterns(df: pd.DataFrame) -> pd.Series:
    """
    يكشف نماذج Price Action الشائعة:
    - Bullish/Bearish Engulfing
    - Hammer / Inverted Hammer
    - Doji
    - Morning Star / Evening Star
    """
    signals = ["Hold"] * len(df)

    for i in range(2, len(df)):
        prev2 = df.iloc[i - 2]
        prev1 = df.iloc[i - 1]
        curr = df.iloc[i]

        body_curr = abs(curr['close'] - curr['open'])
        body_prev1 = abs(prev1['close'] - prev1['open'])
        range_curr = curr['high'] - curr['low']
        range_prev1 = prev1['high'] - prev1['low']

        # ==== Bullish Engulfing ====
        if prev1['close'] < prev1['open'] and curr['close'] > curr['open']:
            if curr['open'] < prev1['close'] and curr['close'] > prev1['open']:
                signals[i] = "Buy"

        # ==== Bearish Engulfing ====
        elif prev1['close'] > prev1['open'] and curr['close'] < curr['open']:
            if curr['open'] > prev1['close'] and curr['close'] < prev1['open']:
                signals[i] = "Sell"

        # ==== Hammer ====
        if body_curr < (range_curr * 0.3) and (curr['low'] < min(curr['open'], curr['close']) - range_curr * 0.4):
            signals[i] = "Buy"

        # ==== Inverted Hammer ====
        if body_curr < (range_curr * 0.3) and (curr['high'] > max(curr['open'], curr['close']) + range_curr * 0.4):
            signals[i] = "Sell"

        # ==== Doji ====
        if body_curr < range_curr * 0.1:
            signals[i] = "Hold"

        # ==== Morning Star ====
        if prev2['close'] < prev2['open'] and body_prev1 < range_prev1 * 0.3 and curr['close'] > curr['open']:
            if curr['close'] > prev2['open']:
                signals[i] = "Buy"

        # ==== Evening Star ====
        if prev2['close'] > prev2['open'] and body_prev1 < range_prev1 * 0.3 and curr['close'] < curr['open']:
            if curr['close'] < prev2['open']:
                signals[i] = "Sell"

    return pd.Series(signals, index=df.index)
