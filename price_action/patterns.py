import pandas as pd


def detect_price_action_patterns(df: pd.DataFrame) -> pd.Series:
    """
    يكشف أنماط Price Action مثل:
    - Bullish Engulfing
    - Bearish Engulfing
    - Hammer
    - Morning Star
    - Doji

    Returns:
        pd.Series: إشارات Price Action (Buy/Sell/Hold)
    """
    signals = ["Hold"] * len(df)

    for i in range(2, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        pre_prev = df.iloc[i - 2]

        body_curr = abs(curr['close'] - curr['open'])
        body_prev = abs(prev['close'] - prev['open'])
        high_low_range = curr['high'] - curr['low']

        # ✅ Bullish Engulfing
        if prev['close'] < prev['open'] and curr['close'] > curr['open']:
            if curr['open'] < prev['close'] and curr['close'] > prev['open']:
                signals[i] = "Buy"

        # ✅ Bearish Engulfing
        elif prev['close'] > prev['open'] and curr['close'] < curr['open']:
            if curr['open'] > prev['close'] and curr['close'] < prev['open']:
                signals[i] = "Sell"

        # ✅ Hammer (شمعة صغيرة الجسم، ظل سفلي طويل، غالبًا نهاية هبوط)
        elif (
            body_curr / high_low_range < 0.4 and
            (curr['close'] > curr['open']) and
            (curr['low'] < curr['open'] - 2 * body_curr)
        ):
            signals[i] = "Buy"

        # ✅ Morning Star (ثلاث شموع: هبوط - شمعة صغيرة - صعود قوي)
        elif (
            pre_prev['close'] < pre_prev['open'] and  # أول شمعة هابطة
            abs(prev['close'] - prev['open']) / (prev['high'] - prev['low']) < 0.3 and  # الثانية صغيرة
            curr['close'] > curr['open'] and  # الثالثة صاعدة
            curr['close'] > (pre_prev['open'] + pre_prev['close']) / 2  # تخترق نصف الشمعة الأولى
        ):
            signals[i] = "Buy"

        # ✅ Doji (حيادية – نستخدمها فقط لمعلومات، لا إشارة مباشرة)
        elif body_curr / high_low_range < 0.1:
            signals[i] = "Hold"

    return pd.Series(signals, index=df.index)
