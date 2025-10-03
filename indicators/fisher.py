# indicators/fisher.py

import pandas as pd
import numpy as np

def calculate_fisher_signal(df: pd.DataFrame, period: int = 10) -> str:
    """
    يولّد إشارة تداول باستخدام مؤشر Fisher Transform:
    - Buy: إذا Fisher اخترق للأعلى
    - Sell: إذا Fisher اخترق للأسفل
    - Hold: غير ذلك

    يعتمد على متوسط السعر (High + Low) / 2
    """
    if df.shape[0] < period + 2:
        return "Hold"

    median_price = (df['high'] + df['low']) / 2

    value = 2 * ((median_price - median_price.rolling(period).min()) /
                 (median_price.rolling(period).max() - median_price.rolling(period).min()) - 0.5)

    value = value.clip(lower=-0.999, upper=0.999)
    fisher = 0.0 * value  # تهيئة السلسلة
    fisher.iloc[0] = 0

    for i in range(1, len(value)):
        fisher.iloc[i] = 0.5 * np.log((1 + value.iloc[i]) / (1 - value.iloc[i])) + 0.5 * fisher.iloc[i - 1]

    # إشارات التقاطع
    prev = fisher.iloc[-2]
    curr = fisher.iloc[-1]

    if prev < 0 and curr > 0:
        return "Buy"
    elif prev > 0 and curr < 0:
        return "Sell"
    else:
        return "Hold"
