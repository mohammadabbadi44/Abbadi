import numpy as np
import pandas as pd

def identify_support_resistance(df: pd.DataFrame, window: int = 5, threshold: float = 0.005) -> dict:
    """
    يحدد مستويات الدعم والمقاومة بناءً على القمم والقيعان المحلية.

    Args:
        df (DataFrame): بيانات الشموع تحتوي على أعمدة 'high' و 'low'
        window (int): عدد الشموع للفحص قبل وبعد كل نقطة
        threshold (float): فرق النسبة المئوية للفصل بين المستويات القريبة

    Returns:
        dict: يحتوي على قوائم الدعم والمقاومة، بالإضافة لآخر دعم وآخر مقاومة
    """
    highs = df['high']
    lows = df['low']

    support_levels = []
    resistance_levels = []

    for i in range(window, len(df) - window):
        local_max = max(highs[i - window:i + window + 1])
        local_min = min(lows[i - window:i + window + 1])

        current_high = highs[i]
        current_low = lows[i]

        # مقاومة جديدة؟
        if current_high == local_max:
            if not any(abs(current_high - r) / r < threshold for r in resistance_levels):
                resistance_levels.append(round(current_high, 2))

        # دعم جديد؟
        if current_low == local_min:
            if not any(abs(current_low - s) / s < threshold for s in support_levels):
                support_levels.append(round(current_low, 2))

    return {
        "support": sorted(support_levels),
        "resistance": sorted(resistance_levels),
        "last_support": support_levels[-1] if support_levels else None,
        "last_resistance": resistance_levels[-1] if resistance_levels else None,
    }
