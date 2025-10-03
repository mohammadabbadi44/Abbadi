# smc/fvg.py

def detect_fvg_zones(df):
    """
    يكشف عن فجوات Fair Value Gap (FVG) على أساس الشموع.
    
    ▪ Buy FVG: إذا كان Low[1] > High[0] => يعني فجوة صعودية.
    ▪ Sell FVG: إذا كان High[1] < Low[0] => يعني فجوة هبوطية.
    
    يعود بقائمة إشارات: ["Buy", "Sell", "Hold"]
    """
    signals = ["Hold"] * len(df)

    for i in range(2, len(df)):
        high_0 = df.at[i - 2, 'high']
        low_0 = df.at[i - 2, 'low']
        high_1 = df.at[i - 1, 'high']
        low_1 = df.at[i - 1, 'low']
        high_2 = df.at[i, 'high']
        low_2 = df.at[i, 'low']

        # كشف فجوة FVG صعودية (Buy)
        if low_1 > high_0 and low_2 > high_1:
            signals[i] = "Buy"

        # كشف فجوة FVG هبوطية (Sell)
        elif high_1 < low_0 and high_2 < low_1:
            signals[i] = "Sell"

    return signals
