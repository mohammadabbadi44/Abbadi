# strategy/ict_entry.py

def generate_ict_signal(df, i, ob_signal, bos_signal, liq_signal, fvg_signal, mss_signal, pa_signal):
    """
    توليد إشارة دخول ICT مرنة تشمل MSS وFVG وOB وPA + فلترة EMA200
    """
    try:
        candle = df.iloc[i]
        close = candle['close']
        ema200 = candle.get('ema200', None)

        buy_count = 0
        sell_count = 0

        # ===== إشارات شراء =====
        if ob_signal == "Bullish":
            buy_count += 1
        if bos_signal == "Bullish BOS":
            buy_count += 1
        if liq_signal in ["Sell Trap", "Equal Lows"]:
            buy_count += 1
        if fvg_signal == "Bullish FVG":
            buy_count += 1
        if mss_signal == "Bullish MSS":
            buy_count += 1
        if pa_signal == "Bullish Engulfing":
            buy_count += 2  # أفضل من Pin Bar
        elif pa_signal == "Pin Bar":
            buy_count += 1

        # ===== إشارات بيع =====
        if ob_signal == "Bearish":
            sell_count += 1
        if bos_signal == "Bearish BOS":
            sell_count += 1
        if liq_signal in ["Buy Trap", "Equal Highs"]:
            sell_count += 1
        if fvg_signal == "Bearish FVG":
            sell_count += 1
        if mss_signal == "Bearish MSS":
            sell_count += 1
        if pa_signal == "Bearish Engulfing":
            sell_count += 2
        elif pa_signal == "Pin Bar":
            sell_count += 1

        # ===== شرط الترند عبر EMA200 =====
        above_ema = ema200 is None or close > ema200
        below_ema = ema200 is None or close < ema200

        # ===== قرار الدخول =====
        if buy_count >= 2 and above_ema:
            return "Buy"
        elif sell_count >= 2 and below_ema:
            return "Sell"
        else:
            return "Hold"

    except Exception as e:
        print(f"❌ Error in ICT Signal Generation: {e}")
        return "Hold"
