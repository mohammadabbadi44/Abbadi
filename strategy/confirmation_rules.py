from logs.logger import log_error


def evaluate_signal_strength(details: dict, signal: str) -> str:
    """
    يحلل قوة الإشارة باستخدام:
    - المؤشرات الفنية (EMA, RSI, MACD, Bollinger)
    - أدوات SMC (Order Block, BOS, Liquidity Grab)
    - استراتيجية فيبوناتشي
    - حجم الفوليوم
    - القرب من مستويات الدعم والمقاومة

    Returns:
        str: Weak / Medium / Strong / Very Strong / Ignore
    """
    try:
        match_count = 0

        # ✅ المؤشرات الفنية
        for key in ['ema', 'rsi', 'macd', 'bollinger']:
            if details.get(key) == signal:
                match_count += 1

        # ✅ الفوليوم
        if details.get("volume") == "Strong" and signal != "Hold":
            match_count += 1

        # ✅ أدوات SMC
        if signal == "Buy":
            if details.get("order_block") == "Bullish":
                match_count += 1
            if details.get("bos") == "Bullish BOS":
                match_count += 1
            if details.get("liquidity") == "Sell Trap":
                match_count += 1

        elif signal == "Sell":
            if details.get("order_block") == "Bearish":
                match_count += 1
            if details.get("bos") == "Bearish BOS":
                match_count += 1
            if details.get("liquidity") == "Buy Trap":
                match_count += 1

        # ✅ فيبوناتشي
        if details.get("fibonacci") == signal:
            match_count += 1

        # ✅ مستويات الدعم والمقاومة
        price = details.get("price")  # السعر الحالي يجب أن يُمرر من الخارج
        if price:
            last_support = details.get("last_support")
            last_resistance = details.get("last_resistance")

            if signal == "Buy" and last_support:
                if abs(price - last_support) / price < 0.003:
                    match_count += 1

            if signal == "Sell" and last_resistance:
                if abs(price - last_resistance) / price < 0.003:
                    match_count += 1

        # ✅ التقييم النهائي حسب عدد التطابقات
        if match_count >= 6:
            return "Very Strong"
        elif match_count == 4 or match_count == 5:
            return "Strong"
        elif match_count == 3:
            return "Medium"
        elif match_count == 2:
            return "Weak"
        else:
            return "Ignore"

    except Exception as e:
        log_error(f"[❌] Signal Strength Evaluation Error: {e}")
        return "Weak"
