def detect_mss(df, bos_signals, liq_signals, fvg_zones, ob_signals):
    """
    MSS = BOS + Liquidity Grab + شمعة انعكاس داخل FVG أو Order Block
    """
    mss_signals = []

    length = len(df)
    # تأكد إن جميع القوائم بنفس الطول
    def pad(signals):
        if isinstance(signals, str):
            signals = [signals] * length
        return signals + ["Hold"] * (length - len(signals)) if len(signals) < length else signals[:length]

    bos_signals = pad(bos_signals)
    liq_signals = pad(liq_signals)
    fvg_zones = pad(fvg_zones)
    ob_signals = pad(ob_signals)

    for i in range(length):
        signal = "Hold"

        bos = bos_signals[i]
        liq = liq_signals[i]
        fvg = fvg_zones[i]
        ob = ob_signals[i]
        close = df.at[i, "close"]

        # نتحقق من شروط MSS
        if bos == "Bullish BOS" and liq == "Sell Trap" and (fvg == "Bullish FVG" or ob == "Bullish OB"):
            signal = "Bullish MSS"
        elif bos == "Bearish BOS" and liq == "Buy Trap" and (fvg == "Bearish FVG" or ob == "Bearish OB"):
            signal = "Bearish MSS"

        mss_signals.append(signal)

    return mss_signals
