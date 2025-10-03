def format_symbol(symbol: str, broker: str = "mt5") -> str:
    """
    تنسيق اسم الرمز حسب الوسيط.
    """
    if broker.lower() == "mt5":
        # لو رموز MT5 فيها m في النهاية زي BTCUSDm
        if symbol.upper() in ["BTCUSD", "ETHUSD", "XAUUSD"]:
            return symbol.upper() + "m"
        else:
            return symbol.upper()
    
    elif broker.lower() == "binance":
        return symbol.upper()  # Binance عادةً بدون تعديل
    
    else:
        return symbol.upper()

# ✅ أمثلة:
if __name__ == "__main__":
    print(format_symbol("btcusd", "mt5"))      # BTCUSDm
    print(format_symbol("xauusd", "mt5"))      # XAUUSDm
    print(format_symbol("ethusd", "binance"))  # ETHUSD
