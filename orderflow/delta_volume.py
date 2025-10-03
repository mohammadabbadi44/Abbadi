# orderflow/delta_volume.py

"""
حساب Delta Volume من دفتر الأوامر اللحظي
"""

def calculate_delta_volume(order_book: dict) -> dict:
    """
    يحسب Delta Volume و Imbalance من دفتر الأوامر

    Delta Volume = مجموع Bids - مجموع Asks
    Imbalance (%) = (Delta / Total Volume) * 100

    Args:
        order_book (dict): يحتوي على 'bids' و 'asks' بصيغة Binance

    Returns:
        dict: {
            'delta_volume': float,
            'bid_volume': float,
            'ask_volume': float,
            'imbalance_pct': float
        }
    """
    try:
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        bid_volume = sum(float(bid[1]) for bid in bids if float(bid[1]) > 0)
        ask_volume = sum(float(ask[1]) for ask in asks if float(ask[1]) > 0)

        total_volume = bid_volume + ask_volume
        delta = bid_volume - ask_volume
        imbalance_pct = (delta / total_volume) * 100 if total_volume > 0 else 0

        return {
            "delta_volume": round(delta, 2),
            "bid_volume": round(bid_volume, 2),
            "ask_volume": round(ask_volume, 2),
            "imbalance_pct": round(imbalance_pct, 2)
        }

    except Exception as e:
        try:
            from logs.logger import log_error
            log_error(f"Delta Volume Calculation Error: {e}")
        except:
            print(f"[❌] Delta Volume Error: {e}")

        return {
            "delta_volume": 0.0,
            "bid_volume": 0.0,
            "ask_volume": 0.0,
            "imbalance_pct": 0.0
        }
