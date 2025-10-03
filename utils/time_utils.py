from datetime import datetime

def format_timestamp(ts: str) -> str:
    """
    يحول timestamp إلى صيغة مقروءة (YYYY-MM-DD HH:MM:SS)

    يقبل string بصيغة ISO 8601 أو كائن datetime مباشرة
    """
    try:
        # إذا كان datetime فعلاً
        if isinstance(ts, datetime):
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        
        # إذا كان string بصيغة ISO
        elif isinstance(ts, str):
            dt = datetime.fromisoformat(ts)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # fallback: رجّعها نص عادي
        else:
            return str(ts)

    except Exception:
        return str(ts)
