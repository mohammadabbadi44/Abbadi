import os
import shutil
import time
from datetime import datetime

SOURCE_PATH = "logs/live_signal.json"
DEST_PATH = os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal\Common\Files\live_signal.json")

def copy_file_loop():
    while True:
        try:
            if os.path.exists(SOURCE_PATH):
                shutil.copy(SOURCE_PATH, DEST_PATH)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ تم نسخ live_signal.json إلى مجلد MT5")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ الملف غير موجود: {SOURCE_PATH}")
        except Exception as e:
            print(f"❌ خطأ أثناء النسخ: {e}")
        time.sleep(60)

if __name__ == "__main__":
    copy_file_loop()
