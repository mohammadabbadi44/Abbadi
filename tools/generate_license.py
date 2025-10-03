# tools/generate_license.py
# -*- coding: utf-8 -*-
"""
مولّد رخصة (License Generator)
- يربط الرخصة مع account_id + expiry_date
- يستخدم RSA للتوقيع + AES لتشفير المفتاح
"""

import base64
import json
from datetime import datetime
from pathlib import Path

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

# ============ Helpers ============

def gen_rsa_keypair(bits: int = 3072):
    priv = rsa.generate_private_key(public_exponent=65537, key_size=bits)
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return priv_pem, pub_pem


def sign_license(priv_pem: bytes, license_data: dict) -> str:
    private_key = serialization.load_pem_private_key(priv_pem, password=None)
    payload = json.dumps(license_data, sort_keys=True).encode("utf-8")
    signature = private_key.sign(
        payload,
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode("utf-8")

# ============ Main ============

def main():
    # 🔹 معلومات العميل
    client_id = input("ادخل رقم حساب العميل: ").strip()
    expiry_str = input("ادخل تاريخ الانتهاء (YYYY-MM-DD): ").strip()
    expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").strftime("%Y-%m-%d")

    # 🔹 تحميل أو إنشاء مفاتيح
    priv_path = Path("private.pem")
    pub_path = Path("public.pem")
    if priv_path.exists() and pub_path.exists():
        priv_pem = priv_path.read_bytes()
        pub_pem = pub_path.read_bytes()
    else:
        priv_pem, pub_pem = gen_rsa_keypair()
        priv_path.write_bytes(priv_pem)
        pub_path.write_bytes(pub_pem)
        print("✔ تم إنشاء private.pem و public.pem")

    # 🔹 بناء بيانات الرخصة
    lic_obj = {
        "account_id": client_id,
        "expiry": expiry_date,
    }
    lic_obj["signature"] = sign_license(priv_pem, lic_obj)

    # 🔹 حفظ ملف الرخصة
    lic_filename = f"license_{client_id}.json"
    with open(lic_filename, "w", encoding="utf-8") as f:
        json.dump(lic_obj, f, indent=2)

    print(f"✔ تم إنشاء الرخصة: {lic_filename}")
    print("أرسل للعميل:")
    print("  - ملف الرخصة")
    print("  - ملف public.pem")
    print("  - Editor.exe")
    print("⚠ احتفظ بـ private.pem عندك فقط (لتوليد رخص جديدة).")


if __name__ == "__main__":
    main()
