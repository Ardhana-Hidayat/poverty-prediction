import os
from database import get_engine
from sqlalchemy import text

print("=== Tes Koneksi Aiven PostgreSQL ===")

# Cek file .env
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if not os.path.isfile(env_path):
    print("[FAIL] File .env tidak ditemukan")
    print("       Jalankan: cp .env.example .env  lalu isi kredensialnya")
    exit(1)
else:
    print("[OK]   File .env ditemukan")

try:
    engine = get_engine()
    with engine.connect() as conn:
        ver = conn.execute(text("SELECT version();")).fetchone()[0]
        print(f"[OK]   Koneksi berhasil!")
        print(f"       {ver[:60]}...")
except Exception as e:
    print(f"[FAIL] Koneksi gagal: {e}")
    exit(1)
