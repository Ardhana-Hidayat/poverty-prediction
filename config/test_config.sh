#!/bin/bash
# config/test_config.sh
# Jalankan dari root project: bash config/test_config.sh

echo "=== Tes Koneksi Aiven PostgreSQL ==="

# Cek file .env
if [ ! -f ".env" ]; then
    echo "[FAIL] File .env tidak ditemukan"
    echo "       Jalankan: cp .env.example .env  lalu isi kredensialnya"
    exit 1
fi
echo "[OK]   File .env ditemukan"

# Cek python & dependencies
# python -c "import sqlalchemy, dotenv, psycopg2" 2>/dev/null
# if [ $? -ne 0 ]; then
#     echo "[FAIL] Dependensi kurang. Jalankan: pip install -r requirements.txt"
#     exit 1
# fi
# echo "[OK]   Dependensi tersedia"

# Tes koneksi
python - <<'EOF'
from config.database import get_engine
from sqlalchemy import text

try:
    engine = get_engine()
    with engine.connect() as conn:
        ver = conn.execute(text("SELECT version();")).fetchone()[0]
        print(f"[OK]   Koneksi berhasil!")
        print(f"       {ver[:60]}...")
except Exception as e:
    print(f"[FAIL] Koneksi gagal: {e}")
    exit(1)
EOF