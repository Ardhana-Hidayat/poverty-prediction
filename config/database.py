import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Memuat file .env yang berada di root direktori proyek
load_dotenv()

def get_database_uri():
    """
    Membangun dan mengembalikan URI koneksi database dari variabel lingkungan (.env).
    """
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "25060")
    db_name = os.getenv("DB_NAME", "defaultdb")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    sslmode = os.getenv("DB_SSLMODE", "require")
    
    if not all([host, user, password]):
        raise ValueError(
            "Kredensial database tidak lengkap di file .env.\n"
            "Pastikan Anda telah menyalin .env.example menjadi .env "
            "dan mengisi variabel DB_HOST, DB_USER, serta DB_PASSWORD."
        )
        
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}?sslmode={sslmode}"

def get_engine():
    """
    Membuat dan mengembalikan SQLAlchemy engine untuk koneksi PostgreSQL Aiven.
    """
    try:
        uri = get_database_uri()
        # Menggunakan psycopg2-binary yang sudah didaftarkan di requirements.txt
        engine = create_engine(uri)
        return engine
    except Exception as e:
        print(f"Error saat membuat engine database: {e}")
        raise e