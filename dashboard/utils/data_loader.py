"""
data_loader.py
==============
Modul untuk memuat dataset dan model .pkl ke dalam dashboard Streamlit.

Prioritas sumber data:
  1. Streamlit Secrets (untuk Streamlit Cloud deployment)
  2. .env lokal via config.database (untuk development lokal)
  3. CSV lokal sebagai fallback terakhir
"""

import os
import sys
import pickle

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

# ── Path Setup ────────────────────────────────────────────────────────────────
# Root proyek = dua level di atas file ini (dashboard/utils/ → dashboard/ → root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Konstanta ─────────────────────────────────────────────────────────────────
PAPUA_DOB = [
    "PAPUA BARAT DAYA", "PAPUA PEGUNUNGAN",
    "PAPUA SELATAN",    "PAPUA TENGAH",
]
STRUCTURAL_BREAK = {"provinsi": "PAPUA", "tahun_min": 2024}

MODEL_PATHS = {
    "regresi"    : os.path.join(_PROJECT_ROOT, "model", "regresi",    "model_regresi_rf.pkl"),
    "klasifikasi": os.path.join(_PROJECT_ROOT, "model", "klasifikasi","model_klasifikasi_lr.pkl"),
    "clustering" : os.path.join(_PROJECT_ROOT, "model", "clustering", "model_clustering_kmeans.pkl"),
}
CSV_FALLBACK = os.path.join(_PROJECT_ROOT, "data", "dataset_cleaned.csv")


# ── Database Engine ───────────────────────────────────────────────────────────
def _build_engine():
    """Buat SQLAlchemy engine, coba Streamlit secrets dulu, lalu .env."""
    # 1) Streamlit Cloud: gunakan st.secrets
    try:
        db  = st.secrets["database"]
        uri = (
            f"postgresql://{db['user']}:{db['password']}"
            f"@{db['host']}:{db['port']}/{db['db_name']}"
            f"?sslmode={db.get('sslmode', 'require')}"
        )
        return create_engine(uri)
    except Exception:
        pass

    # 2) Development lokal: gunakan .env
    try:
        from config.database import get_database_uri
        return create_engine(get_database_uri())
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_engine():
    return _build_engine()


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """
    Muat dataset dari Aiven PostgreSQL.
    Fallback ke CSV lokal jika koneksi gagal.
    Menerapkan filter Papua DOB + structural break secara konsisten.
    """
    engine = get_engine()
    df     = None

    if engine is not None:
        try:
            df = pd.read_sql("SELECT * FROM poverty_panel_data", engine)
        except Exception as e:
            st.warning(f"⚠️ Koneksi database gagal: {e}. Menggunakan data lokal.")

    if df is None:
        if not os.path.exists(CSV_FALLBACK):
            st.error("❌ Data tidak ditemukan. Pastikan database atau CSV tersedia.")
            st.stop()
        df = pd.read_csv(CSV_FALLBACK)

    # Filter provinsi Papua DOB & structural break
    mask_dob = df["Provinsi"].isin(PAPUA_DOB)
    mask_sb  = (
        (df["Provinsi"] == STRUCTURAL_BREAK["provinsi"]) &
        (df["Tahun"]    >= STRUCTURAL_BREAK["tahun_min"])
    )
    df = df[~mask_dob & ~mask_sb].reset_index(drop=True)

    # Normalisasi format nama provinsi (Title Case)
    df["Provinsi"] = df["Provinsi"].str.strip().str.title()

    return df


# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """
    Muat model .pkl dari disk.
    model_name: 'regresi' | 'klasifikasi' | 'clustering'
    """
    path = MODEL_PATHS.get(model_name)
    if path is None or not os.path.exists(path):
        st.error(f"❌ File model tidak ditemukan: {path}")
        st.stop()

    with open(path, "rb") as f:
        return pickle.load(f)
