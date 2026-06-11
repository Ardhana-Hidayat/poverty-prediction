"""
data_loader.py
==============
Memuat dataset dan model untuk dashboard Streamlit.

Prioritas sumber data:
  1. Streamlit Secrets (Streamlit Cloud)
  2. .env lokal via config.database (development lokal)
  3. CSV lokal sebagai fallback
"""

import os
import sys
import pickle

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

# ── Path Setup ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.constants import (
    CSV_PATH, MODEL_PATH, DATABASE_TABLE,
)


# ── Database Engine ───────────────────────────────────────────────────────────
def _build_engine():
    """Bangun SQLAlchemy engine: Streamlit secrets → .env lokal."""
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
    Muat dataset dari Aiven PostgreSQL, fallback ke CSV lokal.
    Filter Papua DOB dan structural break diterapkan secara konsisten.
    """
    engine = get_engine()
    df     = None

    if engine is not None:
        try:
            df = pd.read_sql(f"SELECT * FROM {DATABASE_TABLE}", engine)
        except Exception as e:
            st.warning(f"⚠️ Koneksi database gagal: {e}. Menggunakan data lokal.")

    if df is None:
        if not os.path.exists(CSV_PATH):
            st.error("❌ Data tidak ditemukan. Pastikan database atau CSV tersedia.")
            st.stop()
        df = pd.read_csv(CSV_PATH)

    df["Provinsi"] = df["Provinsi"].str.strip().str.title()
    return df


# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model() -> object:
    """Muat model Random Forest Regressor dari disk."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model tidak ditemukan: {MODEL_PATH}")
        st.error("Jalankan terlebih dahulu: python model/model_regresi.py")
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
