"""
app.py — Entry point Dashboard Streamlit
=========================================
Halaman utama (Home) dashboard prediksi kemiskinan Indonesia.
Jalankan dengan: streamlit run dashboard/app.py
"""

import os
import sys

# Tambahkan direktori dashboard ke path agar utils bisa diimport dari pages
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

import streamlit as st

# ── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Kemiskinan Indonesia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hapus padding atas berlebih */
    .block-container { padding-top: 2.5rem; padding-bottom: 2rem; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.5rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #F8FAFC; }

    /* Info card kustom */
    .card {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .card h4 { margin: 0 0 0.3rem 0; color: #1E40AF; }
    .card p  { margin: 0; color: #374151; font-size: 0.92rem; }

    /* Hero */
    .hero { margin-bottom: 1.5rem; }
    .hero h1 { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.25rem; }
    .hero p  { font-size: 1.05rem; color: #64748B; }

    /* Badge */
    .badge {
        display: inline-block;
        background: #DBEAFE;
        color: #1E40AF;
        border-radius: 9999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📊 Dashboard Prediksi Kemiskinan Indonesia</h1>
    <p>Analisis sosial-ekonomi provinsi Indonesia (2015–2025) berbasis <b>Machine Learning</b></p>
    <span class="badge">Data Engineering</span>
    <span class="badge">Semester 4</span>
    <span class="badge">BPS Indonesia</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Cards Model ───────────────────────────────────────────────────────────────
st.markdown("### 🤖 Model yang Digunakan")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h4>📈 Regresi — Random Forest</h4>
        <p>Memprediksi <b>persentase kemiskinan</b> (nilai kontinu) menggunakan
        fitur pendidikan dan ketenagakerjaan. Dievaluasi dengan RMSE, MAE, dan R².</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h4>🔴 Klasifikasi — Logistic Regression</h4>
        <p>Mengklasifikasikan provinsi sebagai <b>Risiko Tinggi</b> (kemiskinan > 15%)
        atau <b>Aman</b>. Dievaluasi dengan F1-Score dan ROC AUC.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h4>🔵 Clustering — K-Means</h4>
        <p>Mengelompokkan provinsi ke dalam <b>segmen sosial-ekonomi</b> berdasarkan
        kondisi terkini. Jumlah cluster dipilih berdasarkan Silhouette Score tertinggi.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Panduan Navigasi ──────────────────────────────────────────────────────────
st.markdown("### 🗺️ Panduan Navigasi")

col_nav, col_info = st.columns([1.2, 1])
with col_nav:
    st.markdown("""
    Gunakan **sidebar di sebelah kiri** untuk berpindah antar halaman:

    | Halaman | Isi |
    |---------|-----|
    | 🏠 **Home** | Halaman ini — ringkasan proyek |
    | 📋 **1 Overview** | Eksplorasi dataset & tren kemiskinan per provinsi |
    | 📈 **2 Regresi** | Hasil prediksi numerik kemiskinan + feature importance |
    | 🔴 **3 Klasifikasi** | Prediksi risiko tinggi + confusion matrix + ROC curve |
    | 🔵 **4 Clustering** | Segmentasi provinsi + karakteristik tiap cluster |
    """)

with col_info:
    st.markdown("""
    **Tentang Data:**
    - 📍 **30 Provinsi** (Papua DOB & structural break dikecualikan)
    - 📅 **Rentang:** 2015–2025
    - 💾 **Sumber:** Aiven PostgreSQL (tabel `poverty_panel_data`)
    - 📊 **Indikator:** Kemiskinan, Pendidikan (RLS), Pengangguran (TPT)
    """)

st.divider()
st.caption("Proyek Data Engineering — Semester 4 · Dataset: BPS Indonesia (2015–2025)")
