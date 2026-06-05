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
    .block-container { padding-top: 2.5rem; padding-bottom: 2rem; }
    [data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.5rem;
    }
    [data-testid="stSidebar"] { background-color: #F8FAFC; }

    .card {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .card h4 { margin: 0 0 0.3rem 0; color: #1E40AF; }
    .card p  { margin: 0; color: #374151; font-size: 0.92rem; }

    .card-red {
        background: #FFF1F2;
        border-left: 4px solid #E11D48;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .card-red h4 { margin: 0 0 0.3rem 0; color: #BE123C; }
    .card-red p  { margin: 0; color: #374151; font-size: 0.92rem; }

    .card-green {
        background: #F0FDF4;
        border-left: 4px solid #16A34A;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .card-green h4 { margin: 0 0 0.3rem 0; color: #15803D; }
    .card-green p  { margin: 0; color: #374151; font-size: 0.92rem; }

    .step-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        height: 100%;
    }
    .step-num   { font-size: 2.2rem; font-weight: 800; color: #2563EB; }
    .step-title { font-size: 1.05rem; font-weight: 600; color: #1E293B; margin: 0.4rem 0 0.3rem; }
    .step-desc  { font-size: 0.88rem; color: #64748B; }

    .hero { margin-bottom: 1.5rem; }
    .hero h1 { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.25rem; }
    .hero p  { font-size: 1.05rem; color: #64748B; }

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
    <p>Analisis sosial-ekonomi 30 provinsi Indonesia (2015–2025) berbasis <b>Machine Learning</b></p>
    <span class="badge">Data Engineering</span>
    <span class="badge">Semester 4</span>
    <span class="badge">BPS Indonesia</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Cara Menggunakan ──────────────────────────────────────────────────────────
st.markdown("### 🗺️ Cara Menggunakan Dashboard Ini")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="step-box">
        <div class="step-num">1️⃣</div>
        <div class="step-title">Pilih Halaman</div>
        <div class="step-desc">Gunakan menu di sidebar kiri untuk berpindah antar topik analisis.</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="step-box">
        <div class="step-num">2️⃣</div>
        <div class="step-title">Baca Kesimpulan Utama</div>
        <div class="step-desc">Setiap halaman memiliki kotak ringkasan di bagian atas — mulailah dari sana.</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="step-box">
        <div class="step-num">3️⃣</div>
        <div class="step-title">Cari Provinsi Anda</div>
        <div class="step-desc">Gunakan filter tabel di setiap halaman untuk mencari data provinsi tertentu.</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Kemampuan Dashboard ───────────────────────────────────────────────────────
st.markdown("### 🤖 Apa yang Bisa Dashboard Ini Lakukan?")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h4>📈 Prediksi Angka Kemiskinan</h4>
        <p>Memperkirakan <b>berapa persen penduduk miskin</b> di setiap provinsi
        berdasarkan data pendidikan dan ketenagakerjaan tahun-tahun sebelumnya.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card-red">
        <h4>🔴 Deteksi Daerah Berisiko</h4>
        <p>Mengidentifikasi provinsi yang <b>berisiko kemiskinan tinggi</b> (di atas 15%)
        — seperti sistem peringatan dini untuk pengambil kebijakan.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card-green">
        <h4>🔵 Peta Segmentasi Provinsi</h4>
        <p>Mengelompokkan provinsi ke dalam <b>segmen sosial-ekonomi</b> agar
        kebijakan dapat dirancang lebih tepat sasaran per kelompok.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Navigasi ──────────────────────────────────────────────────────────────────
st.markdown("### 📋 Halaman yang Tersedia")
col_nav, col_info = st.columns([1.2, 1])

with col_nav:
    st.markdown("""
    | Halaman | Isi |
    |---------|-----|
    | 🏠 **Home** | Halaman ini — panduan penggunaan |
    | 📋 **1 Overview** | Tren kemiskinan nasional & per provinsi |
    | 📈 **2 Prediksi Angka** | Seberapa akurat prediksi model? |
    | 🔴 **3 Deteksi Risiko** | Mana daerah yang perlu diwaspadai? |
    | 🔵 **4 Segmentasi** | Kelompok provinsi & karakteristiknya |
    """)

with col_info:
    st.markdown("""
    **Tentang Data:**
    - 📍 **30 Provinsi** (Papua DOB & structural break dikecualikan)
    - 📅 **Rentang:** 2015–2025
    - 💾 **Sumber:** BPS Indonesia via Aiven PostgreSQL
    - 📊 **Indikator:** Kemiskinan, Pendidikan (RLS), Pengangguran (TPT)
    """)

st.divider()

# ── FAQ ───────────────────────────────────────────────────────────────────────
with st.expander("❓ Pertanyaan yang Sering Ditanyakan"):
    st.markdown("""
    **Apa yang dimaksud "kemiskinan > 15%"?**
    > Artinya lebih dari 15 dari setiap 100 penduduk di provinsi tersebut hidup di bawah garis kemiskinan.
    > Angka ini digunakan sebagai ambang batas "Risiko Tinggi" dalam dashboard ini.

    ---
    **Apa itu RLS dan TPT?**
    > - **RLS (Rata-rata Lama Sekolah):** Rata-rata jumlah tahun penduduk usia 25+ menempuh pendidikan formal.
    > - **TPT (Tingkat Pengangguran Terbuka):** Persentase angkatan kerja yang sedang aktif mencari pekerjaan.

    ---
    **Mengapa Papua tidak termasuk?**
    > Empat provinsi pemekaran Papua 2022 (Papua Barat Daya, Papua Pegunungan, Papua Selatan, Papua Tengah)
    > dikecualikan karena datanya merupakan perkiraan, bukan hasil survei langsung.
    > Papua induk sejak 2024 juga dikecualikan karena terjadi perubahan struktur data yang signifikan.

    ---
    **Seberapa akurat prediksi model ini?**
    > Model prediksi angka rata-rata meleset sekitar 1–2 poin persen dari angka kemiskinan aktual.
    > Lihat halaman **📈 Prediksi Angka** untuk melihat detail akurasi per tahun dan per provinsi.
    """)

st.divider()
st.caption("Proyek Data Engineering — Semester 4 · Dataset: BPS Indonesia (2015–2025)")
