# Analisis dan Prediksi Indeks Kemiskinan Berbasis Rata-rata Lama Sekolah dan Tingkat Pengangguran Terbuka pada 38 Provinsi di Indonesia (2015-2025)

Proyek ini dibangun oleh **Kelompok 3** untuk memenuhi tugas mata kuliah Data Engineering. Proyek ini mengintegrasikan pipeline data (ETL) modular berbasis Python dan Machine Learning untuk memprediksi persentase kemiskinan di 38 provinsi di Indonesia berdasarkan indikator Rata-rata Lama Sekolah (RLS) dan Tingkat Pengangguran Terbuka (TPT).

---

## 🚀 Arsitektur & Struktur Repositori

Repositori ini disusun secara ringkas dengan struktur berikut:

```text
poverty-predictions/
├── .gitignore                 # File konfigurasi Git ignore
├── README.md                  # Dokumentasi proyek (file ini)
├── requirements.txt           # Daftar pustaka Python yang dibutuhkan
│
├── config/                    # Modul Konfigurasi
│   └── database.py            # Utilitas koneksi Aiven PostgreSQL
│
├── data/                      # Direktori data (diabaikan oleh Git kecuali .gitkeep)
│   ├── raw/                   # Dataset CSV asli dari BPS (33 file)
│   └── data_bersih.csv        # Backup hasil join akhir format CSV
│
├── model/                     # Penyimpanan model ML terlatih
│   └── model_kemiskinan.pkl   # File model ML yang sudah dilatih (.pkl)
│
└── notebooks/                 # Eksperimen menggunakan Jupyter Notebook
    ├── 01_pipeline_etl.ipynb  # ETL Pipeline (Extract -> Transform -> Load ke Aiven)
    └── 02_machine_learning.ipynb # Analisis (EDA) & Random Forest (Iqbal & Adam)
```

---

## 🛠️ Langkah Memulai (Setup Lokal)

### 1. Kloning Repositori & Persiapan Environment
Pastikan Anda memiliki Python 3.8+ terinstal di sistem Anda.

```bash
# 1. Buat virtual environment
python -m venv .venv

# 2. Aktifkan virtual environment
# Di Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Di macOS/Linux:
source .venv/bin/activate

# 3. Instal dependensi yang diperlukan
pip install -r requirements.txt
```

### 2. Konfigurasi Database (Aiven PostgreSQL)
1. Salin berkas `.env.example` menjadi `.env` (jika ada) atau buat file `.env` baru di root direktori.
2. Buka berkas `.env` dan masukkan kredensial database Cloud Aiven PostgreSQL yang Anda miliki.

---

## 📈 Alur ETL & Eksekusi

### 1. Menjalankan ETL Pipeline (01_pipeline_etl.ipynb)
1. Letakkan 33 berkas CSV asli dari BPS ke dalam folder `data/raw/`.
2. Buka dan jalankan Jupyter Notebook [01_pipeline_etl.ipynb](notebooks/01_pipeline_etl.ipynb).
3. Notebook ini akan mengekstrak data dari `data/raw/`, melakukan transformasi data panel (pembersihan provinsi, penggabungan RLS, TPT, dan Kemiskinan), menyimpan hasil bersih ke `data/data_bersih.csv`, dan mengunggahnya ke database cloud Aiven PostgreSQL.

### 2. Analisis & Pemodelan Machine Learning (02_machine_learning.ipynb)
1. Buka dan jalankan Jupyter Notebook [02_machine_learning.ipynb](notebooks/02_machine_learning.ipynb).
2. Notebook ini akan membaca data bersih dari `data/data_bersih.csv`, melakukan Exploratory Data Analysis (EDA) sederhana, serta melatih model **Random Forest Regressor** untuk memprediksi persentase kemiskinan.
3. Model yang berhasil dilatih akan disimpan secara otomatis di folder `model/model_kemiskinan.pkl`.

---

## 👥 Kelompok 3 - Anggota Tim
- [Nama Anggota 1] - Data Engineer
- [Nama Anggota 2] - Data Scientist / ML Engineer
- [Nama Anggota 3] - Business Intelligence Analyst
*(Sesuaikan dengan nama anggota kelompok Anda)*
