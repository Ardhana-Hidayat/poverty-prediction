# 📊 Analisis Data Sosial Ekonomi Indonesia 2015–2025

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.23+-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

*Analisis panel data sosial ekonomi 38 provinsi Indonesia selama 11 tahun*

</div>

---

## 📋 Project Information
| Kategori | Keterangan |
|:---|:---|
| **Project Name** | Analisis Data Sosial Ekonomi Indonesia 2015–2025 |
| **Created By** | Kelompok 3 Data Engineering |

---

## 📌 Table of Contents

1. [Executive Summary](#-executive-summary)
2. [Dataset](#-dataset)
3. [Struktur Proyek](#-struktur-proyek)
4. [Cara Menjalankan](#-cara-menjalankan)
5. [Fungsi Transformasi](#-fungsi-transformasi)
6. [Output](#-output)
7. [Temuan Utama](#-temuan-utama)
8. [Catatan Penting](#-catatan-penting)

---

## 🎯 Executive Summary

### Overview
Proyek ini menganalisis kondisi sosial ekonomi 38 provinsi di Indonesia selama periode **2015–2025** menggunakan tiga indikator utama dari BPS:

- 📚 **RLS** — Rata-rata Lama Sekolah (tahun)
- 💼 **TPT** — Tingkat Pengangguran Terbuka (%)
- 💰 **Kemiskinan** — Persentase penduduk miskin perkotaan & perdesaan (%)

### Tujuan
- Membaca, membersihkan, dan memvalidasi tiga master data panel tahunan
- Membangun dataset gabungan siap analisis dengan fitur turunan
- Mendokumentasikan tren dan disparitas antarprovinsi secara sistematis

### Timeline
```
Maret 2025 ──────────────────────────────── selesai
     │
     ├── Week 1 : Pembacaan & Profiling Data
     ├── Week 2 : Transformasi & Fitur Turunan
     ├── Week 3 : Validasi & Agregasi
     └── Week 4 : Dokumentasi & Output
```

### Stakeholders
| Role | Nama |
|:---|:---|
| Data Engineer | *Ardhana Syah Hidayat* |
| Data Analyst | *Iqbal Abdullah* |
| Project Manager | *Muhammad Adam Al Kidri* |

---

## 🗄️ Dataset

Seluruh data bersumber dari **Badan Pusat Statistik (BPS)** dengan cakupan 38 provinsi dan 11 tahun observasi.

### Master Data RLS
> Rata-rata Lama Sekolah per provinsi per tahun

| Properti | Detail |
|:---|:---|
| **File** | `Master_Data_RLS_2015_2025__1_.csv` |
| **Dimensi** | 418 baris × 3 kolom |
| **Kolom** | `Provinsi`, `RLS`, `Tahun` |
| **Rentang Nilai** | 4.76 – 11.58 tahun |
| **Missing** | 36 sel (8.6%) |
| **Completeness** | 91.4% |

---

### Master Data TPT
> Tingkat Pengangguran Terbuka — survei Februari & Agustus

| Properti | Detail |
|:---|:---|
| **File** | `Master_Data_TPT_2015_2025__1_.csv` |
| **Dimensi** | 418 baris × 4 kolom |
| **Kolom** | `Provinsi`, `TPT_Februari`, `TPT_Agustus`, `Tahun` |
| **Rentang Nilai** | 0.88% – 10.95% |
| **Missing** | 72 sel (8.6%) |
| **Completeness** | 91.4% |

---

### Master Data Kemiskinan
> Persentase penduduk miskin perkotaan & perdesaan, semester 1 & 2

| Properti | Detail |
|:---|:---|
| **File** | `Master_Data_Kemiskinan_2015_2025_Cleaned__1_.csv` |
| **Dimensi** | 418 baris × 8 kolom |
| **Kolom** | `Provinsi`, `Perkotaan_S1_Maret`, `Perkotaan_S2_Sept`, `Perdesaan_S1_Maret`, `Perdesaan_S2_Sept`, `Jumlah_S1_Maret`, `Jumlah_S2_Sept`, `Tahun` |
| **Rentang Nilai** | 0.00% – 39.14% |
| **Missing** | 338 sel (10.1%) |
| **Completeness** | 89.9% |

> ⚠️ **Catatan Missing Values:** Seluruh missing berasal dari 4 Daerah Otonomi Baru (DOB) hasil pemekaran Papua tahun 2022 — Papua Barat Daya, Papua Pegunungan, Papua Selatan, dan Papua Tengah. Bersifat **struktural**, bukan error input.

---

## 📁 Struktur Proyek

```
📦 project/
│
├── 📂 data/
│   ├── Master_Data_RLS_2015_2025.csv
│   ├── Master_Data_TPT_2015_2025.csv
│   └── Master_Data_Kemiskinan_2015_2025_Cleaned.csv
│
├── 📂 outputs/
│   ├── dataset_gabungan.csv
│   └── ringkasan_nasional.csv
│
├── analisis_data_colab.py
└── README.md
```

---

## 🚀 Cara Menjalankan

### Prasyarat
```
Python   >= 3.10
pandas   >= 1.5
numpy    >= 1.23
```

### Menggunakan Google Colab

**Step 1 — Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Step 2 — Set path folder project**
```python
BASE_PATH = "/content/drive/MyDrive/nama_folder/"

PATH_RLS = BASE_PATH + "Master_Data_RLS_2015_2025__1_.csv"
PATH_TPT = BASE_PATH + "Master_Data_TPT_2015_2025__1_.csv"
PATH_KEM = BASE_PATH + "Master_Data_Kemiskinan_2015_2025_Cleaned__1_.csv"
```

**Step 3 — Jalankan section secara berurutan**

| Section | Deskripsi |
|:---:|:---|
| 1 | Import library |
| 2 | Mount Drive & pembacaan data |
| 3 | Analisis struktur data |
| 4 | Data profiling |
| 5 | Standarisasi provinsi |
| 6 | Imputasi median |
| 7 | Normalisasi Min-Max |
| 8 | Fitur turunan & merge |
| 9 | Agregasi nasional |
| 10 | Deteksi outlier |
| 11 | Validasi transformasi |
| 12 | Simpan output ke Drive |

---

## ⚙️ Fungsi Transformasi

| Fungsi | Tujuan | Teknik |
|:---|:---|:---|
| `standarisasi_provinsi()` | Seragamkan nama provinsi | `strip()` + `upper()` |
| `imputasi_median_provinsi()` | Isi missing value | Group-wise median |
| `normalisasi_minmax()` | Skala 0–1 | Min-Max scaling |
| `tambah_fitur_turunan()` | Buat fitur baru + merge | Aritmetika & `pct_change` |
| `agregasi_nasional()` | Tren makro per tahun | `groupby` mean/min/max/std |
| `deteksi_outlier()` | Tandai nilai ekstrem | Metode IQR |

### Fitur Turunan yang Dibuat

| Fitur | Rumus | Interpretasi |
|:---|:---|:---|
| `TPT_Rata2` | `(TPT_Feb + TPT_Agt) / 2` | Rata-rata TPT sepanjang tahun |
| `Kemiskinan_Gap` | `Perdesaan_S1 − Perkotaan_S1` | Kesenjangan kemiskinan kota vs desa |
| `IPM_Proxy` | `RLS + (100 − TPT_Rata2)` | Skor proxy komposit kesejahteraan |
| `RLS_YoY_%` | `((RLS_t − RLS_{t-1}) / RLS_{t-1}) × 100` | Pertumbuhan RLS tahun ke tahun |

---

## 📤 Output

### File yang Dihasilkan

| File | Dimensi | Deskripsi |
|:---|:---:|:---|
| `dataset_gabungan.csv` | 418 × 15 | Merge 3 dataset + 4 fitur turunan |
| `ringkasan_nasional.csv` | 11 × 13 | Agregasi statistik per tahun nasional |

### Hasil Validasi

| Pemeriksaan | RLS | TPT | Kemiskinan |
|:---|:---:|:---:|:---:|
| Jumlah baris konsisten | ✅ | ✅ | ✅ |
| Tidak ada duplikat Provinsi-Tahun | ✅ | ✅ | ✅ |
| Tahun dalam rentang 2015–2025 | ✅ | ✅ | ✅ |
| Provinsi tidak mengandung null | ✅ | ✅ | ✅ |
| Bebas nilai negatif (kolom indikator) | ✅ | ✅ | ✅ |

---

## 📈 Temuan Utama

### Tren Nasional 2015–2025

| Indikator | 2015 | 2025 | Perubahan |
|:---|:---:|:---:|:---:|
| RLS rata-rata nasional | 8.52 tahun | 9.46 tahun | ⬆️ +0.93 tahun |
| TPT Agustus | 5.98% | 4.47% | ⬇️ −1.51 poin% |
| Kemiskinan Maret | 13.17% | 10.61% | ⬇️ −2.56 poin% |

> 📌 Anomali 2020–2021: TPT dan Kemiskinan sempat naik akibat dampak pandemi COVID-19 sebelum kembali membaik.

### Disparitas Antarprovinsi (RLS Rata-rata)

| Peringkat | Provinsi | RLS |
|:---:|:---|:---:|
| 🥇 Tertinggi | DKI Jakarta | 11.19 tahun |
| 🥈 | Papua Barat Daya | 10.54 tahun |
| 🥉 | Kepulauan Riau | 10.26 tahun |
| ⬇️ Terendah | Papua | 7.53 tahun |
| ⬇️ | Papua Tengah | 6.22 tahun |
| ⬇️ | Papua Pegunungan | 4.93 tahun |

---

## ⚠️ Catatan Penting

- **Missing by design:** 11 baris sisa setelah imputasi (Perdesaan DOB Papua) tidak dapat diimputasi karena tidak ada data historis referensi pada provinsi tersebut
- **IPM_Proxy bukan IPM resmi:** Tidak memperhitungkan harapan hidup dan pengeluaran per kapita — digunakan sebagai proxy eksplorasi awal
- **Left join pada merge:** Dipilih agar semua 418 baris RLS tetap terjaga; baris tanpa pasangan di TPT/KEM akan bernilai `NaN`
- **Output Colab bersifat sementara:** Simpan selalu ke Google Drive agar tidak terhapus saat sesi berakhir

---

<div align="center">

*Dokumentasi ini merupakan bagian dari proyek analisis data sosial ekonomi Indonesia 2015–2025*

</div>


# Data Engineering Kelompok 3

## Penerapan Machine Learning untuk Prediksi dan Segmentasi Tingkat Kemiskinan Berdasarkan Indikator Pendidikan dan Pengangguran di Indonesia

### Kontributor

| Nama Lengkap | NIM | Peran |
| :--- | :--- | :--- |
| Ardhana Syah Hidayat | 244311037 | Data Engineer |
| Iqbal Abdullah | 244311044 | Data Analyst |
| Muhammad Adam Al-kidri | 244311050 | Project Manager |

---

### Deskripsi Proyek
Proyek ini dikembangkan untuk menganalisis hubungan dan memprediksi tingkat kemiskinan di 38 provinsi di Indonesia berdasarkan indikator Rata-rata Lama Sekolah (RLS) dan Tingkat Pengangguran Terbuka (TPT) menggunakan data panel dari tahun 2015 hingga 2025. Melalui penerapan teknik Data Engineering, data diekstrak dari sumber resmi BPS, dibersihkan (transform), dan digabungkan menjadi dataset terpadu. Tujuan utamanya adalah menemukan korelasi serta membangun model machine learning (Clustering, Klasifikasi, dan Regresi) untuk mengelompokkan dan memprediksi status sosial ekonomi antarwilayah secara komprehensif.

### Manfaat Data / Use Case
- **Tujuan Proyek**: Membangun pipeline data terotomatisasi untuk memproses indikator makroekonomi dan sosial, serta mengimplementasikan model Machine Learning untuk menganalisis dan memprediksi profil kemiskinan tiap provinsi berdasarkan kualitas pendidikan dan serapan tenaga kerja.
- **Manfaat**: Memberikan wawasan berbasis data (*data-driven insights*) bagi pemerintah dan *stakeholders* terkait penyusunan kebijakan pengentasan kemiskinan. Menjadi referensi penelitian lanjutan di bidang ekonomi pembangunan, serta memberikan pengalaman praktis dalam integrasi *pipeline* ETL (Extract, Transform, Load) dengan arsitektur pemodelan Machine Learning dan analisis data panel.

### Serving Analisis
Analisis pada proyek ini terdiri dari beberapa tahapan utama, dimulai dari pengumpulan data mentah indikator RLS, TPT, dan Kemiskinan BPS (*Extract*). Selanjutnya, dilakukan penanganan *missing values* (terutama pada kasus khusus Daerah Otonomi Baru Papua), standarisasi nama provinsi, dan pembuatan fitur turunan seperti agregasi kesenjangan kemiskinan (*Transform*). Data yang telah terstandarisasi kemudian disimpan dalam format yang bersih (*Load*). Hasil akhirnya disajikan dalam bentuk statistik deskriptif dan visualisasi tren secara nasional yang digunakan untuk melihat disparitas antarprovinsi sebelum masuk ke tahap pemodelan prediktif.

### Serving Machine Learning
Dalam proyek ini, data bersih yang dihasilkan dari pipeline disajikan (*served*) untuk pemodelan Machine Learning yang dikategorikan dalam tiga pendekatan utama:
1. **Clustering (K-Means)**: Melakukan segmentasi 38 provinsi berdasarkan kerentanan sosial-ekonomi terkini (mengelompokkan provinsi dengan karakteristik tingkat kemiskinan dan pendidikan yang mirip).
2. **Regresi**: Memprediksi secara kontinu tingkat atau persentase kemiskinan provinsi di masa depan menggunakan historis RLS dan TPT.
3. **Klasifikasi**: Mengategorikan provinsi ke dalam kelas kerentanan tertentu berdasarkan ambang batas (*threshold*) kemiskinan.
Tahapan ini memfasilitasi pengambilan keputusan strategis yang *data-driven* untuk melihat tren kerentanan per wilayah.

### Pipeline

#### 1. Extract (Pengambilan Data)
**Sumber Data (Badan Pusat Statistik - BPS)**:
- **Data Rata-rata Lama Sekolah (RLS)** - https://www.bps.go.id/id/statistics-table/2/MTQyOSMy/rata-rata-lama-sekolah-penduduk-umur-15-tahun-ke-atas-menurut-provinsi.html
- **Data Tingkat Pengangguran Terbuka (TPT)** - https://www.bps.go.id/id/statistics-table/2/NTQzIzI=/tingkat-pengangguran-terbuka-menurut-provinsi--persen-.html
- **Persentase Penduduk Miskin** - https://www.bps.go.id/id/statistics-table/2/MTkyIzI=/persentase-penduduk-miskin--p0--menurut-provinsi-dan-daerah--persen-.html

**Metode Pengambilan**:
- Pengumpulan *Master Data* dalam format berkas CSV secara terstruktur dengan cakupan 38 provinsi pada rentang observasi tahun 2015–2025.

#### 2. Transform (Pembersihan & Transformasi)
**Pembersihan**:
- Menghapus baris duplikat atau menyesuaikan *missing values*, terutama dikarenakan pemekaran 4 Daerah Otonomi Baru (DOB) Papua yang secara struktural datanya kosong pada tahun-tahun awal.
- Menstandarkan penulisan nama provinsi (menggunakan metode `strip()` dan `upper()`) agar proses penggabungan tabel (*join*) antar 3 dataset konsisten.
- Melakukan imputasi menggunakan nilai median grup (*Group-wise median*) dan memvalidasi tipe numerik.

**Transformasi**:
- Menggabungkan (*merge*) ketiga dataset utama berdasarkan *primary key* `Provinsi` dan `Tahun`.
- Melakukan normalisasi data (seperti *Min-Max Scaling*) agar rentang skala indikator RLS dan TPT setara untuk pemodelan clustering.
- Melakukan *Feature Engineering* turunan, antara lain:
  - `TPT_Rata2`: Rata-rata dari TPT bulan Februari dan Agustus.
  - `Kemiskinan_Gap`: Selisih persentase kemiskinan wilayah pedesaan dengan perkotaan.
  - `IPM_Proxy`: Indeks proksi gabungan untuk kesejahteraan.

#### 3. Load (Pemindahan ke Target)
**Target**: Data yang telah terintegrasi di-*load* dan disimpan dalam bentuk `dataset_gabungan.csv` dan `ringkasan_nasional.csv` sebagai *Single Source of Truth* penyimpanan utama di Environment (seperti Google Drive / Local Storage).
**Skema Dataset Utama**: 
- `[Provinsi, Tahun, RLS, TPT_Februari, TPT_Agustus, Perkotaan_S1_Maret, Perdesaan_S1_Maret, ...]`
**Proses Load**:
- **Penyimpanan Awal**: Menyimpan data akhir dalam format `.csv` untuk memudahkan akses, dokumentasi, dan keperluan audit.
- **Validasi Kualitas (*Quality Control*)**: Sebelum data disalurkan, dilakukan pengecekan otomatis untuk memastikan:
  - Jumlah baris data konsisten (418 observasi dari kombinasi 38 provinsi × 11 tahun).
  - Tidak ada data yang terduplikasi pada kombinasi Provinsi dan Tahun.
  - Semua rentang nilai masuk akal (misalnya tidak ada angka persentase negatif).
- **Integrasi Machine Learning**: Data yang lolos validasi ini akan menjadi sumber data utama (*Single Source of Truth*) yang di-*load* langsung oleh *script* pelatihan Machine Learning.

### Arsitektur/Workflow ETL
- **Alur Modular**: Pipeline mendefinisikan alur kerja jelas yang dimulai dari pembacaan dataset BPS mentah (*Extract*), dilanjutkan *profiling* & standarisasi (*Transform*), hingga eksport dataset bersih (*Load*). Data bersih tersebut kemudian dialirkan ke dalam modul-modul Machine Learning (`/model/clustering`, `/model/regresi`, `/model/klasifikasi`) untuk pelatihan model.
- **Teknologi yang Digunakan**:
  - **ETL & Data Processing**: Python, Pandas, NumPy
  - **Machine Learning**: Scikit-learn
  - **Database**: PostGreSQL (Aiven)
  - **Visualisasi**: Matplotlib, Seaborn, Streamlit

### Kode Program
**Struktur Kode**:
- Kode tersusun rapi berdasarkan tahapan dan fungsionalitas. Terdapat skrip khusus pengolahan data pipeline (`analisis_data_colab.py`) serta direktori fungsional untuk model machine learning.
- Algoritma Machine Learning dipisahkan pada direktori seperti `model/clustering/model_clustering.py` dengan pipeline evaluasi model (seperti *Silhouette Score*).

**Link Proyek**:
- **ETL Pipeline**: 
- **Machine Learning**:
- **Streamlit**:

