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
Proyek ini dikembangkan untuk menganalisis hubungan dan memprediksi tingkat kemiskinan di 38 provinsi di Indonesia berdasarkan indikator Rata-rata Lama Sekolah (RLS) dan Tingkat Pengangguran Terbuka (TPT) menggunakan data panel dari tahun 2015 hingga 2025. Melalui penerapan teknik Data Engineering, data diekstrak dari sumber resmi BPS, dibersihkan (transform), dan digabungkan menjadi dataset terpadu. Tujuan utamanya adalah menemukan korelasi serta membangun model machine learning (Regresi) untuk memprediksi persentase kemiskinan antarwilayah secara komprehensif.

### Manfaat Data / Use Case
- **Tujuan Proyek**: Membangun pipeline data terotomatisasi untuk memproses indikator makroekonomi dan sosial, serta mengimplementasikan model Machine Learning untuk menganalisis dan memprediksi profil kemiskinan tiap provinsi berdasarkan kualitas pendidikan dan serapan tenaga kerja.
- **Manfaat**: Memberikan wawasan berbasis data (*data-driven insights*) bagi pemerintah dan *stakeholders* terkait penyusunan kebijakan pengentasan kemiskinan. Menjadi referensi penelitian lanjutan di bidang ekonomi pembangunan, serta memberikan pengalaman praktis dalam integrasi *pipeline* ETL (Extract, Transform, Load) dengan arsitektur pemodelan Machine Learning dan analisis data panel.

### Serving Analisis
Analisis pada proyek ini terdiri dari beberapa tahapan utama, dimulai dari pengumpulan data mentah indikator RLS, TPT, dan Kemiskinan BPS (*Extract*). Selanjutnya, dilakukan penanganan *missing values* (terutama pada kasus khusus Daerah Otonomi Baru Papua), standarisasi nama provinsi, dan pembuatan fitur turunan seperti agregasi kesenjangan kemiskinan (*Transform*). Data yang telah terstandarisasi kemudian disimpan dalam format yang bersih (*Load*). Hasil akhirnya disajikan dalam bentuk statistik deskriptif dan visualisasi tren secara nasional yang digunakan untuk melihat disparitas antarprovinsi sebelum masuk ke tahap pemodelan prediktif.

### Serving Machine Learning
Dalam proyek ini, data bersih yang dihasilkan dari pipeline disajikan (*served*) untuk pemodelan Machine Learning dengan pendekatan:
1. **Regresi**: Memprediksi secara kontinu persentase kemiskinan provinsi menggunakan model regresi (seperti Random Forest) berdasarkan historis data Rata-rata Lama Sekolah (RLS) dan Tingkat Pengangguran Terbuka (TPT).
Tahapan ini memfasilitasi pengambilan keputusan strategis yang *data-driven* untuk melihat tren kerentanan dan prediksi tingkat kemiskinan per wilayah.

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
- **Alur Modular**: Pipeline mendefinisikan alur kerja jelas yang dimulai dari pembacaan dataset BPS mentah (*Extract*), dilanjutkan *profiling* & standarisasi (*Transform*), hingga eksport dataset bersih (*Load*). Data bersih tersebut kemudian dialirkan ke dalam modul Machine Learning (`/model/model_regresi.py`) untuk pelatihan model.
- **Teknologi yang Digunakan**:
  - **ETL & Data Processing**: Python, Pandas, NumPy
  - **Machine Learning**: Scikit-learn
  - **Database**: PostGreSQL (Aiven)
  - **Visualisasi**: Matplotlib, Seaborn, Streamlit

### Kode Program
**Struktur Kode**:
- Kode tersusun rapi berdasarkan tahapan dan fungsionalitas. Terdapat skrip khusus pengolahan data pipeline (`pipeline_etl.py`) serta fungsionalitas untuk model machine learning.
- Algoritma Machine Learning terdapat pada `model/model_regresi.py` yang melatih model regresi beserta tahapan evaluasinya.

**Link Proyek**:
- **ETL Pipeline**: https://github.com/Ardhana-Hidayat/poverty-prediction/blob/main/pipeline_etl.py
- **Machine Learning**: https://github.com/Ardhana-Hidayat/poverty-prediction/blob/main/model/model_regresi.py
- **Streamlit**: https://poverty-prediction-dashboard-app-ia8bwg.streamlit.app/

