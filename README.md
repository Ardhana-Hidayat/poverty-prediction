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

| | |
|:---|:---|
| **Project Name** | Analisis Data Sosial Ekonomi Indonesia 2015–2025 |
| **Created By** | Tim Analisis Data |
| **Date** | March 10, 2025 |
| **Version** | 1.0 |

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
| Data Engineer | *(nama)* |
| Data Analyst | *(nama)* |
| Project Manager | *(nama)* |

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
│   ├── Master_Data_RLS_2015_2025__1_.csv
│   ├── Master_Data_TPT_2015_2025__1_.csv
│   └── Master_Data_Kemiskinan_2015_2025_Cleaned__1_.csv
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
