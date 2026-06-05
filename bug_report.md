# 🐛 Laporan Bug — Proyek `poverty-predictions`

Hasil pemeriksaan menyeluruh seluruh file Python pada proyek ini.

---

## 🔴 Bug Kritis (Dapat Menyebabkan Error / Data Salah)

### 1. `etl.py` — Nama kolom tidak konsisten dengan model
**File:** [etl.py](file:///d:/MCP/poverty-predictions/etl.py#L62-L66)

```diff
- rls_list.append(wide_to_long(df, "Rata_rata_Lama_Sekolah"))   # "rata" huruf kecil
+ rls_list.append(wide_to_long(df, "Rata_Rata_Lama_Sekolah"))   # harus "Rata" huruf besar
```

**Penjelasan:** Nama kolom yang dibuat oleh ETL adalah `"Rata_rata_Lama_Sekolah"` (huruf kecil di `rata`), sedangkan semua model, feature_engineering.py, dan data_loader.py menggunakan `"Rata_Rata_Lama_Sekolah"` (huruf besar di kedua kata). Ini akan menyebabkan `KeyError` saat dashboard mencoba mengakses kolom tersebut dari CSV fallback.

Baris yang bermasalah: **L62**
```python
rls_list.append(wide_to_long(df, "Rata_rata_Lama_Sekolah"))  # ❌ SALAH
```
Demikian juga di **L76**:
```python
df_rls = gabung_dan_agregasi(rls_list, "Rata_rata_Lama_Sekolah")  # ❌ SALAH
```

**Perbaikan:** Ubah ke `"Rata_Rata_Lama_Sekolah"` (R kapital kedua).

---

### 2. `etl.py` — Nama kolom Kemiskinan tidak konsisten
**File:** [etl.py](file:///d:/MCP/poverty-predictions/etl.py#L66-L78)

```diff
- kemiskinan_list.append(wide_to_long(df, "Persentase_Kemiskinan"))  # L66
- df_kemiskinan = gabung_dan_agregasi(kemiskinan_list, "Persentase_Kemiskinan")  # L78
```

**Penjelasan:** ETL menggunakan nama `"Persentase_Kemiskinan"`, sementara semua model menggunakan `"Persentase_Penduduk_Miskin"`. Ini akan menyebabkan kolom target tidak ditemukan.

---

### 3. `etl.py` — Path CSV output tidak sesuai dengan konstanta di `data_loader.py`
**File:** [etl.py](file:///d:/MCP/poverty-predictions/etl.py#L8) vs [data_loader.py](file:///d:/MCP/poverty-predictions/dashboard/utils/data_loader.py#L38)

```diff
# etl.py
CLEAN_PATH = "data/data_bersih.csv"          # ❌ nama file berbeda

# data_loader.py
CSV_FALLBACK = "...data/dataset_cleaned.csv"  # ✅ nama yang digunakan dashboard
```

**Penjelasan:** ETL menyimpan ke `data/data_bersih.csv`, tetapi dashboard mencari fallback di `data/dataset_cleaned.csv`. Jika database tidak tersedia, fallback ke CSV lokal akan gagal.

---

### 4. `model/regresi/model_regresi.py` — Path simpan model salah
**File:** [model_regresi.py](file:///d:/MCP/poverty-predictions/model/regresi/model_regresi.py#L37-L38)

```python
MODEL_DIR  = "model"                                         # ❌ Folder terlalu umum
MODEL_PATH = os.path.join(MODEL_DIR, "model_regresi_rf.pkl") # ❌ Tersimpan di model/ langsung
```

**Penjelasan:** Script regresi menyimpan model ke `model/model_regresi_rf.pkl`, tetapi `data_loader.py` mencari model di `model/regresi/model_regresi_rf.pkl`. Kemungkinan model sudah ada di lokasi yang benar (karena ada file `.pkl` di subdirektori), tapi jika script dijalankan ulang, model akan tersimpan di lokasi yang salah.

---

## 🟡 Bug Sedang (Potensi Masalah / Perilaku Tidak Terduga)

### 5. `2_Regresi.py` — Kalkulasi `per_year` berpotensi error di pandas versi baru
**File:** [2_Regresi.py](file:///d:/MCP/poverty-predictions/dashboard/pages/2_Regresi.py#L71-L82)

```python
per_year = (
    df_res.groupby("Tahun")
    .apply(lambda g: {           # ❌ apply mengembalikan dict → deprecated di pandas 2.x
        "N"   : len(g),
        ...
    })
    .apply(lambda x: x)          # ❌ double apply tidak diperlukan dan membingungkan
)
```

**Penjelasan:** Menggunakan `.apply()` yang mengembalikan dict sudah deprecated di pandas 2.0+ dan dapat menghasilkan `DeprecationWarning` atau error. Cara yang lebih baik menggunakan `.agg()` atau membuat DataFrame secara manual.

**Perbaikan yang Disarankan:**
```python
records = []
for tahun, g in df_res.groupby("Tahun"):
    records.append({
        "Tahun": tahun,
        "N": len(g),
        "RMSE": round(np.sqrt(mean_squared_error(g["Aktual (%)"], g["Prediksi (%)"])), 4),
        "MAE":  round(mean_absolute_error(g["Aktual (%)"], g["Prediksi (%)"]), 4),
        "R²":   round(r2_score(g["Aktual (%)"], g["Prediksi (%)"]), 4),
    })
per_year_df = pd.DataFrame(records)
```

---

### 6. `feature_engineering.py` — `build_features` drop NaN menggunakan `FEATURES + [TARGET]` tapi bukan `TARGET_CLASS`
**File:** [feature_engineering.py](file:///d:/MCP/poverty-predictions/dashboard/utils/feature_engineering.py#L48-L50)

```python
df[TARGET_CLASS] = (df[TARGET] > THRESHOLD).astype(int)
return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
```

**Penjelasan:** `TARGET_CLASS` (`Risiko_Tinggi`) dibuat dari `TARGET`, jadi jika `TARGET` ada nilainya, `TARGET_CLASS` pasti tidak NaN. Ini tidak bug fatal, tetapi bisa lebih eksplisit.

---

### 7. `3_Klasifikasi.py` — `decision_function` menghasilkan nilai yang bukan probabilitas
**File:** [3_Klasifikasi.py](file:///d:/MCP/poverty-predictions/dashboard/pages/3_Klasifikasi.py) dan [feature_engineering.py L91](file:///d:/MCP/poverty-predictions/dashboard/utils/feature_engineering.py#L88-L96)

```python
else:
    y_prob = model.decision_function(df_feat[FEATURES])  # ❌ Bukan probabilitas 0-1!
```

**Penjelasan:** Jika model tidak memiliki `predict_proba`, digunakan `decision_function` yang menghasilkan nilai yang **tidak dalam rentang 0–1**. Kemudian di halaman 3, nilai ini langsung dikalikan 100 (`y_prob * 100`) dan ditampilkan sebagai "Prob. Risiko (%)". Ini akan menghasilkan angka yang tidak masuk akal (bisa negatif atau > 100%). Logistic Regression dengan Pipeline seharusnya punya `predict_proba`, tapi fallback ini rawan jika model diganti.

---

### 8. `4_Clustering.py` — Radar chart akan error jika hanya ada 1 cluster
**File:** [4_Clustering.py](file:///d:/MCP/poverty-predictions/dashboard/pages/4_Clustering.py#L173-L181)

```python
radar_raw = cluster_means[CLUSTERING_FEATURES].copy()
scaler_r = MinMaxScaler()
radar_norm = pd.DataFrame(
    scaler_r.fit_transform(radar_raw),  # ❌ Jika K=1, MinMaxScaler akan error (range=0)
    ...
)
```

**Penjelasan:** Jika hanya ada 1 cluster (K=1), `MinMaxScaler` akan menghasilkan NaN karena range = 0. Namun karena `K_MIN = 2` di training, ini tidak akan terjadi dalam kondisi normal. Tetap perlu guard jika model diganti.

---

### 9. `4_Clustering.py` — Kolom `cluster_means` diubah namanya, tetapi dipakai lagi dengan nama lama
**File:** [4_Clustering.py](file:///d:/MCP/poverty-predictions/dashboard/pages/4_Clustering.py#L154-L173)

```python
cluster_means = df_clust.groupby("Cluster")[CLUSTERING_FEATURES].mean().round(2)
cluster_counts = df_clust["Cluster"].value_counts().sort_index()
cluster_means.insert(0, "Jumlah Provinsi", cluster_counts)
cluster_means.index = [f"Cluster {i}" for i in cluster_means.index]  # Index berubah!

st.dataframe(
    cluster_means.rename(columns={...}),  # OK, ini ditampilkan dengan nama baru
)

# Tapi di bawahnya:
radar_raw = cluster_means[CLUSTERING_FEATURES].copy()  # ❌ Akses kolom ORIGINAL meski sudah rename index
```

**Penjelasan:** `CLUSTERING_FEATURES` masih bisa diakses karena hanya *index* yang berubah, bukan nama kolom — jadi tidak error. Namun kolom "Jumlah Provinsi" juga ikut di `cluster_means` ketika dipakai untuk radar. Baris berikut memilih hanya `CLUSTERING_FEATURES`, sehingga kolom "Jumlah Provinsi" tidak ikut — ini aman. **Tidak bug fatal**, tapi kode sedikit membingungkan.

---

## 🔵 Catatan / Peningkatan Kode (Code Smell)

### 10. `2_Regresi.py` — Import `pandas` dua kali
**File:** [2_Regresi.py](file:///d:/MCP/poverty-predictions/dashboard/pages/2_Regresi.py#L81) dan [L158](file:///d:/MCP/poverty-predictions/dashboard/pages/2_Regresi.py#L158)

```python
import pandas as pd   # L81 — di tengah fungsi
...
import pandas as pd   # L158 — di tengah kode lagi
```
Python tidak error karena import duplicate hanya mengacu modul yang sama, tapi ini tidak rapi. Import seharusnya di awal file.

---

### 11. `data_loader.py` — Filter provinsi menggunakan case-sensitive string `"PAPUA"` sebelum normalisasi Title Case
**File:** [data_loader.py](file:///d:/MCP/poverty-predictions/dashboard/utils/data_loader.py#L93-L101)

```python
mask_sb = (
    (df["Provinsi"] == STRUCTURAL_BREAK["provinsi"])  # "PAPUA" — uppercase
    ...
)
df = df[~mask_dob & ~mask_sb].reset_index(drop=True)

# Normalisasi SETELAH filter
df["Provinsi"] = df["Provinsi"].str.strip().str.title()  # Jadi "Papua"
```

**Penjelasan:** Filter dilakukan dulu dengan string `"PAPUA"` (uppercase), lalu normalisasi ke Title Case. Jika data dari database sudah di-title case, filter `PAPUA` tidak akan cocok. Sebaiknya filter dilakukan **setelah** normalisasi, atau gunakan case-insensitive comparison.

---

## ✅ Ringkasan Bug

| # | Tingkat | File | Deskripsi |
|---|---------|------|-----------|
| 1 | 🔴 Kritis | `etl.py` | Nama kolom RLS tidak konsisten (`Rata_rata` vs `Rata_Rata`) |
| 2 | 🔴 Kritis | `etl.py` | Nama kolom kemiskinan tidak konsisten (`Persentase_Kemiskinan` vs `Persentase_Penduduk_Miskin`) |
| 3 | 🔴 Kritis | `etl.py` | Path CSV output tidak cocok dengan path CSV fallback di dashboard |
| 4 | 🟡 Sedang | `model_regresi.py` | Path simpan model salah (jika dijalankan ulang) |
| 5 | 🟡 Sedang | `2_Regresi.py` | `groupby.apply` dengan dict — deprecated di pandas 2.x |
| 6 | 🟡 Sedang | `feature_engineering.py` | Fallback `decision_function` bukan probabilitas sejati |
| 7 | 🟡 Sedang | `4_Clustering.py` | Tidak ada guard untuk K=1 pada MinMaxScaler radar |
| 8 | 🔵 Ringan | `data_loader.py` | Filter provinsi case-sensitive dilakukan sebelum normalisasi |
| 9 | 🔵 Ringan | `2_Regresi.py` | Import `pandas` dobel di tengah file |
