"""
feature_engineering.py
======================
Fungsi rekayasa fitur yang konsisten dengan logika training model.
Digunakan oleh semua halaman dashboard untuk mempersiapkan data prediksi.
"""

import numpy as np
import pandas as pd

# ── Konstanta ─────────────────────────────────────────────────────────────────
FEATURES = [
    "Rata_Rata_Lama_Sekolah",
    "Tingkat_Pengangguran_Terbuka",
    "RLS_lag1",
    "TPT_lag1",
    "RLS_YoY",
    "TPT_YoY",
    "RLS_x_TPT",
    "Tahun_norm",
]
TARGET       = "Persentase_Penduduk_Miskin"
TARGET_CLASS = "Risiko_Tinggi"
THRESHOLD    = 15.0

CLUSTERING_FEATURES = [
    "Persentase_Penduduk_Miskin",
    "Rata_Rata_Lama_Sekolah",
    "Tingkat_Pengangguran_Terbuka",
]


# ── Feature Engineering Utama ─────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buat fitur lag, YoY, interaksi, dan normalisasi tahun.
    Konsisten dengan logika di model_regresi.py & model_klasifikasi.py.
    Baris dengan NaN pada fitur atau target akan dihapus.
    """
    df = df.sort_values(["Provinsi", "Tahun"]).copy()

    df["RLS_lag1"]   = df.groupby("Provinsi")["Rata_Rata_Lama_Sekolah"].shift(1)
    df["TPT_lag1"]   = df.groupby("Provinsi")["Tingkat_Pengangguran_Terbuka"].shift(1)
    df["RLS_YoY"]    = df.groupby("Provinsi")["Rata_Rata_Lama_Sekolah"].pct_change() * 100
    df["TPT_YoY"]    = df.groupby("Provinsi")["Tingkat_Pengangguran_Terbuka"].pct_change() * 100
    df["RLS_x_TPT"]  = df["Rata_Rata_Lama_Sekolah"] * df["Tingkat_Pengangguran_Terbuka"]
    df["Tahun_norm"] = (df["Tahun"] - 2015) / 10
    df[TARGET_CLASS] = (df[TARGET] > THRESHOLD).astype(int)

    return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


# ── Prediksi Regresi ──────────────────────────────────────────────────────────
def predict_regression(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Terapkan model regresi (Random Forest) ke seluruh data dengan fitur lengkap.
    Target di-log1p saat training → hasil prediksi di-expm1 kembali.

    Returns:
        DataFrame dengan kolom Tahun, Provinsi, Aktual (%), Prediksi (%), Error.
    """
    df_feat = build_features(df)

    y_pred = np.expm1(model.predict(df_feat[FEATURES]))
    y_true = df_feat[TARGET].values

    result = df_feat[["Tahun", "Provinsi"]].copy()
    result["Aktual (%)"]    = y_true.round(2)
    result["Prediksi (%)"]  = y_pred.round(2)
    result["Error (poin%)"] = (y_true - y_pred).round(2)

    return result.reset_index(drop=True)


# ── Prediksi Klasifikasi ──────────────────────────────────────────────────────
def predict_classification(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Terapkan model klasifikasi (Logistic Regression Pipeline) ke seluruh data.
    Menghasilkan label prediksi dan probabilitas risiko tinggi.

    Returns:
        DataFrame dengan kolom Tahun, Provinsi, Aktual, Prediksi, Prob. Risiko (%).
    """
    df_feat = build_features(df)

    y_pred = model.predict(df_feat[FEATURES])

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(df_feat[FEATURES])[:, 1]
    else:
        y_prob = model.decision_function(df_feat[FEATURES])

    result = df_feat[["Tahun", "Provinsi"]].copy()
    result["Aktual"]           = df_feat[TARGET_CLASS].values.astype(int)
    result["Prediksi"]         = y_pred.astype(int)
    result["Prob. Risiko (%)"] = (y_prob * 100).round(1)
    result["Aktual Label"]     = result["Aktual"].map({1: "Risiko Tinggi", 0: "Aman"})
    result["Prediksi Label"]   = result["Prediksi"].map({1: "Risiko Tinggi", 0: "Aman"})

    return result.reset_index(drop=True)


# ── Clustering ────────────────────────────────────────────────────────────────
def apply_clustering(df: pd.DataFrame, model_pipeline: dict) -> pd.DataFrame:
    """
    Terapkan model clustering K-Means pada kondisi terkini tiap provinsi.
    Menggunakan data tahun terakhir per provinsi.

    model_pipeline: dict berisi keys 'scaler', 'model', 'features'

    Returns:
        DataFrame dengan kolom Provinsi, Tahun, fitur, dan Cluster.
    """
    df_sorted = df.sort_values(["Provinsi", "Tahun"])
    df_latest = df_sorted.groupby("Provinsi").last().reset_index()
    df_feat   = df_latest[["Provinsi", "Tahun"] + CLUSTERING_FEATURES].dropna().reset_index(drop=True)

    scaler   = model_pipeline["scaler"]
    model    = model_pipeline["model"]
    X_scaled = scaler.transform(df_feat[CLUSTERING_FEATURES])

    df_feat["Cluster"] = model.predict(X_scaled)
    return df_feat
