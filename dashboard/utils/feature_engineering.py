"""
feature_engineering.py
======================
Fungsi prediksi yang konsisten dengan logika training model_regresi.py.
Fitur: Rata_Rata_Lama_Sekolah, Tingkat_Pengangguran_Terbuka, Tahun
Split: 80/20 random (shuffle=True, random_state=42)
"""

import numpy as np
import pandas as pd

# ── Konstanta ─────────────────────────────────────────────────────────────────
FEATURES = [
    "Rata_Rata_Lama_Sekolah",
    "Tingkat_Pengangguran_Terbuka",
    "Tahun",
]
TARGET = "Persentase_Penduduk_Miskin"


# ── Prediksi Regresi ──────────────────────────────────────────────────────────
def predict_regression(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Terapkan model Random Forest Regressor ke seluruh data.
    Fitur: Rata_Rata_Lama_Sekolah, Tingkat_Pengangguran_Terbuka, Tahun.

    Returns:
        DataFrame dengan kolom: Tahun, Provinsi, Aktual (%), Prediksi (%), Error (poin%).
    """
    # Pastikan tidak ada kolom duplikat saat memilih fitur untuk prediksi.
    df_feat = df[["Provinsi"] + FEATURES + [TARGET]].dropna().reset_index(drop=True)

    y_pred = model.predict(df_feat[FEATURES])
    y_true = df_feat[TARGET].values

    result = df_feat[["Tahun", "Provinsi"]].copy()
    result["Aktual (%)"]    = y_true.round(2)
    result["Prediksi (%)"]  = y_pred.round(2)
    result["Error (poin%)"] = (y_true - y_pred).round(2)

    return result.reset_index(drop=True)
