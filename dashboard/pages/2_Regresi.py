"""
2_Regresi.py — Prediksi Angka Kemiskinan
==========================================
Hasil prediksi persentase kemiskinan menggunakan Random Forest.
"""

import os, sys
_DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.data_loader         import load_raw_data, load_model
from utils.feature_engineering import predict_regression, FEATURES

# ── Label Ramah Pengguna ──────────────────────────────────────────────────────
FEATURE_LABELS = {
    "Rata_Rata_Lama_Sekolah":       "Rata-rata Lama Sekolah",
    "Tingkat_Pengangguran_Terbuka": "Tingkat Pengangguran",
    "RLS_lag1":                     "Lama Sekolah (Tahun Lalu)",
    "TPT_lag1":                     "Pengangguran (Tahun Lalu)",
    "RLS_YoY":                      "Perubahan Lama Sekolah (%)",
    "TPT_YoY":                      "Perubahan Pengangguran (%)",
    "RLS_x_TPT":                    "Interaksi Pendidikan × Pengangguran",
    "Tahun_norm":                   "Tren Waktu",
}

# ── Konfigurasi ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Angka | Dashboard Kemiskinan",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Seberapa Akurat Prediksi Angka Kemiskinan?")
st.caption("Model memperkirakan persentase penduduk miskin di setiap provinsi berdasarkan data historis")

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Memuat data dan model..."):
    df    = load_raw_data()
    model = load_model("regresi")

@st.cache_data(show_spinner=False)
def get_regression_results(_df, _model):
    return predict_regression(_df, _model)

with st.spinner("Menghitung prediksi..."):
    df_res = get_regression_results(df, model)

# ── Hitung Metrik ─────────────────────────────────────────────────────────────
y_true = df_res["Aktual (%)"].values
y_pred = df_res["Prediksi (%)"].values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)
r2_pct = r2 * 100

# ── Narasi Otomatis ───────────────────────────────────────────────────────────
quality = "sangat baik ✅" if r2 >= 0.90 else "baik ✅" if r2 >= 0.75 else "cukup 🟡"

st.success(
    f"📌 **Kesimpulan:** Model ini memiliki akurasi **{r2_pct:.1f}%** dalam menjelaskan "
    f"perbedaan kemiskinan antar provinsi — performa {quality}. "
    f"Rata-rata prediksi meleset sekitar **{mae:.2f} poin%** dari angka sebenarnya. "
    f"Artinya jika kemiskinan sebenarnya 12%, prediksi model berkisar antara "
    f"**{max(0, 12 - mae):.1f}%** hingga **{12 + mae:.1f}%**."
)

st.divider()

# ── Metrik Kunci ──────────────────────────────────────────────────────────────
st.markdown("### 📊 Performa Model Secara Keseluruhan")
col1, col2, col3 = st.columns(3)

col1.metric(
    "Akurasi Model",
    f"{r2_pct:.1f}%",
    help="Seberapa besar variasi kemiskinan antar provinsi yang berhasil dijelaskan model. "
         "Semakin tinggi semakin baik (100% = sempurna).",
)
col2.metric(
    "Rata-rata Kesalahan Prediksi",
    f"{mae:.2f} poin%",
    help="Rata-rata selisih antara prediksi model dan angka kemiskinan aktual. "
         "Semakin kecil semakin baik.",
)
col3.metric(
    "Batas Kesalahan Maksimal",
    f"{rmse:.2f} poin%",
    help="Ukuran kesalahan yang memberi penalti lebih besar pada kesalahan besar. "
         "Semakin kecil semakin baik.",
)

with st.expander("🔬 Lihat Detail Teknis Model"):
    st.markdown(f"""
    | Metrik Teknis | Nilai | Penjelasan |
    |---------------|-------|-----------|
    | R² (Koefisien Determinasi) | `{r2:.4f}` | Mendekati 1.0 = sempurna |
    | MAE (Mean Absolute Error)  | `{mae:.4f} poin%` | Rata-rata kesalahan absolut |
    | RMSE (Root Mean Sq. Error) | `{rmse:.4f} poin%` | Penalti lebih besar untuk error besar |

    **Algoritma:** Random Forest Regressor (n_estimators=200, max_depth=6, min_samples_leaf=3)
    **Target saat training:** log1p(kemiskinan) → dibalik dengan expm1 ke skala asli
    **Metode validasi:** Walk-forward (train tahun < t, test tahun t) mulai tahun 2020
    """)

st.divider()

# ── Scatter Aktual vs Prediksi ────────────────────────────────────────────────
st.markdown("### 🎯 Prediksi vs Kenyataan")
st.caption(
    "Titik-titik yang mendekati garis merah putus-putus = prediksi mendekati angka sebenarnya. "
    "Semakin jauh dari garis, semakin besar kesalahan prediksi."
)
col_sc, col_err = st.columns(2)

with col_sc:
    fig_sc = px.scatter(
        df_res, x="Aktual (%)", y="Prediksi (%)",
        color="Tahun", color_continuous_scale="Blues",
        hover_data=["Provinsi", "Tahun", "Error (poin%)"],
        opacity=0.75,
        labels={
            "Aktual (%)":  "Kemiskinan Aktual (%)",
            "Prediksi (%)": "Prediksi Model (%)",
        },
    )
    lims = [
        min(y_true.min(), y_pred.min()) - 1,
        max(y_true.max(), y_pred.max()) + 1,
    ]
    fig_sc.add_trace(go.Scatter(
        x=lims, y=lims, mode="lines",
        line=dict(color="#EF4444", dash="dash", width=1.5),
        name="Prediksi Sempurna",
    ))
    fig_sc.update_layout(
        height=400, plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#E2E8F0"), xaxis=dict(gridcolor="#E2E8F0"),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

with col_err:
    st.markdown("**Distribusi Selisih Prediksi**")
    st.caption("Idealnya sebagian besar batang berkumpul di sekitar angka 0 (tidak ada selisih).")
    fig_err = px.histogram(
        df_res, x="Error (poin%)", nbins=25,
        color_discrete_sequence=["#2563EB"],
        labels={"Error (poin%)": "Selisih Prediksi (poin%)"},
    )
    fig_err.add_vline(
        x=0, line_dash="dash", line_color="#EF4444", line_width=1.5,
        annotation_text="Tidak ada selisih", annotation_position="top left",
    )
    fig_err.update_layout(
        height=400, plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#E2E8F0", title="Jumlah Data"),
        xaxis=dict(gridcolor="#E2E8F0"),
    )
    st.plotly_chart(fig_err, use_container_width=True)

st.divider()

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown("### 🔑 Faktor Paling Berpengaruh terhadap Kemiskinan")

if hasattr(model, "feature_importances_"):
    imp_df = pd.DataFrame({
        "Fitur":      FEATURES,
        "Label":      [FEATURE_LABELS.get(f, f) for f in FEATURES],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    top_factor    = imp_df.iloc[-1]["Label"]
    second_factor = imp_df.iloc[-2]["Label"]

    fig_imp = px.bar(
        imp_df, x="Importance", y="Label",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#BFDBFE", "#1D4ED8"],
        text="Importance",
        labels={"Importance": "Tingkat Pengaruh", "Label": "Faktor"},
    )
    fig_imp.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig_imp.update_layout(
        coloraxis_showscale=False, height=380, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#E2E8F0", tickformat=".0%"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption(
        f"💡 Faktor paling menentukan prediksi kemiskinan adalah **{top_factor}**, "
        f"diikuti **{second_factor}**. Semakin panjang batangnya, semakin besar pengaruhnya."
    )
else:
    st.info("Informasi faktor pengaruh tidak tersedia untuk model ini.")

st.divider()

# ── Akurasi Per Tahun (Detail Teknis) ─────────────────────────────────────────
with st.expander("📅 Detail Akurasi per Tahun"):
    records = []
    for tahun, g in df_res.groupby("Tahun"):
        records.append({
            "Tahun":                    int(tahun),
            "Jumlah Provinsi":          len(g),
            "Rata-rata Kesalahan (MAE)": round(mean_absolute_error(g["Aktual (%)"], g["Prediksi (%)"]), 4),
            "Akurasi (R²)":             round(r2_score(g["Aktual (%)"], g["Prediksi (%)"]), 4),
        })
    per_year_df = pd.DataFrame(records)

    col_tbl, col_bar = st.columns([1, 2])
    with col_tbl:
        st.dataframe(per_year_df, use_container_width=True, hide_index=True)
    with col_bar:
        fig_r2 = px.bar(
            per_year_df, x="Tahun", y="Akurasi (R²)",
            text="Akurasi (R²)",
            color="Akurasi (R²)",
            color_continuous_scale=["#BFDBFE", "#1D4ED8"],
        )
        fig_r2.add_hline(
            y=per_year_df["Akurasi (R²)"].mean(), line_dash="dash", line_color="#EF4444",
            annotation_text=f"Rata-rata R²={per_year_df['Akurasi (R²)'].mean():.3f}",
            annotation_position="top right",
        )
        fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_r2.update_layout(
            coloraxis_showscale=False, height=300, showlegend=False,
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#E2E8F0", range=[0, 1.15]),
        )
        st.plotly_chart(fig_r2, use_container_width=True)

st.divider()

# ── Tabel Hasil Prediksi ──────────────────────────────────────────────────────
st.markdown("### 📄 Detail Prediksi per Provinsi")

col_f1, col_f2 = st.columns(2)
with col_f1:
    tahun_opts = ["Semua"] + sorted(df_res["Tahun"].unique().tolist(), reverse=True)
    tahun_sel  = st.selectbox("Filter Tahun:", tahun_opts, key="reg_tahun")
with col_f2:
    prov_opts = ["Semua"] + sorted(df_res["Provinsi"].unique().tolist())
    prov_sel  = st.selectbox("Filter Provinsi:", prov_opts, key="reg_prov")

df_show = df_res.copy()
if tahun_sel != "Semua":
    df_show = df_show[df_show["Tahun"] == int(tahun_sel)]
if prov_sel != "Semua":
    df_show = df_show[df_show["Provinsi"] == prov_sel]

# Tambah kolom status & rename
df_show = df_show.rename(columns={"Error (poin%)": "Selisih Prediksi"})

def _status_prediksi(err):
    if abs(err) <= 1.0:
        return "✅ Akurat"
    elif abs(err) <= 2.5:
        return "🟡 Cukup Akurat"
    else:
        return "🔴 Perlu Perhatian"

df_show["Status"] = df_show["Selisih Prediksi"].apply(_status_prediksi)

def _highlight_error(val):
    if abs(val) > 2.5:
        return "background-color: #FEE2E2; color: #991B1B"
    if abs(val) > 1.0:
        return "background-color: #FEF9C3; color: #854D0E"
    return ""

styled = df_show.style.map(_highlight_error, subset=["Selisih Prediksi"])
st.dataframe(styled, use_container_width=True, hide_index=True, height=420)
st.caption(
    f"Menampilkan {len(df_show):,} baris · "
    "✅ Selisih ≤1 poin% = Akurat | 🟡 Selisih ≤2.5 poin% = Cukup | 🔴 Selisih >2.5 poin% = Perlu Perhatian"
)
