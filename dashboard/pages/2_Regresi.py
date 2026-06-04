"""
2_Regresi.py — Model Regresi
==============================
Hasil prediksi persentase kemiskinan menggunakan Random Forest.
"""

import os, sys
_DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.data_loader        import load_raw_data, load_model
from utils.feature_engineering import predict_regression, FEATURES

# ── Konfigurasi ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regresi | Dashboard Kemiskinan",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Model Regresi")
st.caption("Prediksi persentase kemiskinan menggunakan Random Forest Regressor")

st.info(
    "**Model:** Random Forest Regressor (n=200, max_depth=6) · "
    "**Target:** log1p(kemiskinan) → dibalik dengan expm1 ke skala asli · "
    "**Fitur:** lag, YoY, interaksi RLS×TPT, tren waktu"
)

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Memuat data dan model regresi..."):
    df    = load_raw_data()
    model = load_model("regresi")

@st.cache_data(show_spinner=False)
def get_regression_results(_df, _model):
    return predict_regression(_df, _model)

with st.spinner("Menghitung prediksi..."):
    df_res = get_regression_results(df, model)

# ── Metrik Keseluruhan ────────────────────────────────────────────────────────
y_true = df_res["Aktual (%)"].values
y_pred = df_res["Prediksi (%)"].values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

st.markdown("### 📊 Metrik Evaluasi")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.4f} poin%",
            help="Root Mean Square Error — semakin kecil semakin baik")
col2.metric("MAE",  f"{mae:.4f} poin%",
            help="Mean Absolute Error — rata-rata kesalahan absolut prediksi")
col3.metric("R²",   f"{r2:.4f}",
            help="Koefisien determinasi — 1.0 = sempurna")

st.divider()

# ── Metrik Per Tahun ──────────────────────────────────────────────────────────
st.markdown("### 📅 Metrik Per Tahun")

per_year = (
    df_res.groupby("Tahun")
    .apply(lambda g: {
        "N"   : len(g),
        "RMSE": round(np.sqrt(mean_squared_error(g["Aktual (%)"], g["Prediksi (%)"])), 4),
        "MAE" : round(mean_absolute_error(g["Aktual (%)"], g["Prediksi (%)"]), 4),
        "R²"  : round(r2_score(g["Aktual (%)"], g["Prediksi (%)"]), 4),
    })
    .apply(lambda x: x)
)
import pandas as pd
per_year_df = pd.DataFrame(per_year.tolist(), index=per_year.index).reset_index()

col_tbl, col_bar = st.columns([1, 2])
with col_tbl:
    st.dataframe(per_year_df, use_container_width=True, hide_index=True)

with col_bar:
    fig_r2 = px.bar(
        per_year_df, x="Tahun", y="R²",
        text="R²",
        color="R²",
        color_continuous_scale=["#BFDBFE", "#1D4ED8"],
        labels={"Tahun": "Tahun", "R²": "R² Score"},
    )
    fig_r2.add_hline(
        y=per_year_df["R²"].mean(), line_dash="dash", line_color="#EF4444",
        annotation_text=f"Rata-rata R²={per_year_df['R²'].mean():.3f}",
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

# ── Scatter Aktual vs Prediksi & Residual ─────────────────────────────────────
st.markdown("### 🎯 Aktual vs Prediksi")
col_sc, col_res = st.columns(2)

with col_sc:
    fig_sc = px.scatter(
        df_res, x="Aktual (%)", y="Prediksi (%)",
        color="Tahun", color_continuous_scale="Blues",
        hover_data=["Provinsi", "Tahun", "Error (poin%)"],
        opacity=0.75,
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

with col_res:
    fig_res = px.scatter(
        df_res, x="Prediksi (%)", y="Error (poin%)",
        color="Tahun", color_continuous_scale="Blues",
        hover_data=["Provinsi"],
        opacity=0.75,
        labels={"Error (poin%)": "Residual (Aktual − Prediksi)"},
    )
    fig_res.add_hline(y=0, line_dash="dash", line_color="#EF4444", line_width=1.5)
    fig_res.update_layout(
        height=400, plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#E2E8F0"), xaxis=dict(gridcolor="#E2E8F0"),
    )
    st.plotly_chart(fig_res, use_container_width=True)

st.divider()

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown("### 🔑 Feature Importance (Random Forest)")

if hasattr(model, "feature_importances_"):
    import pandas as pd
    imp_df = pd.DataFrame({
        "Fitur"      : FEATURES,
        "Importance" : model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig_imp = px.bar(
        imp_df, x="Importance", y="Fitur",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#BFDBFE", "#1D4ED8"],
        text="Importance",
    )
    fig_imp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_imp.update_layout(
        coloraxis_showscale=False, height=380, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#E2E8F0"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("Feature importance tidak tersedia untuk model ini.")

st.divider()

# ── Tabel Hasil Prediksi ──────────────────────────────────────────────────────
st.markdown("### 📄 Tabel Hasil Prediksi")

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

# Highlight error besar
def _highlight_error(val):
    if abs(val) > 3:
        return "background-color: #FEE2E2; color: #991B1B"
    if abs(val) > 1.5:
        return "background-color: #FEF9C3; color: #854D0E"
    return ""

styled = df_show.style.applymap(_highlight_error, subset=["Error (poin%)"])
st.dataframe(styled, use_container_width=True, hide_index=True, height=420)
st.caption(
    f"Menampilkan {len(df_show):,} baris · "
    "🔴 Error > 3 poin% | 🟡 Error > 1.5 poin%"
)
