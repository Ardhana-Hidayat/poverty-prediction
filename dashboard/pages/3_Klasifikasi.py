"""
3_Klasifikasi.py — Model Klasifikasi
======================================
Prediksi kategori risiko kemiskinan (> 15%) menggunakan Logistic Regression.
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

from utils.data_loader         import load_raw_data, load_model
from utils.feature_engineering import predict_classification

# ── Konfigurasi ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi | Dashboard Kemiskinan",
    page_icon="🔴",
    layout="wide",
)

st.title("🔴 Model Klasifikasi")
st.caption("Prediksi kategori risiko kemiskinan (Risiko Tinggi jika > 15%) menggunakan Logistic Regression")

st.info(
    "**Model:** Logistic Regression dengan `class_weight='balanced'` · "
    "**Target:** Biner — Risiko Tinggi (kemiskinan > 15%) atau Aman · "
    "**Imbalance:** ~18% data positif, ditangani dengan pembobotan kelas"
)

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Memuat data dan model klasifikasi..."):
    df    = load_raw_data()
    model = load_model("klasifikasi")

@st.cache_data(show_spinner=False)
def get_classification_results(_df, _model):
    return predict_classification(_df, _model)

with st.spinner("Menghitung prediksi..."):
    df_res = get_classification_results(df, model)

y_true = df_res["Aktual"].values
y_pred = df_res["Prediksi"].values
y_prob = df_res["Prob. Risiko (%)"].values / 100

# ── Metrik Keseluruhan ────────────────────────────────────────────────────────
st.markdown("### 📊 Metrik Evaluasi")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy",  f"{accuracy_score(y_true, y_pred):.4f}")
col2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.4f}")
col3.metric("Recall",    f"{recall_score(y_true, y_pred, zero_division=0):.4f}")
col4.metric("F1-Score",  f"{f1_score(y_true, y_pred, zero_division=0):.4f}")
col5.metric("ROC AUC",   f"{roc_auc_score(y_true, y_prob):.4f}")

st.divider()

# ── Confusion Matrix & ROC Curve ──────────────────────────────────────────────
col_cm, col_roc = st.columns(2)

with col_cm:
    st.markdown("### 🔲 Confusion Matrix")
    cm     = confusion_matrix(y_true, y_pred)
    labels = ["Aman (0)", "Risiko Tinggi (1)"]

    # Buat heatmap manual dengan plotly
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels, y=labels,
        colorscale=[[0, "#EFF6FF"], [1, "#1D4ED8"]],
        showscale=False,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20, "color": "white"},
    ))
    # Override warna teks untuk nilai kecil (warna gelap)
    fig_cm.update_layout(
        xaxis_title="Prediksi",
        yaxis_title="Aktual",
        yaxis=dict(autorange="reversed"),
        height=360,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Interpretasi
    tn, fp, fn, tp = cm.ravel()
    st.markdown(f"""
    | | Keterangan |
    |-|-----------|
    | **TN = {tn}** | Benar diprediksi Aman |
    | **FP = {fp}** | Salah diprediksi Risiko Tinggi (sebenarnya Aman) |
    | **FN = {fn}** | Salah diprediksi Aman (sebenarnya Risiko Tinggi) |
    | **TP = {tp}** | Benar diprediksi Risiko Tinggi |
    """)

with col_roc:
    st.markdown("### 📐 Kurva ROC")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val     = roc_auc_score(y_true, y_prob)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"Logistic Regression (AUC = {auc_val:.3f})",
        line=dict(color="#2563EB", width=2.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Acak (AUC = 0.500)",
        line=dict(color="#EF4444", dash="dash", width=1.5),
    ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        height=360,
        legend=dict(x=0.45, y=0.05),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#E2E8F0"),
        xaxis=dict(gridcolor="#E2E8F0"),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

st.divider()

# ── Koefisien Model ───────────────────────────────────────────────────────────
st.markdown("### 🔑 Koefisien Logistic Regression")
try:
    coefs = model.named_steps["model"].coef_[0]
    from utils.feature_engineering import FEATURES
    coef_df = pd.DataFrame({
        "Fitur": FEATURES,
        "Koefisien": coefs,
    }).sort_values("Koefisien", ascending=True)

    colors = ["#1D4ED8" if c >= 0 else "#EF4444" for c in coef_df["Koefisien"]]
    fig_coef = go.Figure(go.Bar(
        x=coef_df["Koefisien"], y=coef_df["Fitur"],
        orientation="h",
        marker_color=colors,
        text=coef_df["Koefisien"].round(4),
        textposition="outside",
    ))
    fig_coef.add_vline(x=0, line_width=1, line_color="#94A3B8")
    fig_coef.update_layout(
        height=380, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#E2E8F0"),
        xaxis_title="Nilai Koefisien",
    )
    st.plotly_chart(fig_coef, use_container_width=True)
    st.caption("🔵 Positif = meningkatkan probabilitas Risiko Tinggi · 🔴 Negatif = menurunkan probabilitas")
except AttributeError:
    st.info("Koefisien tidak dapat ditampilkan untuk jenis model ini.")

st.divider()

# ── Tabel Prediksi ────────────────────────────────────────────────────────────
st.markdown("### 📄 Tabel Prediksi Risiko per Provinsi")

col_f1, col_f2 = st.columns(2)
with col_f1:
    tahun_opts = ["Semua"] + sorted(df_res["Tahun"].unique().tolist(), reverse=True)
    tahun_sel  = st.selectbox("Filter Tahun:", tahun_opts, key="kls_tahun")
with col_f2:
    status_opts = ["Semua", "Risiko Tinggi", "Aman"]
    status_sel  = st.selectbox("Filter Status Prediksi:", status_opts, key="kls_status")

df_show = df_res[["Tahun", "Provinsi", "Aktual Label", "Prediksi Label", "Prob. Risiko (%)"]].copy()
df_show = df_show.rename(columns={"Aktual Label": "Aktual", "Prediksi Label": "Prediksi"})

if tahun_sel != "Semua":
    df_show = df_show[df_show["Tahun"] == int(tahun_sel)]
if status_sel != "Semua":
    df_show = df_show[df_show["Prediksi"] == status_sel]

def _style_risk(row):
    styles = [""] * len(row)
    cols   = list(df_show.columns)
    if row.get("Prediksi") == "Risiko Tinggi":
        styles[cols.index("Prediksi")] = "background-color:#FEE2E2; color:#991B1B; font-weight:bold"
    if row.get("Prob. Risiko (%)") >= 70:
        styles[cols.index("Prob. Risiko (%)")] = "color:#991B1B; font-weight:bold"
    return styles

styled = df_show.style.apply(_style_risk, axis=1)
st.dataframe(styled, use_container_width=True, hide_index=True, height=420)
st.caption(f"Menampilkan {len(df_show):,} baris · 🔴 Risiko Tinggi (> 15%) | 🟢 Aman")

st.divider()

# ── Distribusi Probabilitas ───────────────────────────────────────────────────
st.markdown("### 📊 Distribusi Probabilitas Risiko Tinggi")
year_filter_opts = ["Semua"] + sorted(df_res["Tahun"].unique().tolist(), reverse=True)
year_dist = st.selectbox("Pilih Tahun:", year_filter_opts, key="kls_dist")

df_dist = df_res if year_dist == "Semua" else df_res[df_res["Tahun"] == int(year_dist)]
fig_dist = px.histogram(
    df_dist, x="Prob. Risiko (%)", nbins=20,
    color_discrete_sequence=["#2563EB"],
    labels={"Prob. Risiko (%)": "Probabilitas Risiko Tinggi (%)"},
)
fig_dist.add_vline(
    x=50, line_dash="dash", line_color="#EF4444",
    annotation_text="Threshold 50%", annotation_position="top right",
)
fig_dist.update_layout(
    height=300, plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(gridcolor="#E2E8F0"),
    xaxis=dict(gridcolor="#E2E8F0"),
)
st.plotly_chart(fig_dist, use_container_width=True)
