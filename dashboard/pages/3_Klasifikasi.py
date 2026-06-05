"""
3_Klasifikasi.py — Mana Daerah yang Perlu Diwaspadai?
=======================================================
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
from utils.feature_engineering import predict_classification, FEATURES

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
    page_title="Deteksi Risiko | Dashboard Kemiskinan",
    page_icon="🔴",
    layout="wide",
)

st.title("🔴 Mana Daerah yang Perlu Diwaspadai?")
st.caption("Model mendeteksi provinsi berisiko kemiskinan tinggi (di atas 15%)")

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Memuat data dan model..."):
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

# ── Hitung Metrik ─────────────────────────────────────────────────────────────
acc    = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred, zero_division=0)
prec   = precision_score(y_true, y_pred, zero_division=0)
f1     = f1_score(y_true, y_pred, zero_division=0)
auc    = roc_auc_score(y_true, y_prob)
cm     = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

false_alarm_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
total_positive   = int(y_true.sum())
detected         = int(tp)

# ── Narasi Otomatis ───────────────────────────────────────────────────────────
quality = "sangat baik ✅" if acc >= 0.90 else "baik ✅" if acc >= 0.80 else "cukup 🟡"

st.success(
    f"📌 **Kesimpulan:** Dari **{total_positive} kasus** provinsi dengan kemiskinan tinggi, "
    f"model berhasil mendeteksi **{detected} kasus ({recall*100:.0f}%)**. "
    f"Akurasi keseluruhan model adalah **{acc*100:.1f}%** — performa {quality}. "
    f"Hanya **{int(fn)} kasus** risiko tinggi yang tidak berhasil terdeteksi."
)

st.divider()

# ── 3 Metrik Kunci ────────────────────────────────────────────────────────────
st.markdown("### 📊 Performa Model Secara Keseluruhan")
col1, col2, col3 = st.columns(3)

col1.metric(
    "Ketepatan Deteksi Risiko",
    f"{recall*100:.1f}%",
    help=f"Dari {total_positive} provinsi yang benar-benar berisiko tinggi, "
         f"model berhasil mendeteksi {detected} ({recall*100:.0f}%). "
         "Semakin tinggi semakin baik.",
)
col2.metric(
    "Akurasi Keseluruhan",
    f"{acc*100:.1f}%",
    help="Persentase prediksi yang benar (baik Aman maupun Risiko Tinggi) dari seluruh data.",
)
col3.metric(
    "Tingkat Alarm Palsu",
    f"{false_alarm_rate:.1f}%",
    help=f"Dari provinsi yang sebenarnya aman, sekitar {false_alarm_rate:.0f}% "
         "salah dikategorikan sebagai Risiko Tinggi. Semakin kecil semakin baik.",
)

with st.expander("🔬 Lihat Detail Teknis Evaluasi Model"):
    st.markdown(f"""
    | Metrik Teknis | Nilai | Penjelasan |
    |---------------|-------|-----------|
    | Accuracy      | `{acc:.4f}`  | Ketepatan prediksi keseluruhan |
    | Precision     | `{prec:.4f}` | Dari yang diprediksi risiko, seberapa banyak yang benar |
    | Recall        | `{recall:.4f}` | Dari yang benar risiko, seberapa banyak yang terdeteksi |
    | F1-Score      | `{f1:.4f}`  | Rata-rata harmonis Precision & Recall |
    | ROC AUC       | `{auc:.4f}` | Kemampuan memisahkan dua kelas (0.5 = acak, 1.0 = sempurna) |

    **Algoritma:** Logistic Regression dengan `class_weight='balanced'`
    **Penanganan ketidakseimbangan data:** Pembobotan kelas (~18% data positif)
    **Metode validasi:** Walk-forward (train tahun < t, test tahun t) mulai tahun 2020
    """)

st.divider()

# ── Ringkasan Prediksi ────────────────────────────────────────────────────────
st.markdown("### 🔲 Ringkasan Hasil Prediksi")
col_cm, col_summary = st.columns(2)

with col_cm:
    labels_cm = ["Aman", "Risiko Tinggi"]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels_cm, y=labels_cm,
        colorscale=[[0, "#EFF6FF"], [1, "#1D4ED8"]],
        showscale=False,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 24, "color": "white"},
    ))
    fig_cm.update_layout(
        xaxis_title="Prediksi Model",
        yaxis_title="Kondisi Aktual",
        yaxis=dict(autorange="reversed"),
        height=320,
        plot_bgcolor="white", paper_bgcolor="white",
        title="Matriks Prediksi",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_summary:
    st.markdown("**Apa arti angka-angka ini?**")
    st.markdown(f"""
    | Hasil | Jumlah | Keterangan |
    |-------|--------|-----------|
    | ✅ Benar Aman | **{tn}** | Provinsi aman dan diprediksi aman |
    | ✅ Benar Terdeteksi Risiko | **{tp}** | Provinsi risiko dan berhasil terdeteksi |
    | ⚠️ Alarm Palsu | **{fp}** | Provinsi aman, tapi dikira risiko |
    | ❌ Terlewat | **{fn}** | Provinsi risiko, tapi tidak terdeteksi |
    """)
    st.caption(
        f"Model berhasil mendeteksi **{tp} dari {tp+fn}** kasus risiko tinggi. "
        f"Hanya **{fn} kasus** yang terlewat tidak terdeteksi."
    )

with st.expander("📐 Lihat Kurva ROC (Detail Teknis)"):
    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_prob)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr_vals, y=tpr_vals, mode="lines",
        name=f"Model ini (AUC = {auc:.3f})",
        line=dict(color="#2563EB", width=2.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Tebakan Acak (AUC = 0.500)",
        line=dict(color="#EF4444", dash="dash", width=1.5),
    ))
    fig_roc.update_layout(
        xaxis_title="Tingkat Alarm Palsu (False Positive Rate)",
        yaxis_title="Ketepatan Deteksi (True Positive Rate)",
        height=360,
        legend=dict(x=0.45, y=0.05),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#E2E8F0"),
        xaxis=dict(gridcolor="#E2E8F0"),
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.caption(
        f"Kurva ROC menunjukkan kemampuan model membedakan daerah berisiko dan aman. "
        f"AUC = {auc:.3f} berarti model "
        f"{'jauh lebih baik' if auc > 0.80 else 'lebih baik'} dari tebakan acak (AUC = 0.5)."
    )

st.divider()

# ── Faktor Penentu Risiko ─────────────────────────────────────────────────────
st.markdown("### 🔑 Faktor Penentu Risiko Kemiskinan")
try:
    coefs = model.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({
        "Fitur":     FEATURES,
        "Label":     [FEATURE_LABELS.get(f, f) for f in FEATURES],
        "Koefisien": coefs,
    }).sort_values("Koefisien", ascending=True)

    top_pos = (
        coef_df[coef_df["Koefisien"] > 0].iloc[-1]["Label"]
        if (coef_df["Koefisien"] > 0).any() else "-"
    )
    top_neg = (
        coef_df[coef_df["Koefisien"] < 0].iloc[0]["Label"]
        if (coef_df["Koefisien"] < 0).any() else "-"
    )

    colors = ["#1D4ED8" if c >= 0 else "#EF4444" for c in coef_df["Koefisien"]]
    fig_coef = go.Figure(go.Bar(
        x=coef_df["Koefisien"], y=coef_df["Label"],
        orientation="h",
        marker_color=colors,
        text=coef_df["Koefisien"].round(2),
        textposition="outside",
    ))
    fig_coef.add_vline(x=0, line_width=1, line_color="#94A3B8")
    fig_coef.update_layout(
        height=380, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#E2E8F0"),
        xaxis_title="Arah Pengaruh terhadap Risiko Kemiskinan",
    )
    st.plotly_chart(fig_coef, use_container_width=True)
    st.caption(
        f"🔵 **Meningkatkan risiko kemiskinan:** {top_pos} — semakin tinggi nilainya, "
        f"semakin besar kemungkinan daerah tersebut masuk kategori risiko tinggi.  \n"
        f"🔴 **Menurunkan risiko kemiskinan:** {top_neg} — semakin tinggi nilainya, "
        f"semakin kecil kemungkinan risiko."
    )
except AttributeError:
    st.info("Informasi faktor penentu tidak tersedia untuk model ini.")

st.divider()

# ── Tabel Prediksi ────────────────────────────────────────────────────────────
st.markdown("### 📄 Status Risiko per Provinsi")

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    tahun_opts = ["Semua"] + sorted(df_res["Tahun"].unique().tolist(), reverse=True)
    tahun_sel  = st.selectbox("Filter Tahun:", tahun_opts, key="kls_tahun")
with col_f2:
    status_opts = ["Semua", "Risiko Tinggi", "Aman"]
    status_sel  = st.selectbox("Filter Status Prediksi:", status_opts, key="kls_status")
with col_f3:
    salah_only = st.checkbox("Hanya tampilkan prediksi yang meleset", value=False)

df_show = df_res[["Tahun", "Provinsi", "Aktual Label", "Prediksi Label", "Prob. Risiko (%)"]].copy()
df_show = df_show.rename(columns={
    "Aktual Label":  "Kondisi Aktual",
    "Prediksi Label": "Prediksi Model",
})

if tahun_sel != "Semua":
    df_show = df_show[df_show["Tahun"] == int(tahun_sel)]
if status_sel != "Semua":
    df_show = df_show[df_show["Prediksi Model"] == status_sel]
if salah_only:
    df_show = df_show[df_show["Kondisi Aktual"] != df_show["Prediksi Model"]]

def _style_risk(row):
    styles = [""] * len(row)
    cols   = list(df_show.columns)
    if row.get("Prediksi Model") == "Risiko Tinggi":
        styles[cols.index("Prediksi Model")] = (
            "background-color:#FEE2E2; color:#991B1B; font-weight:bold"
        )
    if row.get("Kondisi Aktual") != row.get("Prediksi Model"):
        styles[cols.index("Kondisi Aktual")] = (
            "background-color:#FEF9C3; color:#92400E"
        )
    if row.get("Prob. Risiko (%)") >= 70:
        styles[cols.index("Prob. Risiko (%)")] = (
            "color:#991B1B; font-weight:bold"
        )
    return styles

styled = df_show.style.apply(_style_risk, axis=1)
st.dataframe(styled, use_container_width=True, hide_index=True, height=400)
st.caption(
    f"Menampilkan {len(df_show):,} baris · "
    "🔴 Risiko Tinggi (>15%) | 🟡 Prediksi Meleset | Prob. ≥70% = Risiko Sangat Tinggi"
)

st.divider()

# ── Distribusi Probabilitas ───────────────────────────────────────────────────
st.markdown("### 📊 Sebaran Skor Risiko Kemiskinan")
st.caption(
    "Histogram ini menunjukkan seberapa sering model memberikan skor risiko tinggi atau rendah. "
    "Provinsi dengan skor di atas 50% dikategorikan sebagai Risiko Tinggi."
)

year_filter_opts = ["Semua"] + sorted(df_res["Tahun"].unique().tolist(), reverse=True)
year_dist = st.selectbox("Pilih Tahun:", year_filter_opts, key="kls_dist")

df_dist = df_res if year_dist == "Semua" else df_res[df_res["Tahun"] == int(year_dist)]
fig_dist = px.histogram(
    df_dist, x="Prob. Risiko (%)", nbins=20,
    color_discrete_sequence=["#2563EB"],
    labels={"Prob. Risiko (%)": "Skor Risiko Kemiskinan (%)"},
)
fig_dist.add_vline(
    x=50, line_dash="dash", line_color="#EF4444",
    annotation_text="Batas Risiko (50%)", annotation_position="top right",
)
fig_dist.update_layout(
    height=300, plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(gridcolor="#E2E8F0", title="Jumlah Provinsi"),
    xaxis=dict(gridcolor="#E2E8F0"),
)
st.plotly_chart(fig_dist, use_container_width=True)
