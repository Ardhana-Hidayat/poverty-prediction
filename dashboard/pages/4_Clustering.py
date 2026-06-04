"""
4_Clustering.py — Model Clustering
=====================================
Segmentasi provinsi berdasarkan kondisi sosial-ekonomi terkini menggunakan K-Means.
"""

import os, sys
_DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from utils.data_loader         import load_raw_data, load_model
from utils.feature_engineering import apply_clustering, CLUSTERING_FEATURES

# ── Konfigurasi ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clustering | Dashboard Kemiskinan",
    page_icon="🔵",
    layout="wide",
)

st.title("🔵 Model Clustering")
st.caption("Segmentasi provinsi berdasarkan kondisi sosial-ekonomi terkini menggunakan K-Means")

st.info(
    "**Model:** K-Means Clustering · "
    "**Data:** Kondisi tahun terakhir tiap provinsi (paling mutakhir) · "
    "**K Optimal:** Dipilih berdasarkan Silhouette Score tertinggi (K=2–6)"
)

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Memuat data dan model clustering..."):
    df       = load_raw_data()
    pipeline = load_model("clustering")

@st.cache_data(show_spinner=False)
def get_clustering_results(_df, _pipeline):
    return apply_clustering(_df, _pipeline)

with st.spinner("Menerapkan model clustering..."):
    df_clust = get_clustering_results(df, pipeline)

# Silhouette score
scaler   = pipeline["scaler"]
kmeans   = pipeline["model"]
best_k   = kmeans.n_clusters
X_scaled = scaler.transform(df_clust[CLUSTERING_FEATURES])
sil      = silhouette_score(X_scaled, df_clust["Cluster"])

# ── Metrik Ringkasan ──────────────────────────────────────────────────────────
st.markdown("### 📊 Ringkasan Model")
col1, col2, col3 = st.columns(3)
col1.metric("Jumlah Cluster (K)", best_k)
col2.metric("Silhouette Score",   f"{sil:.4f}",
            help="Mendekati 1.0 = cluster terpisah dengan baik")
col3.metric("Jumlah Provinsi",    len(df_clust))

st.divider()

# ── Warna Cluster ─────────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Set2
df_clust["Cluster Label"] = df_clust["Cluster"].apply(lambda c: f"Cluster {c}")

# ── Scatter Plot ──────────────────────────────────────────────────────────────
st.markdown("### 🗺️ Peta Segmentasi Provinsi")
col_sc1, col_sc2 = st.columns(2)

with col_sc1:
    st.markdown("**Kemiskinan vs Rata-rata Lama Sekolah**")
    fig1 = px.scatter(
        df_clust,
        x="Persentase_Penduduk_Miskin",
        y="Rata_Rata_Lama_Sekolah",
        color="Cluster Label",
        text="Provinsi",
        color_discrete_sequence=PALETTE,
        hover_data={
            "Tingkat_Pengangguran_Terbuka": ":.2f",
            "Tahun": True,
            "Cluster Label": False,
        },
        labels={
            "Persentase_Penduduk_Miskin" : "Persentase Kemiskinan (%)",
            "Rata_Rata_Lama_Sekolah"     : "Rata-rata Lama Sekolah (Tahun)",
        },
        height=440,
    )
    fig1.add_vline(
        x=15, line_dash="dash", line_color="#EF4444",
        annotation_text="Batas 15%", annotation_position="top right",
    )
    fig1.update_traces(
        textposition="top center",
        textfont=dict(size=8, color="#374151"),
        marker=dict(size=11, line=dict(width=1, color="white")),
    )
    fig1.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#E2E8F0"),
        xaxis=dict(gridcolor="#E2E8F0"),
        legend=dict(title="Segmen"),
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_sc2:
    st.markdown("**Kemiskinan vs Tingkat Pengangguran**")
    fig2 = px.scatter(
        df_clust,
        x="Persentase_Penduduk_Miskin",
        y="Tingkat_Pengangguran_Terbuka",
        color="Cluster Label",
        text="Provinsi",
        color_discrete_sequence=PALETTE,
        hover_data={
            "Rata_Rata_Lama_Sekolah": ":.2f",
            "Tahun": True,
            "Cluster Label": False,
        },
        labels={
            "Persentase_Penduduk_Miskin"      : "Persentase Kemiskinan (%)",
            "Tingkat_Pengangguran_Terbuka"    : "Tingkat Pengangguran (%)",
        },
        height=440,
    )
    fig2.add_vline(
        x=15, line_dash="dash", line_color="#EF4444",
        annotation_text="Batas 15%", annotation_position="top right",
    )
    fig2.update_traces(
        textposition="top center",
        textfont=dict(size=8, color="#374151"),
        marker=dict(size=11, line=dict(width=1, color="white")),
    )
    fig2.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#E2E8F0"),
        xaxis=dict(gridcolor="#E2E8F0"),
        legend=dict(title="Segmen"),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Karakteristik Cluster ─────────────────────────────────────────────────────
st.markdown("### 📊 Karakteristik Per Cluster")

cluster_means  = df_clust.groupby("Cluster")[CLUSTERING_FEATURES].mean().round(2)
cluster_counts = df_clust["Cluster"].value_counts().sort_index()
cluster_means.insert(0, "Jumlah Provinsi", cluster_counts)
cluster_means.index = [f"Cluster {i}" for i in cluster_means.index]

st.dataframe(
    cluster_means.rename(columns={
        "Persentase_Penduduk_Miskin"  : "% Kemiskinan",
        "Rata_Rata_Lama_Sekolah"      : "Rata-rata Lama Sekolah (Thn)",
        "Tingkat_Pengangguran_Terbuka": "Tingkat Pengangguran (%)",
    }),
    use_container_width=True,
)

st.divider()

# ── Radar Chart ───────────────────────────────────────────────────────────────
st.markdown("### 🕸️ Perbandingan Cluster (Radar)")

radar_raw = cluster_means[CLUSTERING_FEATURES].copy()
radar_raw.columns = ["% Kemiskinan", "Rata-rata Lama Sekolah (Thn)", "Tingkat Pengangguran (%)"]
# Normalisasi 0–1 untuk radar
scaler_r = MinMaxScaler()
radar_norm = pd.DataFrame(
    scaler_r.fit_transform(radar_raw),
    columns=radar_raw.columns,
    index=radar_raw.index,
)

categories = list(radar_norm.columns)
fig_radar  = go.Figure()

for i, (cname, row) in enumerate(radar_norm.iterrows()):
    vals = list(row.values) + [row.values[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals, theta=categories + [categories[0]],
        fill="toself", name=cname,
        line_color=PALETTE[i],
        fillcolor=PALETTE[i],
        opacity=0.35,
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True, height=400,
    legend=dict(orientation="h", yanchor="bottom", y=-0.25),
)
st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ── Daftar Provinsi per Cluster ───────────────────────────────────────────────
st.markdown("### 📋 Daftar Provinsi per Cluster")
cols = st.columns(best_k)

for c in range(best_k):
    with cols[c]:
        prov_list = sorted(df_clust[df_clust["Cluster"] == c]["Provinsi"].tolist())
        st.markdown(
            f"<b style='color:{PALETTE[c]}'>Cluster {c}</b> — {len(prov_list)} Provinsi",
            unsafe_allow_html=True,
        )
        for p in prov_list:
            st.markdown(f"• {p}")
