"""
4_Clustering.py — Peta Segmentasi Provinsi Indonesia
======================================================
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

# ── Penamaan Otomatis Cluster ─────────────────────────────────────────────────
_CLUSTER_NAMES = [
    "⚠️ Prioritas Utama",
    "🟡 Perlu Perhatian",
    "🟢 Relatif Sejahtera",
    "💚 Paling Sejahtera",
    "🟠 Menengah-Atas",
]

_CLUSTER_RECOMMENDATIONS = {
    "⚠️ Prioritas Utama": (
        "Provinsi di kelompok ini memiliki kemiskinan tertinggi. "
        "Dibutuhkan **intervensi segera**: bantuan sosial yang tepat sasaran, "
        "peningkatan akses pendidikan, dan program penciptaan lapangan kerja."
    ),
    "🟡 Perlu Perhatian": (
        "Provinsi di kelompok ini berada di zona peringatan. "
        "Diperlukan **pemantauan rutin** dan program pencegahan agar tidak memasuki "
        "kategori risiko tinggi."
    ),
    "🟢 Relatif Sejahtera": (
        "Kondisi sosial-ekonomi kelompok ini relatif lebih baik. "
        "Fokus pada **peningkatan kualitas** pendidikan dan diversifikasi lapangan kerja "
        "untuk mempertahankan tren positif."
    ),
    "💚 Paling Sejahtera": (
        "Kelompok dengan kondisi terbaik. Dapat dijadikan **model referensi** "
        "untuk pengembangan kebijakan di daerah lain."
    ),
    "🟠 Menengah-Atas": (
        "Provinsi dengan kondisi menengah ke atas. "
        "Perlu menjaga momentum pertumbuhan dan pemerataan."
    ),
}


def assign_cluster_names(df_clust: pd.DataFrame, best_k: int) -> dict:
    """Beri nama deskriptif berdasarkan ranking kemiskinan rata-rata tiap cluster."""
    poverty_col   = "Persentase_Penduduk_Miskin"
    cluster_means = df_clust.groupby("Cluster")[poverty_col].mean()
    ranked        = cluster_means.sort_values(ascending=False)  # tertinggi dulu
    name_map      = {}
    for rank, cluster_id in enumerate(ranked.index):
        label = _CLUSTER_NAMES[rank] if rank < len(_CLUSTER_NAMES) else f"Kelompok {cluster_id}"
        name_map[cluster_id] = label
    return name_map


# ── Konfigurasi ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Segmentasi Provinsi | Dashboard Kemiskinan",
    page_icon="🔵",
    layout="wide",
)

st.title("🔵 Peta Segmentasi Provinsi Indonesia")
st.caption("Provinsi dikelompokkan berdasarkan kondisi kemiskinan, pendidikan, dan pengangguran terkini")

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Memuat data dan model clustering..."):
    df       = load_raw_data()
    pipeline = load_model("clustering")

@st.cache_data(show_spinner=False)
def get_clustering_results(_df, _pipeline):
    return apply_clustering(_df, _pipeline)

with st.spinner("Menerapkan model clustering..."):
    df_clust = get_clustering_results(df, pipeline)

# ── Hitung Silhouette & Nama Cluster ─────────────────────────────────────────
scaler   = pipeline["scaler"]
kmeans   = pipeline["model"]
best_k   = kmeans.n_clusters
X_scaled = scaler.transform(df_clust[CLUSTERING_FEATURES])
sil      = silhouette_score(X_scaled, df_clust["Cluster"])

cluster_name_map = assign_cluster_names(df_clust, best_k)
df_clust["Nama Kelompok"] = df_clust["Cluster"].map(cluster_name_map)

# Interpretasi Silhouette
if sil >= 0.5:
    sil_text  = "Sangat Baik ✅"
    sil_desc  = "Kelompok-kelompok provinsi terpisah dengan sangat jelas."
elif sil >= 0.25:
    sil_text  = "Cukup Baik 🟡"
    sil_desc  = "Kelompok provinsi cukup terpisah, ada sedikit tumpang tindih."
else:
    sil_text  = "Perlu Peningkatan ⚠️"
    sil_desc  = "Batas antar kelompok kurang jelas, interpretasi perlu hati-hati."

# ── Narasi Otomatis ───────────────────────────────────────────────────────────
st.info(
    f"📌 **Kesimpulan:** {len(df_clust)} provinsi dikelompokkan menjadi **{best_k} segmen** "
    f"berdasarkan kondisi terkini. Kualitas pengelompokan: **{sil_text}** — {sil_desc} "
    f"Cari provinsi Anda di daftar bawah halaman ini."
)

st.divider()

# ── Ringkasan Model ───────────────────────────────────────────────────────────
st.markdown("### 📊 Ringkasan Pengelompokan")
col1, col2, col3 = st.columns(3)
col1.metric("Jumlah Kelompok (Cluster)", best_k)
col2.metric("Kualitas Pengelompokan",    sil_text)
col3.metric("Jumlah Provinsi",           len(df_clust))

with st.expander("🔬 Lihat Detail Teknis Model"):
    st.markdown(f"""
    | Parameter Teknis | Nilai |
    |-----------------|-------|
    | Algoritma | K-Means Clustering |
    | Jumlah Cluster (K) | {best_k} (dipilih berdasarkan Silhouette Score tertinggi dari K=2–6) |
    | Silhouette Score | `{sil:.4f}` (mendekati 1.0 = pemisahan cluster sangat baik) |
    | Normalisasi Data | StandardScaler (agar skala fitur setara) |
    | Data yang digunakan | Kondisi tahun terakhir tiap provinsi (paling mutakhir) |
    """)

st.divider()

# ── Karakteristik Per Cluster ─────────────────────────────────────────────────
st.markdown("### 📊 Karakteristik Tiap Kelompok")

cluster_means  = df_clust.groupby("Cluster")[CLUSTERING_FEATURES].mean().round(2)
cluster_counts = df_clust["Cluster"].value_counts().sort_index()
cluster_means.insert(0, "Jumlah Provinsi", cluster_counts)
cluster_means.index = [cluster_name_map.get(i, f"Cluster {i}") for i in cluster_means.index]

st.dataframe(
    cluster_means.rename(columns={
        "Persentase_Penduduk_Miskin":   "Rata-rata Kemiskinan (%)",
        "Rata_Rata_Lama_Sekolah":       "Rata-rata Lama Sekolah (Thn)",
        "Tingkat_Pengangguran_Terbuka": "Tingkat Pengangguran (%)",
    }),
    use_container_width=True,
)

# Rekomendasi kebijakan per cluster
st.markdown("#### 💡 Rekomendasi Kebijakan per Kelompok")
rec_cols = st.columns(min(best_k, 3))
for idx, (cluster_id, name) in enumerate(cluster_name_map.items()):
    col_idx = idx % len(rec_cols)
    with rec_cols[col_idx]:
        rec = _CLUSTER_RECOMMENDATIONS.get(name, "Lihat karakteristik kelompok untuk rekomendasi lebih lanjut.")
        st.info(f"**{name}**\n\n{rec}")

st.divider()

# ── Peta Segmentasi (Scatter Plot) ────────────────────────────────────────────
PALETTE = px.colors.qualitative.Set2
st.markdown("### 🗺️ Peta Segmentasi Provinsi")

col_sc1, col_sc2 = st.columns(2)

with col_sc1:
    st.markdown("**Kemiskinan vs Rata-rata Lama Sekolah**")
    fig1 = px.scatter(
        df_clust,
        x="Persentase_Penduduk_Miskin",
        y="Rata_Rata_Lama_Sekolah",
        color="Nama Kelompok",
        text="Provinsi",
        color_discrete_sequence=PALETTE,
        hover_data={
            "Tingkat_Pengangguran_Terbuka": ":.2f",
            "Tahun":         True,
            "Nama Kelompok": False,
        },
        labels={
            "Persentase_Penduduk_Miskin": "Persentase Kemiskinan (%)",
            "Rata_Rata_Lama_Sekolah":     "Rata-rata Lama Sekolah (Tahun)",
            "Nama Kelompok":              "Kelompok",
        },
        height=440,
    )
    fig1.add_vline(
        x=15, line_dash="dash", line_color="#EF4444",
        annotation_text="Batas Risiko (15%)", annotation_position="top right",
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
        legend=dict(title="Kelompok"),
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_sc2:
    st.markdown("**Kemiskinan vs Tingkat Pengangguran**")
    fig2 = px.scatter(
        df_clust,
        x="Persentase_Penduduk_Miskin",
        y="Tingkat_Pengangguran_Terbuka",
        color="Nama Kelompok",
        text="Provinsi",
        color_discrete_sequence=PALETTE,
        hover_data={
            "Rata_Rata_Lama_Sekolah": ":.2f",
            "Tahun":         True,
            "Nama Kelompok": False,
        },
        labels={
            "Persentase_Penduduk_Miskin":   "Persentase Kemiskinan (%)",
            "Tingkat_Pengangguran_Terbuka": "Tingkat Pengangguran (%)",
            "Nama Kelompok":                "Kelompok",
        },
        height=440,
    )
    fig2.add_vline(
        x=15, line_dash="dash", line_color="#EF4444",
        annotation_text="Batas Risiko (15%)", annotation_position="top right",
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
        legend=dict(title="Kelompok"),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Radar Chart ───────────────────────────────────────────────────────────────
st.markdown("### 🕸️ Profil Tiap Kelompok")
st.caption(
    "Radar chart menunjukkan profil relatif tiap kelompok. "
    "Kemiskinan tinggi = buruk, Lama Sekolah tinggi = baik, Pengangguran tinggi = buruk."
)

radar_raw = cluster_means[CLUSTERING_FEATURES].copy()
radar_raw.columns = ["Kemiskinan (%)", "Lama Sekolah (Thn)", "Pengangguran (%)"]

if len(radar_raw) > 1:
    scaler_r   = MinMaxScaler()
    radar_norm = pd.DataFrame(
        scaler_r.fit_transform(radar_raw),
        columns=radar_raw.columns,
        index=radar_raw.index,
    )
else:
    radar_norm = radar_raw.copy()

categories = list(radar_norm.columns)
fig_radar  = go.Figure()

for i, (cname, row) in enumerate(radar_norm.iterrows()):
    vals = list(row.values) + [row.values[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals, theta=categories + [categories[0]],
        fill="toself", name=cname,
        line_color=PALETTE[i % len(PALETTE)],
        fillcolor=PALETTE[i % len(PALETTE)],
        opacity=0.35,
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True, height=420,
    legend=dict(orientation="h", yanchor="bottom", y=-0.30),
)
st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ── Daftar Provinsi per Cluster ───────────────────────────────────────────────
st.markdown("### 📋 Daftar Provinsi per Kelompok")
cols = st.columns(best_k)

for idx, (cluster_id, name) in enumerate(cluster_name_map.items()):
    prov_list = sorted(df_clust[df_clust["Cluster"] == cluster_id]["Provinsi"].tolist())
    col_idx   = idx % best_k
    with cols[col_idx]:
        st.markdown(
            f"<b style='color:{PALETTE[idx % len(PALETTE)]}'>{name}</b> "
            f"<span style='font-size:0.85rem;color:#64748B'>({len(prov_list)} provinsi)</span>",
            unsafe_allow_html=True,
        )
        for p in prov_list:
            st.markdown(f"• {p}")
