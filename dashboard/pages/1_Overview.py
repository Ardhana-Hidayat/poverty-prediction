"""
1_Overview.py — Tren Kemiskinan Indonesia
==========================================
Statistik deskriptif dan tren kemiskinan 30 provinsi Indonesia (2015–2025).
"""

import os, sys
_DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_raw_data

# ── Konfigurasi ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Overview | Dashboard Kemiskinan",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Tren Kemiskinan Indonesia")
st.caption("Statistik dan tren kemiskinan 30 provinsi Indonesia (2015–2025)")

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Memuat data..."):
    df = load_raw_data()

latest_year = int(df["Tahun"].max())
first_year  = int(df["Tahun"].min())
df_latest   = df[df["Tahun"] == latest_year]
df_first    = df[df["Tahun"] == first_year]

avg_latest      = df_latest["Persentase_Penduduk_Miskin"].mean()
avg_first       = df_first["Persentase_Penduduk_Miskin"].mean()
high_risk_count = int((df_latest["Persentase_Penduduk_Miskin"] > 15).sum())
worst_prov      = df_latest.loc[df_latest["Persentase_Penduduk_Miskin"].idxmax(), "Provinsi"]
best_prov       = df_latest.loc[df_latest["Persentase_Penduduk_Miskin"].idxmin(), "Provinsi"]
trend_direction = "turun" if avg_latest < avg_first else "naik"
trend_delta     = abs(avg_latest - avg_first)

# ── Narasi Otomatis ───────────────────────────────────────────────────────────
st.info(
    f"📌 **Ringkasan {latest_year}:** Rata-rata kemiskinan nasional adalah **{avg_latest:.1f}%** — "
    f"{trend_direction} {trend_delta:.1f} poin% dibanding {first_year}. "
    f"Terdapat **{high_risk_count} provinsi** dengan kemiskinan di atas 15%. "
    f"Kemiskinan tertinggi: **{worst_prov}**, terendah: **{best_prov}**."
)

# ── Metrik Ringkasan ──────────────────────────────────────────────────────────
st.markdown("### 📊 Ringkasan Data")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Provinsi Dianalisis",    df["Provinsi"].nunique())
col2.metric("Periode Data",                  f"{first_year} – {latest_year}")
col3.metric(f"Rata-rata Kemiskinan ({latest_year})",
            f"{avg_latest:.2f}%")
col4.metric(f"Provinsi Kemiskinan >15% ({latest_year})",
            high_risk_count)

st.divider()

# ── Tren Nasional ─────────────────────────────────────────────────────────────
st.markdown("### 📈 Tren Kemiskinan Nasional per Tahun")
st.caption(
    "Garis biru = rata-rata nasional · "
    "Area biru = rentang antara provinsi tertinggi dan terendah · "
    "Garis merah putus-putus = batas risiko tinggi (15%)"
)

trend = (
    df.groupby("Tahun")["Persentase_Penduduk_Miskin"]
    .agg(["mean", "min", "max"])
    .reset_index()
)
trend.columns = ["Tahun", "Rata-rata", "Minimum", "Maksimum"]

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=trend["Tahun"], y=trend["Maksimum"],
    fill=None, mode="lines",
    line=dict(color="rgba(37,99,235,0.0)"),
    name="Tertinggi", showlegend=True,
    hovertemplate="Tertinggi: %{y:.2f}%<extra></extra>",
))
fig_trend.add_trace(go.Scatter(
    x=trend["Tahun"], y=trend["Minimum"],
    fill="tonexty", mode="lines",
    fillcolor="rgba(37,99,235,0.10)",
    line=dict(color="rgba(37,99,235,0.0)"),
    name="Terendah", showlegend=True,
    hovertemplate="Terendah: %{y:.2f}%<extra></extra>",
))
fig_trend.add_trace(go.Scatter(
    x=trend["Tahun"], y=trend["Rata-rata"],
    mode="lines+markers",
    line=dict(color="#2563EB", width=2.5),
    marker=dict(size=7, color="#2563EB"),
    name="Rata-rata Nasional",
    hovertemplate="Rata-rata: %{y:.2f}%<extra></extra>",
))
fig_trend.add_hline(
    y=15, line_dash="dash", line_color="#EF4444",
    annotation_text="Batas Risiko Tinggi (15%)",
    annotation_position="bottom right",
)
fig_trend.update_layout(
    xaxis_title="Tahun",
    yaxis_title="Persentase Kemiskinan (%)",
    hovermode="x unified",
    height=380,
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(gridcolor="#E2E8F0", zeroline=False),
    xaxis=dict(gridcolor="#E2E8F0", dtick=1),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_trend, use_container_width=True)

tahun_peak = int(trend.loc[trend["Rata-rata"].idxmax(), "Tahun"])
st.caption(
    f"💡 Secara nasional, kemiskinan cenderung **{trend_direction}** dari {avg_first:.1f}% "
    f"({first_year}) menjadi {avg_latest:.1f}% ({latest_year}). "
    f"Puncak tertinggi terjadi pada tahun {tahun_peak}."
)

st.divider()

# ── Top & Bottom Provinces ────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown(f"### 🔴 10 Provinsi Kemiskinan Tertinggi ({latest_year})")
    top10 = (
        df_latest.nlargest(10, "Persentase_Penduduk_Miskin")
        [["Provinsi", "Persentase_Penduduk_Miskin"]]
    )
    fig_top = px.bar(
        top10, x="Persentase_Penduduk_Miskin", y="Provinsi",
        orientation="h",
        color="Persentase_Penduduk_Miskin",
        color_continuous_scale=["#BFDBFE", "#1D4ED8"],
        text="Persentase_Penduduk_Miskin",
        labels={"Persentase_Penduduk_Miskin": "Kemiskinan (%)"},
    )
    fig_top.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_top.update_layout(
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
        xaxis=dict(gridcolor="#E2E8F0"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=370, showlegend=False,
    )
    st.plotly_chart(fig_top, use_container_width=True)

with col_b:
    st.markdown(f"### 🟢 10 Provinsi Kemiskinan Terendah ({latest_year})")
    bot10 = (
        df_latest.nsmallest(10, "Persentase_Penduduk_Miskin")
        [["Provinsi", "Persentase_Penduduk_Miskin"]]
    )
    fig_bot = px.bar(
        bot10, x="Persentase_Penduduk_Miskin", y="Provinsi",
        orientation="h",
        color="Persentase_Penduduk_Miskin",
        color_continuous_scale=["#1D4ED8", "#BFDBFE"],
        text="Persentase_Penduduk_Miskin",
        labels={"Persentase_Penduduk_Miskin": "Kemiskinan (%)"},
    )
    fig_bot.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bot.update_layout(
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
        xaxis=dict(gridcolor="#E2E8F0"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=370, showlegend=False,
    )
    st.plotly_chart(fig_bot, use_container_width=True)

st.divider()

# ── Tren Per Provinsi ─────────────────────────────────────────────────────────
st.markdown("### 🔍 Bandingkan Antar Provinsi")
all_provinces = sorted(df["Provinsi"].unique())
selected = st.multiselect(
    "Pilih provinsi yang ingin dibandingkan:",
    options=all_provinces,
    default=all_provinces[:5],
)

if selected:
    df_sel = df[df["Provinsi"].isin(selected)]
    fig_prov = px.line(
        df_sel, x="Tahun", y="Persentase_Penduduk_Miskin",
        color="Provinsi", markers=True,
        labels={"Persentase_Penduduk_Miskin": "Kemiskinan (%)"},
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig_prov.add_hline(
        y=15, line_dash="dash", line_color="#EF4444",
        annotation_text="Batas Risiko Tinggi (15%)",
        annotation_position="bottom right",
    )
    fig_prov.update_layout(
        hovermode="x unified", height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#E2E8F0"),
        xaxis=dict(gridcolor="#E2E8F0", dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_prov, use_container_width=True)
else:
    st.info("Pilih minimal satu provinsi untuk menampilkan tren.")

st.divider()

# ── Data Table ────────────────────────────────────────────────────────────────
with st.expander("📄 Lihat Data Lengkap"):
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        tahun_opts = ["Semua"] + sorted(df["Tahun"].unique().tolist(), reverse=True)
        tahun_f    = st.selectbox("Filter Tahun:", tahun_opts, key="tbl_tahun")
    with col_f2:
        prov_opts = ["Semua"] + sorted(df["Provinsi"].unique().tolist())
        prov_f    = st.selectbox("Filter Provinsi:", prov_opts, key="tbl_prov")

    df_show = df.copy()
    if tahun_f != "Semua":
        df_show = df_show[df_show["Tahun"] == int(tahun_f)]
    if prov_f != "Semua":
        df_show = df_show[df_show["Provinsi"] == prov_f]

    # Kolom Status berwarna
    def get_status(val):
        if val > 15:
            return "🔴 Tinggi (>15%)"
        elif val > 10:
            return "🟡 Sedang (10–15%)"
        else:
            return "🟢 Rendah (≤10%)"

    df_show = df_show.copy()
    df_show.insert(
        df_show.columns.get_loc("Persentase_Penduduk_Miskin") + 1,
        "Status",
        df_show["Persentase_Penduduk_Miskin"].apply(get_status),
    )

    st.dataframe(df_show, use_container_width=True, height=350, hide_index=True)
    st.caption(
        f"Menampilkan {len(df_show):,} dari {len(df):,} baris · "
        "🔴 Tinggi >15% | 🟡 Sedang 10–15% | 🟢 Rendah ≤10%"
    )
