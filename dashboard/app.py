import sys
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# path setup
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from dashboard.utils.data_loader         import load_raw_data, load_model
from dashboard.utils.feature_engineering import predict_regression

# config
st.set_page_config(
    page_title="Prediksi Kemiskinan Indonesia",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Prediksi Tingkat Kemiskinan Indonesia")
st.caption(
    "Model: Random Forest Regressor · "
    "Fitur: RLS, TPT, Tahun · "
    "Split: 80/20 random · "
    "Data: BPS 2015–2025"
)

# load data
df     = load_raw_data()
model  = load_model()
df_res = predict_regression(df, model)

# evaluasi model
st.markdown("## 📈 Evaluasi Model")

y_true = df_res["Aktual (%)"].values
y_pred = df_res["Prediksi (%)"].values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("R²",   f"{r2:.4f}",
            help="Koefisien determinasi — semakin mendekati 1 semakin baik")
col2.metric("MAE",  f"{mae:.4f} poin%",
            help="Rata-rata kesalahan absolut prediksi")
col3.metric("RMSE", f"{rmse:.4f} poin%",
            help="Root Mean Square Error — sensitif terhadap kesalahan besar")

# scatter + feature importance
col_sc, col_imp = st.columns(2)

with col_sc:
    st.markdown("### Aktual vs Prediksi")
    fig_sc = px.scatter(
        df_res,
        x="Aktual (%)", y="Prediksi (%)",
        hover_data=["Provinsi", "Tahun", "Error (poin%)"],
        opacity=0.7,
        color_discrete_sequence=["#2563EB"],
    )
    min_val = min(y_true.min(), y_pred.min()) - 1
    max_val = max(y_true.max(), y_pred.max()) + 1
    fig_sc.add_shape(
        type="line",
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color="#EF4444", dash="dash"),
    )
    fig_sc.add_annotation(
        x=max_val, y=max_val,
        text="Prediksi Sempurna",
        showarrow=False,
        font=dict(color="#EF4444", size=11),
        xanchor="right",
    )
    fig_sc.update_layout(
        height=360,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#E2E8F0"),
        xaxis=dict(gridcolor="#E2E8F0"),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

with col_imp:
    st.markdown("### Kontribusi Variabel")
    labels    = ["Rata-Rata Lama Sekolah", "Tingkat Pengangguran", "Tahun"]
    values    = model.feature_importances_
    top_label = labels[int(np.argmax(values))]

    fig_imp = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=["#2563EB", "#7C3AED", "#059669"],
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))
    fig_imp.update_layout(
        xaxis=dict(title="Importance", tickformat=".0%", gridcolor="#E2E8F0"),
        yaxis=dict(autorange="reversed"),
        height=360,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=60, t=10, b=20),
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption(f"💡 Variabel paling berpengaruh: **{top_label}**")

# tabel prediksi
with st.expander("📄 Lihat Tabel Detail Prediksi"):
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        prov_opts = ["Semua"] + sorted(df_res["Provinsi"].unique().tolist())
        prov_sel  = st.selectbox("Filter Provinsi", prov_opts)
    with col_f2:
        tahun_opts = ["Semua"] + sorted(df_res["Tahun"].unique().tolist(), reverse=True)
        tahun_sel  = st.selectbox("Filter Tahun", tahun_opts)

    df_show = df_res.copy()
    if prov_sel  != "Semua":
        df_show = df_show[df_show["Provinsi"] == prov_sel]
    if tahun_sel != "Semua":
        df_show = df_show[df_show["Tahun"] == int(tahun_sel)]

    def _highlight(val):
        if abs(val) > 2.5: return "background-color:#FEE2E2;color:#991B1B"
        if abs(val) > 1.0: return "background-color:#FEF9C3;color:#854D0E"
        return ""

    st.dataframe(
        df_show.style.map(_highlight, subset=["Error (poin%)"]),
        use_container_width=True, hide_index=True, height=360,
    )
    st.caption(
        f"{len(df_show):,} baris · "
        "🔴 Error >2.5 poin% | 🟡 Error >1.0 poin%"
    )

st.divider()

# prediksi interaktif
st.markdown("## 🔮 Prediksi Interaktif")
st.caption("Masukkan nilai RLS dan TPT untuk memperoleh estimasi tingkat kemiskinan.")

col1, col2, col3 = st.columns(3)
with col1:
    rls = st.number_input(
        "Rata-Rata Lama Sekolah (tahun)",
        min_value=0.0, max_value=20.0,
        value=8.5, step=0.1,
    )
with col2:
    tpt = st.number_input(
        "Tingkat Pengangguran Terbuka (%)",
        min_value=0.0, max_value=20.0,
        value=5.0, step=0.1,
    )
with col3:
    tahun = st.number_input(
        "Tahun",
        min_value=2015, max_value=2035,
        value=2025, step=1,
    )

if st.button("🔍 Prediksi", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([{
        "Rata_Rata_Lama_Sekolah"      : rls,
        "Tingkat_Pengangguran_Terbuka": tpt,
        "Tahun"                       : tahun,
    }])
    hasil = model.predict(input_df)[0]

    ref      = df[df["Tahun"] == tahun]["Persentase_Penduduk_Miskin"]
    ref_mean = ref.mean() if len(ref) > 0 else None

    st.success(f"### Prediksi Kemiskinan: **{hasil:.2f}%**")

    if ref_mean is not None:
        selisih = hasil - ref_mean
        arah    = "di atas" if selisih > 0 else "di bawah"
        st.caption(
            f"Rata-rata nasional tahun {tahun}: **{ref_mean:.2f}%** — "
            f"prediksi ini **{abs(selisih):.2f} poin% {arah}** rata-rata."
        )