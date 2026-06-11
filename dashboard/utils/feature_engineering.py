import pandas as pd

from config.constants import FEATURES, TARGET

def predict_regression(df: pd.DataFrame, model) -> pd.DataFrame:

    # jalankan model ke seluruh data dan kembalikan tabel perbandingan.
    df_feat = df[["Provinsi"] + FEATURES + [TARGET]].dropna().reset_index(drop=True)

    y_pred = model.predict(df_feat[FEATURES])
    y_true = df_feat[TARGET].values

    result = df_feat[["Tahun", "Provinsi"]].copy()
    result["Aktual (%)"]    = y_true.round(2)
    result["Prediksi (%)"]  = y_pred.round(2)
    result["Error (poin%)"] = (y_true - y_pred).round(2)

    return result.reset_index(drop=True)