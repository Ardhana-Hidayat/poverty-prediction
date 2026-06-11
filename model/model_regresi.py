import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# konfigurasi data
DATA_PATH = "data/dataset_cleaned.csv"
MODEL_PATH = "model/model_regresi_rf.pkl"

FEATURES = [
    "Rata_Rata_Lama_Sekolah",
    "Tingkat_Pengangguran_Terbuka",
    "Tahun",
]

TARGET = "Persentase_Penduduk_Miskin"

PAPUA_DOB = {
    "PAPUA BARAT DAYA",
    "PAPUA PEGUNUNGAN",
    "PAPUA SELATAN",
    "PAPUA TENGAH"
}

# load & clean
df = pd.read_csv(DATA_PATH)

df = df[
    ~df["Provinsi"].isin(PAPUA_DOB)
    & ~(
        (df["Provinsi"] == "PAPUA")
        & (df["Tahun"] >= 2024)
    )
].reset_index(drop=True)

# split data 80/20
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# train data dengan Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=3,
    random_state=42
)

model.fit(X_train, y_train)

print(f"Jumlah data training : {len(X_train)}")
print(f"Jumlah data testing  : {len(X_test)}")

# evaluasi
y_pred = model.predict(X_test)

metrics = {
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "MAE": mean_absolute_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred)
}

print("\nHasil Evaluasi:")
for k, v in metrics.items():
    print(f"{k:<5}: {v:.4f}")

# importance feature
importance = (
    pd.Series(model.feature_importances_, index=FEATURES)
    .sort_values(ascending=False)
)

print("\nFeature Importance:")
print(importance)

# save model
os.makedirs("model", exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

# simpan hasil prediksi
hasil = X_test.copy()
hasil["Provinsi"] = df.loc[X_test.index, "Provinsi"]
hasil["Aktual"] = y_test
hasil["Prediksi"] = y_pred
hasil["Error"] = hasil["Aktual"] - hasil["Prediksi"]

hasil.to_csv(
    "model/regresi_predictions.csv",
    index=False
)

# visualisasi
fig, ax = plt.subplots(figsize=(6, 5))

ax.scatter(y_test, y_pred, alpha=0.6)

lims = [
    min(y_test.min(), y_pred.min()),
    max(y_test.max(), y_pred.max())
]

ax.plot(lims, lims, "r--")
ax.set_xlabel("Aktual (%)")
ax.set_ylabel("Prediksi (%)")
ax.set_title("Aktual vs Prediksi")

plt.tight_layout()
plt.savefig("model/evaluasi_regresi.png", dpi=150)
plt.show()