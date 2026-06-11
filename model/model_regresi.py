import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# path setup
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from config.constants import (
    CSV_PATH, MODEL_PATH, DATABASE_TABLE,
    FEATURES, TARGET,
)

# load data
def load_data() -> pd.DataFrame:
    
    df = None

    try:
        from config.database import get_engine
        engine = get_engine()
        df = pd.read_sql(f"SELECT * FROM {DATABASE_TABLE}", engine)
        print(f"[OK] Data dari Aiven ({len(df)} baris)")
    except Exception as e:
        print(f"[WARN] Aiven gagal: {e}")

    return df

# prepare data
df = load_data()

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training : {len(X_train)} baris | Testing : {len(X_test)} baris")

# training
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=3,
    random_state=42,
)
model.fit(X_train, y_train)

# evaluasi
y_pred = model.predict(X_test)

print("\nHasil Evaluasi:")
print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"  MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"  R²   : {r2_score(y_test, y_pred):.4f}")

print("\nFeature Importance:")
for feat, imp in zip(FEATURES, model.feature_importances_):
    print(f"  {feat:<35} {imp:.4f}")

# simpan model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\n[OK] Model tersimpan → {MODEL_PATH}")