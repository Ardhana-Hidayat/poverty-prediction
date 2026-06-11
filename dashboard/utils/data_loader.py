import os
import sys
import pickle
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

# path setup
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.constants import (
    MODEL_PATH,
    DATABASE_TABLE,
)

# database engine
def _build_engine():
    try:
        db = st.secrets["database"]

        uri = (
            f"postgresql://{db['user']}:{db['password']}"
            f"@{db['host']}:{db['port']}/{db['db_name']}"
            f"?sslmode={db.get('sslmode', 'require')}"
        )

        return create_engine(uri)

    except Exception:
        pass

    try:
        from config.database import get_database_uri
        return create_engine(get_database_uri())

    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_engine():
    return _build_engine()

# load data
@st.cache_data(ttl=3600, show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    engine = get_engine()

    if engine is None:
        st.stop()

    try:
        df = pd.read_sql(
            f"SELECT * FROM {DATABASE_TABLE}",
            engine
        )

    except Exception as e:
        st.stop()

    df["Provinsi"] = (
        df["Provinsi"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    return df

# load model
@st.cache_resource(show_spinner=False)
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model tidak ditemukan: {MODEL_PATH}")
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model