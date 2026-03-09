import io

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_dataset_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load a dataset from raw bytes.

    Supports CSV and Excel (.xlsx) formats. Cached for faster reloads.
    """
    if file_bytes is None or filename is None:
        return pd.DataFrame()

    lower = filename.lower()
    buffer = io.BytesIO(file_bytes)

    try:
        if lower.endswith(".csv"):
            return pd.read_csv(buffer)
        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            return pd.read_excel(buffer)

        # fallback: try CSV first
        try:
            buffer.seek(0)
            return pd.read_csv(buffer)
        except Exception:
            buffer.seek(0)
            return pd.read_excel(buffer)
    except Exception as e:
        raise ValueError(f"Unable to load dataset: {e}")
