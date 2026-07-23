import io
import json

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_dataset_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load a dataset from raw bytes.

    Supports CSV/TSV, Excel, JSON and Parquet formats.
    """
    if file_bytes is None or filename is None:
        return pd.DataFrame()

    lower = filename.lower()
    buffer = io.BytesIO(file_bytes)

    try:
        if lower.endswith(".csv"):
            return pd.read_csv(buffer, sep=None, engine="python")
        if lower.endswith((".tsv", ".tab")):
            return pd.read_csv(buffer, sep="\t")
        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            return pd.read_excel(buffer)
        if lower.endswith((".json", ".jsonl", ".ndjson")):
            try:
                return pd.read_json(buffer, lines=lower.endswith((".jsonl", ".ndjson")))
            except ValueError:
                buffer.seek(0)
                payload = json.load(io.TextIOWrapper(buffer, encoding="utf-8-sig"))
                return pd.json_normalize(payload)
        if lower.endswith((".parquet", ".pq")):
            return pd.read_parquet(buffer)

        # fallback: try CSV first
        try:
            buffer.seek(0)
            return pd.read_csv(buffer)
        except Exception:
            buffer.seek(0)
            return pd.read_excel(buffer)
    except Exception as e:
        raise ValueError(f"Unable to load dataset: {e}")
