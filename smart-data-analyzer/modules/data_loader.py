import pandas as pd


def load_dataset(uploaded_file):
    """Load a dataset from an uploaded file handle.

    Supports CSV and Excel (.xlsx) formats.
    Returns a pandas DataFrame.
    """

    if uploaded_file is None:
        return None

    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            return pd.read_excel(uploaded_file)

        # fallback: try CSV first
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        raise ValueError(f"Unable to load dataset: {e}")
