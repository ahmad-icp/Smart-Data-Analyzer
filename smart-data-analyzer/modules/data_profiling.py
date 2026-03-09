import pandas as pd
import numpy as np
import plotly.express as px


def dataset_overview(df: pd.DataFrame) -> dict:
    """Generate a quick overview of the dataset."""
    if df is None:
        return {}

    overview = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": len(df.select_dtypes(include=["number"]).columns),
        "categorical_columns": len(df.select_dtypes(include=["object", "category"]).columns),
    }
    return overview


def column_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for each column."""
    if df is None or df.empty:
        return pd.DataFrame()

    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return pd.DataFrame()

    stats = numeric.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    stats["variance"] = numeric.var()
    stats["missing"] = df.isna().sum()
    stats["unique"] = df.nunique()
    return stats


def missing_heatmap(df: pd.DataFrame, max_rows: int = 1000):
    """Create a heatmap figure showing missing value patterns."""
    if df is None or df.empty:
        return None

    sample = df
    if len(df) > max_rows:
        sample = df.sample(max_rows, random_state=1)

    matrix = sample.isna().astype(int)
    try:
        fig = px.imshow(
            matrix.T,
            aspect="auto",
            color_continuous_scale="Blues",
            labels={"x": "Row", "y": "Column", "color": "Missing"},
            title="Missing Value Heatmap",
        )
        return fig
    except Exception:
        return None


def correlation_heatmap(df: pd.DataFrame):
    """Return a correlation heatmap figure for numeric columns."""
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return None

    corr = numeric.corr()
    fig = px.imshow(
        corr,
        title="Correlation Heatmap",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    return fig


def numeric_distribution_plots(df: pd.DataFrame, max_cols: int = 6) -> dict:
    """Generate distribution histograms for numeric columns."""
    numeric = df.select_dtypes(include=["number"])
    plots = {}
    if numeric.empty:
        return plots

    for col in numeric.columns[:max_cols]:
        try:
            fig = px.histogram(df, x=col, nbins=40, title=f"Distribution of {col}")
            plots[col] = fig
        except Exception:
            continue
    return plots
