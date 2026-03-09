import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


def remove_missing(df: pd.DataFrame, axis: int = 0, how: str = "any") -> pd.DataFrame:
    """Remove rows (axis=0) or columns (axis=1) with missing values."""
    return df.dropna(axis=axis, how=how).copy()


def fill_missing(
    df: pd.DataFrame,
    columns=None,
    method: str = "mean",
    custom_value=None,
) -> pd.DataFrame:
    """Fill missing values in columns using a method (mean/median/mode/custom)."""
    result = df.copy()

    if columns is None or len(columns) == 0:
        columns = result.columns.tolist()

    for col in columns:
        if col not in result.columns:
            continue

        if method == "mean":
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].mean())
        elif method == "median":
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].median())
        elif method == "mode":
            mode_val = result[col].mode()
            if not mode_val.empty:
                result[col] = result[col].fillna(mode_val.iloc[0])
        elif method == "custom":
            result[col] = result[col].fillna(custom_value)
        else:
            raise ValueError(f"Unknown fill method: {method}")

    return result


def remove_duplicates(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame."""
    return df.drop_duplicates(subset=subset).copy()


def drop_columns(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Drop selected columns from the dataset."""
    if not columns:
        return df
    return df.drop(columns=columns, errors="ignore").copy()


def rename_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Rename columns using a mapping dict."""
    return df.rename(columns=rename_map).copy()


def convert_column_type(df: pd.DataFrame, column: str, target_type: str) -> pd.DataFrame:
    """Convert a column to a different dtype."""
    result = df.copy()
    if column not in result.columns:
        return result

    if target_type == "int":
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    elif target_type == "float":
        result[column] = pd.to_numeric(result[column], errors="coerce")
    elif target_type == "str":
        result[column] = result[column].astype(str)
    elif target_type == "datetime":
        result[column] = pd.to_datetime(result[column], errors="coerce")
    elif target_type == "category":
        result[column] = result[column].astype("category")
    else:
        raise ValueError(f"Unsupported target type: {target_type}")

    return result


def filter_rows(df: pd.DataFrame, column: str, op: str, value) -> pd.DataFrame:
    """Filter rows using a simple comparison operator."""
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    series = df[column]

    if op == "contains":
        return df[series.astype(str).str.contains(str(value), na=False)]

    # Try numeric conversion when possible
    try:
        comp_value = float(value)
    except Exception:
        comp_value = value

    if op == "==":
        return df[series == comp_value]
    if op == "!=":
        return df[series != comp_value]
    if op == ">":
        return df[series.astype(float) > float(comp_value)]
    if op == "<":
        return df[series.astype(float) < float(comp_value)]
    if op == ">=":
        return df[series.astype(float) >= float(comp_value)]
    if op == "<=":
        return df[series.astype(float) <= float(comp_value)]

    raise ValueError(f"Unsupported operator: {op}")


def sort_dataframe(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
    """Sort a DataFrame by a column."""
    if column not in df.columns:
        return df
    return df.sort_values(by=column, ascending=ascending).copy()


def standardize_categories(df: pd.DataFrame, column: str, mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Normalize category values in a column using a mapping or casing rules."""
    result = df.copy()
    if column not in result.columns:
        return result

    series = result[column].astype(str)

    if mapping:
        result[column] = series.replace(mapping)
    else:
        # Default standardization: strip + lower + title
        result[column] = series.str.strip().str.lower().str.title()

    return result


def replace_values(
    df: pd.DataFrame,
    replacements: Dict[Any, Any],
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Replace values in the dataframe using a mapping dictionary."""
    result = df.copy()
    if columns:
        for col in columns:
            if col in result.columns:
                result[col] = result[col].replace(replacements)
    else:
        result = result.replace(replacements)
    return result


def log_transform(df: pd.DataFrame, column: str, new_column: Optional[str] = None) -> pd.DataFrame:
    """Apply log transformation to a numeric column."""
    result = df.copy()
    if column not in result.columns:
        return result

    new_column = new_column or f"{column}_log"
    result[new_column] = np.log1p(result[column].astype(float).replace({np.nan: None}))
    return result


def normalize_column(df: pd.DataFrame, column: str, method: str = "minmax") -> pd.DataFrame:
    """Normalize a numeric column using min-max or z-score scaling."""
    result = df.copy()
    if column not in result.columns:
        return result

    series = pd.to_numeric(result[column], errors="coerce")
    if method == "minmax":
        min_val = series.min()
        max_val = series.max()
        result[column] = (series - min_val) / (max_val - min_val)
    elif method == "zscore":
        result[column] = (series - series.mean()) / series.std(ddof=0)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return result


def standardize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Standardize a numeric column to zero mean and unit variance."""
    return normalize_column(df, column, method="zscore")


def clean_string_column(
    df: pd.DataFrame,
    column: str,
    trim: bool = True,
    remove_special: bool = True,
    case: Optional[str] = None,
) -> pd.DataFrame:
    """Clean string column values with trimming, case normalization, and special character removal."""
    result = df.copy()
    if column not in result.columns:
        return result

    series = result[column].astype(str)
    if trim:
        series = series.str.strip()
    if remove_special:
        series = series.str.replace(r"[^\w\s]", "", regex=True)
    if case == "lower":
        series = series.str.lower()
    elif case == "upper":
        series = series.str.upper()
    elif case == "title":
        series = series.str.title()

    result[column] = series
    return result


def parse_dates(df: pd.DataFrame, column: str, format: Optional[str] = None) -> pd.DataFrame:
    """Parse a column to datetime."""
    result = df.copy()
    if column not in result.columns:
        return result

    result[column] = pd.to_datetime(result[column], format=format, errors="coerce")
    return result


def extract_date_parts(
    df: pd.DataFrame,
    column: str,
    parts: Optional[List[str]] = None,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Extract year/month/day parts from a datetime column."""
    result = df.copy()
    if column not in result.columns:
        return result

    if parts is None:
        parts = ["year", "month", "day"]

    col_dt = pd.to_datetime(result[column], errors="coerce")
    prefix = prefix or column
    for part in parts:
        if part == "year":
            result[f"{prefix}_year"] = col_dt.dt.year
        elif part == "month":
            result[f"{prefix}_month"] = col_dt.dt.month
        elif part == "day":
            result[f"{prefix}_day"] = col_dt.dt.day
        elif part == "weekday":
            result[f"{prefix}_weekday"] = col_dt.dt.weekday

    return result


def remove_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """Remove rows that are outliers based on z-score."""
    if column not in df.columns:
        return df

    series = pd.to_numeric(df[column], errors="coerce")
    z = np.abs((series - series.mean()) / series.std(ddof=0))
    return df.loc[z <= threshold].copy()


def remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """Remove rows that are outliers based on the IQR method."""
    if column not in df.columns:
        return df

    series = pd.to_numeric(df[column], errors="coerce")
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return df[(series >= lower) & (series <= upper)].copy()
