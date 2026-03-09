import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .data_cleaning import (
    convert_column_type,
    fill_missing,
    remove_duplicates,
    remove_outliers_iqr,
    remove_outliers_zscore,
    standardize_categories,
)


def _try_parse_numeric(series: pd.Series) -> Tuple[int, int]:
    values = pd.to_numeric(series.dropna(), errors="coerce")
    success = int(values.notna().sum())
    total = int(len(series.dropna()))
    return success, total


def _most_common(series: pd.Series, n: int = 5) -> List[Any]:
    return series.value_counts().nlargest(n).index.tolist()


def analyze_dataset(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return a list of suggested cleaning actions for the dataset."""

    suggestions: List[Dict[str, Any]] = []

    if df is None or df.empty:
        return suggestions

    # Missing values
    missing = df.isna().sum()
    for col, cnt in missing.items():
        if cnt > 0:
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                method = "median"
            else:
                method = "mode"
            suggestions.append(
                {
                    "type": "missing",
                    "column": col,
                    "count": int(cnt),
                    "suggestion": f"Fill missing values in '{col}' using {method} (count={cnt}).",
                    "action": "fill_missing",
                    "params": {"column": col, "method": method},
                }
            )

    # Duplicates
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        suggestions.append(
            {
                "type": "duplicates",
                "count": dup_count,
                "suggestion": f"Dataset contains {dup_count} duplicate rows.",
                "action": "remove_duplicates",
                "params": {},
            }
        )

    # Inconsistent categories
    for col in df.select_dtypes(include=["object", "category"]).columns:
        vals = df[col].dropna().astype(str)
        if vals.empty:
            continue
        normalized = vals.str.strip().str.lower()
        unique = normalized.unique().tolist()
        if len(unique) <= 1:
            continue
        # detect when distinct normalized values are fewer than original
        if len(unique) < vals.nunique():
            top = vals.value_counts().nlargest(5).index.tolist()
            suggestions.append(
                {
                    "type": "inconsistent_categories",
                    "column": col,
                    "sample_values": top,
                    "suggestion": (
                        f"Column '{col}' contains inconsistent categories (e.g. {top[:3]})."
                        " Consider standardizing values."
                    ),
                    "action": "standardize_categories",
                    "params": {"column": col},
                }
            )

    # Incorrect data types
    for col in df.columns:
        if df[col].dtype == object:
            success, total = _try_parse_numeric(df[col])
            if total >= 5 and success / total >= 0.8:
                suggestions.append(
                    {
                        "type": "type_conversion",
                        "column": col,
                        "suggestion": (
                            f"Column '{col}' looks numeric ({success}/{total} values parsed)."
                            " Convert to numeric type."
                        ),
                        "action": "convert_type",
                        "params": {"column": col, "target_type": "float"},
                    }
                )

    # Outliers
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        z_scores = np.abs((series - series.mean()) / series.std(ddof=0))
        z_outliers = int((z_scores > 3).sum())
        if z_outliers > 0:
            suggestions.append(
                {
                    "type": "outliers",
                    "column": col,
                    "count": int(z_outliers),
                    "suggestion": (
                        f"Column '{col}' has {z_outliers} outliers (|z|>3)."
                        " Consider removing or capping."
                    ),
                    "action": "remove_outliers_zscore",
                    "params": {"column": col, "threshold": 3.0},
                }
            )

    return suggestions


def apply_suggestion(df: pd.DataFrame, suggestion: Dict[str, Any]) -> pd.DataFrame:
    """Apply a suggested cleaning action to the DataFrame."""

    action = suggestion.get("action")
    params = suggestion.get("params", {})

    if action == "fill_missing":
        column = params.get("column")
        method = params.get("method", "median")
        return fill_missing(df, columns=[column], method=method)

    if action == "remove_duplicates":
        return remove_duplicates(df)

    if action == "standardize_categories":
        column = params.get("column")
        return standardize_categories(df, column)

    if action == "convert_type":
        column = params.get("column")
        target_type = params.get("target_type")
        if column and target_type:
            return convert_column_type(df, column, target_type)
        return df

    if action == "remove_outliers_zscore":
        column = params.get("column")
        threshold = params.get("threshold", 3.0)
        return remove_outliers_zscore(df, column, threshold)

    if action == "remove_outliers_iqr":
        column = params.get("column")
        multiplier = params.get("multiplier", 1.5)
        return remove_outliers_iqr(df, column, multiplier)

    return df
