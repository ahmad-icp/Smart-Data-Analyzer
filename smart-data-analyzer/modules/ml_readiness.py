from typing import Any, Dict, Optional

import pandas as pd


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = max(len(df), 1)
    for column in df.columns:
        series = df[column]
        examples = [str(value)[:80] for value in series.dropna().unique()[:3]]
        rows.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "missing_percent": round(float(series.isna().sum() / total * 100), 2),
                "unique": int(series.nunique(dropna=True)),
                "examples": " | ".join(examples),
            }
        )
    return pd.DataFrame(rows)


def analyze_ml_readiness(
    df: pd.DataFrame, target: Optional[str] = None
) -> Dict[str, Any]:
    """Return deterministic, explainable ML-readiness checks."""
    issues = []
    recommendations = []
    score = 100

    if df.empty:
        return {"score": 0, "issues": ["Dataset is empty."], "recommendations": []}

    missing_rate = float(df.isna().sum().sum() / max(df.size, 1))
    if missing_rate > 0:
        score -= min(25, round(missing_rate * 100))
        issues.append(f"{missing_rate:.1%} of all cells are missing.")
        recommendations.append("Impute or remove missing values using a documented rule.")

    duplicate_rate = float(df.duplicated().mean())
    if duplicate_rate > 0:
        score -= min(15, round(duplicate_rate * 100))
        issues.append(f"{duplicate_rate:.1%} of rows are duplicates.")
        recommendations.append("Review and remove unintended duplicate observations.")

    identifier_columns = [
        str(col)
        for col in df.columns
        if len(df) > 20 and df[col].nunique(dropna=True) / len(df) > 0.98
    ]
    if identifier_columns:
        score -= min(10, len(identifier_columns) * 2)
        issues.append("Potential identifiers: " + ", ".join(identifier_columns[:8]))
        recommendations.append("Exclude identifiers unless they carry predictive meaning.")

    high_cardinality = []
    for col in df.columns:
        dtype = df[col].dtype
        is_text = pd.api.types.is_string_dtype(dtype)
        is_category = isinstance(dtype, pd.CategoricalDtype)
        if (is_text or is_category) and df[col].nunique(dropna=True) > min(
            100, max(20, len(df) * 0.5)
        ):
            high_cardinality.append(str(col))
    if high_cardinality:
        score -= min(10, len(high_cardinality) * 2)
        issues.append("High-cardinality text columns: " + ", ".join(high_cardinality[:8]))
        recommendations.append("Encode, group, or vectorize high-cardinality features.")

    target_summary = None
    if target and target in df.columns:
        counts = df[target].value_counts(normalize=True, dropna=True)
        target_summary = {
            "column": target,
            "classes": int(df[target].nunique(dropna=True)),
            "majority_share": round(float(counts.iloc[0]), 4) if not counts.empty else None,
        }
        if len(counts) > 1 and float(counts.iloc[0]) > 0.8:
            score -= 15
            issues.append(f"Target '{target}' is strongly imbalanced.")
            recommendations.append("Use stratified splitting and imbalance-aware metrics.")
        if df[target].isna().any():
            score -= 10
            issues.append(f"Target '{target}' contains missing values.")
            recommendations.append("Resolve rows with missing target values.")

    if len(df) < 100:
        score -= 10
        issues.append("The dataset has fewer than 100 rows.")
        recommendations.append("Collect more observations or use careful cross-validation.")

    if not issues:
        recommendations.append("Dataset passes the current automated readiness checks.")

    return {
        "score": max(0, int(score)),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_rate": round(missing_rate, 4),
        "duplicate_rate": round(duplicate_rate, 4),
        "potential_identifiers": identifier_columns,
        "target": target_summary,
        "issues": issues,
        "recommendations": recommendations,
    }
