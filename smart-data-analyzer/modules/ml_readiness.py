import re
from typing import Any, Dict, Optional

import pandas as pd


SENSITIVE_NAME_PATTERN = re.compile(
    r"(^|_|\s)(email|e-mail|phone|mobile|address|cnic|ssn|passport|"
    r"national.?id|full.?name|ip.?address|credit.?card)($|_|\s)",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_PATTERN = re.compile(r"^\+?[\d\s().-]{8,20}$")


def detect_sensitive_columns(df: pd.DataFrame) -> Dict[str, str]:
    findings: Dict[str, str] = {}
    for column in df.columns:
        name = str(column)
        if SENSITIVE_NAME_PATTERN.search(name):
            findings[name] = "sensitive column name"
            continue
        values = df[column].dropna().astype(str).head(200)
        if values.empty:
            continue
        email_matches = values.str.match(EMAIL_PATTERN).mean()
        phone_matches = values.str.match(PHONE_PATTERN).mean()
        if email_matches >= 0.2:
            findings[name] = "email-like values"
        elif phone_matches >= 0.5:
            findings[name] = "phone/identifier-like values"
    return findings


def detect_leakage_risks(df: pd.DataFrame, target: Optional[str]) -> list:
    if not target or target not in df.columns:
        return []
    risks = []
    target_series = df[target]
    for column in df.columns:
        if column == target:
            continue
        feature = df[column]
        comparable = pd.concat([feature, target_series], axis=1).dropna()
        if comparable.empty:
            continue
        if comparable.iloc[:, 0].astype(str).equals(
            comparable.iloc[:, 1].astype(str)
        ):
            risks.append(f"'{column}' duplicates the target.")
            continue
        if (
            pd.api.types.is_numeric_dtype(feature)
            and pd.api.types.is_numeric_dtype(target_series)
            and len(comparable) >= 10
        ):
            correlation = comparable.iloc[:, 0].corr(comparable.iloc[:, 1])
            if pd.notna(correlation) and abs(float(correlation)) >= 0.995:
                risks.append(
                    f"'{column}' has near-perfect correlation with the target "
                    f"({float(correlation):.3f})."
                )
    return risks


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

    sensitive_columns = detect_sensitive_columns(df)
    if sensitive_columns:
        score -= min(15, len(sensitive_columns) * 3)
        issues.append(
            "Potentially sensitive columns: "
            + ", ".join(list(sensitive_columns)[:8])
        )
        recommendations.append(
            "Remove, mask, or document sensitive fields before sharing."
        )

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

    leakage_risks = detect_leakage_risks(df, target)
    if leakage_risks:
        score -= min(25, len(leakage_risks) * 8)
        issues.extend(["Possible leakage: " + risk for risk in leakage_risks])
        recommendations.append(
            "Remove leakage candidates and re-evaluate the model on held-out data."
        )

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
        "sensitive_columns": sensitive_columns,
        "leakage_risks": leakage_risks,
        "target": target_summary,
        "issues": issues,
        "recommendations": recommendations,
    }
