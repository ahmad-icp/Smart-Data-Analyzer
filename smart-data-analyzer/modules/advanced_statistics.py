from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats


def normality_test(series: pd.Series) -> Dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 3:
        raise ValueError("At least three numeric observations are required.")
    sample = values.sample(5000, random_state=42) if len(values) > 5000 else values
    statistic, p_value = stats.shapiro(sample)
    return {
        "test": "Shapiro-Wilk",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "sample_size": int(len(sample)),
        "interpretation": (
            "Evidence against normality"
            if p_value < 0.05
            else "Insufficient evidence to reject normality"
        ),
    }


def mann_whitney_test(
    df: pd.DataFrame, value_column: str, group_column: str
) -> Dict[str, Any]:
    groups = df[[value_column, group_column]].dropna()
    labels = groups[group_column].unique()
    if len(labels) != 2:
        raise ValueError("Mann-Whitney U requires exactly two groups.")
    first = pd.to_numeric(
        groups.loc[groups[group_column] == labels[0], value_column],
        errors="coerce",
    ).dropna()
    second = pd.to_numeric(
        groups.loc[groups[group_column] == labels[1], value_column],
        errors="coerce",
    ).dropna()
    if min(len(first), len(second)) < 2:
        raise ValueError("Each group requires at least two numeric observations.")
    statistic, p_value = stats.mannwhitneyu(first, second, alternative="two-sided")
    return {
        "test": "Mann-Whitney U",
        "groups": [str(labels[0]), str(labels[1])],
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size_rank_biserial": float(
            1 - (2 * statistic) / (len(first) * len(second))
        ),
    }


def one_way_anova(
    df: pd.DataFrame, value_column: str, group_column: str
) -> Dict[str, Any]:
    clean = df[[value_column, group_column]].dropna()
    groups = [
        pd.to_numeric(group[value_column], errors="coerce").dropna()
        for _, group in clean.groupby(group_column, observed=True)
    ]
    groups = [group for group in groups if len(group) >= 2]
    if len(groups) < 2:
        raise ValueError("ANOVA requires at least two groups with two values each.")
    statistic, p_value = stats.f_oneway(*groups)
    all_values = pd.concat(groups)
    grand_mean = all_values.mean()
    between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)
    total = sum(((group - grand_mean) ** 2).sum() for group in groups)
    eta_squared = float(between / total) if total else 0.0
    return {
        "test": "One-way ANOVA",
        "groups": len(groups),
        "f_statistic": float(statistic),
        "p_value": float(p_value),
        "eta_squared": eta_squared,
    }


def chi_square_test(
    df: pd.DataFrame, first_column: str, second_column: str
) -> Dict[str, Any]:
    table = pd.crosstab(df[first_column], df[second_column])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Chi-square requires at least a 2x2 contingency table.")
    statistic, p_value, degrees, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = float(np.sqrt((statistic / n) / denominator)) if denominator else 0.0
    return {
        "test": "Chi-square independence",
        "chi_square": float(statistic),
        "p_value": float(p_value),
        "degrees_of_freedom": int(degrees),
        "cramers_v": cramers_v,
        "minimum_expected_count": float(np.min(expected)),
    }
