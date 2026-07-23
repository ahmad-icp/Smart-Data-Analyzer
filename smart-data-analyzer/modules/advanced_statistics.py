from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats


def _values(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def normality_test(series: pd.Series, method: str = "shapiro") -> Dict[str, Any]:
    """Run Shapiro-Wilk, D'Agostino K² or Anderson-Darling normality checks."""
    values = _values(series)
    minimum = 8 if method == "dagostino" else 3
    if len(values) < minimum:
        raise ValueError(f"At least {minimum} numeric observations are required.")
    sample = values.sample(5000, random_state=42) if len(values) > 5000 else values
    if method == "shapiro":
        statistic, p_value = stats.shapiro(sample)
        name = "Shapiro-Wilk"
        result = {
            "test": name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "sample_size": int(len(sample)),
        }
    elif method == "dagostino":
        statistic, p_value = stats.normaltest(sample)
        name = "D'Agostino K²"
        result = {
            "test": name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "sample_size": int(len(sample)),
        }
    elif method == "anderson":
        test = stats.anderson(sample, dist="norm")
        name = "Anderson-Darling"
        result = {
            "test": name,
            "statistic": float(test.statistic),
            "critical_values": [float(value) for value in test.critical_values],
            "significance_levels": [
                float(value) for value in test.significance_level
            ],
            "sample_size": int(len(sample)),
        }
    else:
        raise ValueError("Method must be shapiro, dagostino or anderson.")
    if "p_value" in result:
        result["interpretation"] = (
            "Evidence against normality"
            if result["p_value"] < 0.05
            else "Insufficient evidence to reject normality"
        )
    return result


def one_sample_t_test(
    series: pd.Series, population_mean: float
) -> Dict[str, Any]:
    values = _values(series)
    if len(values) < 2:
        raise ValueError("At least two observations are required.")
    statistic, p_value = stats.ttest_1samp(values, population_mean)
    effect = (values.mean() - population_mean) / values.std(ddof=1)
    return {
        "test": "One-sample t-test",
        "sample_mean": float(values.mean()),
        "population_mean": float(population_mean),
        "t_statistic": float(statistic),
        "p_value": float(p_value),
        "cohens_d": float(effect) if np.isfinite(effect) else 0.0,
        "n": int(len(values)),
    }


def paired_t_test(
    df: pd.DataFrame, first_column: str, second_column: str
) -> Dict[str, Any]:
    clean = df[[first_column, second_column]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if len(clean) < 2:
        raise ValueError("At least two complete pairs are required.")
    difference = clean[first_column] - clean[second_column]
    statistic, p_value = stats.ttest_rel(
        clean[first_column], clean[second_column]
    )
    effect = difference.mean() / difference.std(ddof=1)
    return {
        "test": "Paired t-test",
        "t_statistic": float(statistic),
        "p_value": float(p_value),
        "mean_difference": float(difference.mean()),
        "cohens_dz": float(effect) if np.isfinite(effect) else 0.0,
        "pairs": int(len(clean)),
    }


def independent_t_test(
    df: pd.DataFrame, value_column: str, group_column: str
) -> Dict[str, Any]:
    first, second, labels = _two_groups(df, value_column, group_column)
    levene_stat, levene_p = stats.levene(first, second, center="median")
    statistic, p_value = stats.ttest_ind(first, second, equal_var=False)
    pooled = np.sqrt(
        ((len(first) - 1) * first.var() + (len(second) - 1) * second.var())
        / (len(first) + len(second) - 2)
    )
    effect = (first.mean() - second.mean()) / pooled if pooled else 0.0
    return {
        "test": "Welch independent t-test",
        "groups": labels,
        "t_statistic": float(statistic),
        "p_value": float(p_value),
        "cohens_d": float(effect),
        "levene_p_value": float(levene_p),
        "equal_variance_supported": bool(levene_p >= 0.05),
        "sample_sizes": [int(len(first)), int(len(second))],
    }


def _two_groups(
    df: pd.DataFrame, value_column: str, group_column: str
) -> tuple[pd.Series, pd.Series, List[str]]:
    groups = df[[value_column, group_column]].dropna()
    labels = groups[group_column].unique()
    if len(labels) != 2:
        raise ValueError("Exactly two groups are required.")
    first = _values(groups.loc[groups[group_column] == labels[0], value_column])
    second = _values(groups.loc[groups[group_column] == labels[1], value_column])
    if min(len(first), len(second)) < 2:
        raise ValueError("Each group requires at least two numeric observations.")
    return first, second, [str(labels[0]), str(labels[1])]


def mann_whitney_test(
    df: pd.DataFrame, value_column: str, group_column: str
) -> Dict[str, Any]:
    first, second, labels = _two_groups(df, value_column, group_column)
    statistic, p_value = stats.mannwhitneyu(first, second, alternative="two-sided")
    return {
        "test": "Mann-Whitney U",
        "groups": labels,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size_rank_biserial": float(
            1 - (2 * statistic) / (len(first) * len(second))
        ),
        "sample_sizes": [int(len(first)), int(len(second))],
    }


def wilcoxon_test(
    df: pd.DataFrame, first_column: str, second_column: str
) -> Dict[str, Any]:
    clean = df[[first_column, second_column]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if len(clean) < 3:
        raise ValueError("At least three complete pairs are required.")
    differences = clean[first_column] - clean[second_column]
    if (differences == 0).all():
        raise ValueError("All paired differences are zero.")
    statistic, p_value = stats.wilcoxon(
        clean[first_column], clean[second_column]
    )
    nonzero = differences[differences != 0]
    z_approx = stats.norm.ppf(max(p_value / 2, np.finfo(float).tiny))
    return {
        "test": "Wilcoxon signed-rank",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size_r": float(abs(z_approx) / np.sqrt(len(nonzero))),
        "pairs": int(len(clean)),
    }


def _group_values(
    df: pd.DataFrame, value_column: str, group_column: str
) -> tuple[List[pd.Series], List[str]]:
    clean = df[[value_column, group_column]].dropna()
    grouped = [
        (str(label), _values(group[value_column]))
        for label, group in clean.groupby(group_column, observed=True)
    ]
    grouped = [(label, values) for label, values in grouped if len(values) >= 2]
    if len(grouped) < 2:
        raise ValueError("At least two groups with two values each are required.")
    return [values for _, values in grouped], [label for label, _ in grouped]


def one_way_anova(
    df: pd.DataFrame, value_column: str, group_column: str
) -> Dict[str, Any]:
    groups, labels = _group_values(df, value_column, group_column)
    statistic, p_value = stats.f_oneway(*groups)
    all_values = pd.concat(groups)
    grand_mean = all_values.mean()
    between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)
    total = sum(((group - grand_mean) ** 2).sum() for group in groups)
    eta_squared = float(between / total) if total else 0.0
    return {
        "test": "One-way ANOVA",
        "group_labels": labels,
        "groups": len(groups),
        "f_statistic": float(statistic),
        "p_value": float(p_value),
        "eta_squared": eta_squared,
    }


def kruskal_wallis_test(
    df: pd.DataFrame, value_column: str, group_column: str
) -> Dict[str, Any]:
    groups, labels = _group_values(df, value_column, group_column)
    statistic, p_value = stats.kruskal(*groups)
    total = sum(len(group) for group in groups)
    epsilon_squared = max(
        0.0, float((statistic - len(groups) + 1) / (total - len(groups)))
    )
    return {
        "test": "Kruskal-Wallis",
        "group_labels": labels,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "epsilon_squared": epsilon_squared,
        "n": int(total),
    }


def levene_test(
    df: pd.DataFrame, value_column: str, group_column: str
) -> Dict[str, Any]:
    groups, labels = _group_values(df, value_column, group_column)
    statistic, p_value = stats.levene(*groups, center="median")
    return {
        "test": "Levene/Brown-Forsythe equal-variance test",
        "group_labels": labels,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "equal_variance_supported": bool(p_value >= 0.05),
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
        "cells_below_expected_5_%": float(np.mean(expected < 5) * 100),
        "observed_table": table.to_dict(),
    }


def bootstrap_mean_interval(
    series: pd.Series,
    confidence: float = 0.95,
    resamples: int = 5000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Return a reproducible percentile bootstrap interval for the mean."""
    values = _values(series).to_numpy()
    if len(values) < 2:
        raise ValueError("At least two numeric observations are required.")
    if not 100 <= resamples <= 100000:
        raise ValueError("Resamples must be between 100 and 100,000.")
    rng = np.random.default_rng(random_state)
    means = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    alpha = 1 - confidence
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "statistic": "mean",
        "estimate": float(np.mean(values)),
        "confidence": float(confidence),
        "lower": float(lower),
        "upper": float(upper),
        "resamples": int(resamples),
        "n": int(len(values)),
    }


def adjust_p_values(
    p_values: Iterable[float], method: str = "benjamini-hochberg"
) -> List[float]:
    """Adjust a family of p-values with Bonferroni or Benjamini-Hochberg."""
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0 or np.any((values < 0) | (values > 1)):
        raise ValueError("Provide p-values between zero and one.")
    if method == "bonferroni":
        return np.minimum(values * len(values), 1).tolist()
    if method != "benjamini-hochberg":
        raise ValueError("Method must be bonferroni or benjamini-hochberg.")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1)
    return adjusted.tolist()
