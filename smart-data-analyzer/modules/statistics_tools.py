from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a broad, analysis-ready summary for every numeric column."""
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return pd.DataFrame()

    result = numeric.describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    ).T.rename(columns={"50%": "median"})
    result["missing"] = numeric.isna().sum()
    result["missing_%"] = numeric.isna().mean().mul(100)
    result["variance"] = numeric.var()
    result["range"] = result["max"] - result["min"]
    result["iqr"] = result["75%"] - result["25%"]
    result["skewness"] = numeric.skew()
    result["kurtosis"] = numeric.kurt()
    result["mode"] = [
        series.mode(dropna=True).iloc[0] if not series.mode(dropna=True).empty else np.nan
        for _, series in numeric.items()
    ]
    result["coefficient_of_variation_%"] = (
        result["std"].div(result["mean"].replace(0, np.nan)).abs().mul(100)
    )
    preferred = [
        "count",
        "missing",
        "missing_%",
        "mean",
        "std",
        "variance",
        "min",
        "5%",
        "25%",
        "median",
        "75%",
        "95%",
        "max",
        "range",
        "iqr",
        "skewness",
        "kurtosis",
        "mode",
        "coefficient_of_variation_%",
    ]
    return result[preferred]


def frequency_table(
    df: pd.DataFrame, column: str, include_missing: bool = True
) -> pd.DataFrame:
    """Return count, percentage and cumulative percentage for a column."""
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")
    counts = df[column].value_counts(dropna=not include_missing)
    total = counts.sum()
    result = counts.rename("count").to_frame()
    result["percentage"] = result["count"].div(total).mul(100)
    result["cumulative_percentage"] = result["percentage"].cumsum()
    result.index.name = column
    return result.reset_index()


def confidence_interval_mean(
    series: pd.Series, confidence: float = 0.95
) -> Dict[str, float]:
    """Calculate a Student-t confidence interval for a population mean."""
    values = _numeric_series(series)
    if len(values) < 2:
        raise ValueError("At least two numeric observations are required.")
    mean = float(values.mean())
    standard_error = float(stats.sem(values))
    critical = float(stats.t.ppf((1 + confidence) / 2, len(values) - 1))
    margin = critical * standard_error
    return {
        "mean": mean,
        "standard_error": standard_error,
        "confidence": float(confidence),
        "lower": mean - margin,
        "upper": mean + margin,
        "n": int(len(values)),
    }


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return pd.DataFrame()
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("Method must be pearson, spearman or kendall.")
    return numeric.corr(method=method)


def correlation_test(
    df: pd.DataFrame, x_col: str, y_col: str, method: str = "pearson"
) -> Dict[str, Any]:
    """Test a pairwise Pearson, Spearman or Kendall association."""
    clean = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean) < 3:
        raise ValueError("At least three paired numeric observations are required.")
    x, y = clean[x_col], clean[y_col]
    if x.nunique() < 2 or y.nunique() < 2:
        raise ValueError("Both columns must contain at least two distinct values.")

    if method == "pearson":
        result = stats.pearsonr(x, y)
        statistic, p_value = result.statistic, result.pvalue
        interval = result.confidence_interval(0.95)
        ci = [float(interval.low), float(interval.high)]
    elif method == "spearman":
        statistic, p_value = stats.spearmanr(x, y)
        ci = None
    elif method == "kendall":
        statistic, p_value = stats.kendalltau(x, y)
        ci = None
    else:
        raise ValueError("Method must be pearson, spearman or kendall.")

    return {
        "method": method.title(),
        "x": x_col,
        "y": y_col,
        "coefficient": float(statistic),
        "p_value": float(p_value),
        "confidence_interval_95": ci,
        "n": int(len(clean)),
    }


def covariance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return pd.DataFrame()
    return numeric.cov()


def linear_regression(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    """Fit a simple linear regression model and return fit and error metrics."""
    numeric = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(numeric) < 3:
        raise ValueError("At least three complete numeric rows are required.")

    X = numeric[[x_col]].to_numpy()
    y = numeric[y_col].to_numpy()
    model = LinearRegression().fit(X, y)
    prediction = model.predict(X)
    residual = y - prediction
    return {
        "x": x_col,
        "y": y_col,
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(model.score(X, y)),
        "adjusted_r2": float(
            1 - (1 - model.score(X, y)) * (len(y) - 1) / (len(y) - 2)
        ),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "n_samples": int(len(X)),
    }


def multiple_linear_regression(
    df: pd.DataFrame, predictors: Iterable[str], target: str
) -> Dict[str, Any]:
    """Fit OLS with inference, diagnostics and coefficient confidence intervals."""
    import statsmodels.api as sm

    predictor_list = list(dict.fromkeys(predictors))
    if not predictor_list:
        raise ValueError("Select at least one predictor.")
    if target in predictor_list:
        raise ValueError("The target cannot also be a predictor.")
    columns: List[str] = predictor_list + [target]
    clean = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    minimum_rows = len(predictor_list) + 3
    if len(clean) < minimum_rows:
        raise ValueError(
            f"At least {minimum_rows} complete rows are required for this model."
        )
    X = sm.add_constant(clean[predictor_list], has_constant="add")
    model = sm.OLS(clean[target], X).fit()
    intervals = model.conf_int()
    coefficients = pd.DataFrame(
        {
            "coefficient": model.params,
            "standard_error": model.bse,
            "t_statistic": model.tvalues,
            "p_value": model.pvalues,
            "ci_95_lower": intervals[0],
            "ci_95_upper": intervals[1],
        }
    )
    jb_stat, jb_pvalue, skew, kurtosis = sm.stats.jarque_bera(model.resid)
    return {
        "metrics": {
            "r2": float(model.rsquared),
            "adjusted_r2": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue) if model.fvalue is not None else None,
            "f_p_value": float(model.f_pvalue)
            if model.f_pvalue is not None
            else None,
            "aic": float(model.aic),
            "bic": float(model.bic),
            "durbin_watson": float(sm.stats.stattools.durbin_watson(model.resid)),
            "jarque_bera_p_value": float(jb_pvalue),
            "residual_skewness": float(skew),
            "residual_kurtosis": float(kurtosis),
            "n": int(model.nobs),
        },
        "coefficients": coefficients,
        "predictions": pd.DataFrame(
            {
                "actual": clean[target],
                "predicted": model.fittedvalues,
                "residual": model.resid,
            }
        ),
    }


def detect_outliers_zscore(
    df: pd.DataFrame, column: str, threshold: float = 3.0
) -> pd.DataFrame:
    series = pd.to_numeric(df[column], errors="coerce")
    valid = series.dropna()
    if valid.empty or valid.std(ddof=0) == 0:
        return df.iloc[0:0].copy()
    z_score = pd.Series(
        np.abs(stats.zscore(valid)), index=valid.index, dtype=float
    )
    return df.loc[z_score[z_score > threshold].index].copy()


def detect_outliers_iqr(
    df: pd.DataFrame, column: str, multiplier: float = 1.5
) -> pd.DataFrame:
    series = pd.to_numeric(df[column], errors="coerce")
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return df.iloc[0:0].copy()
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    return df.loc[series.notna() & ((series < lower) | (series > upper))].copy()


def t_test_independent(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Backward-compatible Welch t-test between two numeric columns."""
    first = _numeric_series(df[col1])
    second = _numeric_series(df[col2])
    if len(first) < 2 or len(second) < 2:
        raise ValueError("At least two observations are required in each sample.")
    statistic, p_value = stats.ttest_ind(first, second, equal_var=False)
    pooled = np.sqrt(
        ((len(first) - 1) * first.var() + (len(second) - 1) * second.var())
        / (len(first) + len(second) - 2)
    )
    effect = (first.mean() - second.mean()) / pooled if pooled else 0.0
    return {
        "test": "Welch independent t-test",
        "column_1": col1,
        "column_2": col2,
        "t_statistic": float(statistic),
        "p_value": float(p_value),
        "cohens_d": float(effect),
        "n1": int(len(first)),
        "n2": int(len(second)),
    }
