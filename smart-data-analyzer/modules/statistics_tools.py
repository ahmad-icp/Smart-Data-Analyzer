import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a descriptive statistics DataFrame with common metrics."""
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return pd.DataFrame()

    stats_df = numeric.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    stats_df["variance"] = numeric.var()
    stats_df["mode"] = numeric.mode().iloc[0]
    return stats_df


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return pd.DataFrame()
    return numeric.corr()


def covariance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return pd.DataFrame()
    return numeric.cov()


def linear_regression(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    """Fit a simple linear regression model (y = a + b*x) and return metrics."""
    numeric = df[[x_col, y_col]].dropna()
    if numeric.empty:
        return {}

    X = numeric[[x_col]].values.reshape(-1, 1)
    y = numeric[y_col].values
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)

    r2 = model.score(X, y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    return {
        "x": x_col,
        "y": y_col,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "n_samples": int(len(X)),
    }


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    numeric = df[[column]].dropna()
    if numeric.empty:
        return pd.DataFrame()

    z = np.abs(stats.zscore(numeric[column]))
    return df.loc[z > threshold]


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    series = df[column].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return df[(df[column] < lower) | (df[column] > upper)]


def t_test_independent(df: pd.DataFrame, col1: str, col2: str) -> dict:
    s1 = df[col1].dropna().astype(float)
    s2 = df[col2].dropna().astype(float)
    if len(s1) < 2 or len(s2) < 2:
        raise ValueError("Not enough data for t-test")

    t_stat, p_value = stats.ttest_ind(s1, s2, equal_var=False)
    return {
        "column_1": col1,
        "column_2": col2,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "n1": int(len(s1)),
        "n2": int(len(s2)),
    }
