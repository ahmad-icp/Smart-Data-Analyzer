import io
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_problem_type(target: pd.Series) -> str:
    unique = target.nunique(dropna=True)
    if (
        not pd.api.types.is_numeric_dtype(target)
        or pd.api.types.is_bool_dtype(target)
        or unique <= max(20, int(np.sqrt(max(len(target), 1))))
    ):
        return "classification"
    return "regression"


def _stratify_target(y: pd.Series, problem_type: str) -> Optional[pd.Series]:
    if problem_type != "classification":
        return None
    counts = y.value_counts()
    return y if len(counts) > 1 and counts.min() >= 3 else None


def split_dataset(
    df: pd.DataFrame,
    target: str,
    problem_type: str = "auto",
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    if target not in df.columns:
        raise ValueError("Select a valid target column.")
    clean = df.dropna(subset=[target]).copy()
    if len(clean) < 20:
        raise ValueError("At least 20 rows with a non-missing target are required.")
    resolved_type = (
        infer_problem_type(clean[target]) if problem_type == "auto" else problem_type
    )
    train_val, test = train_test_split(
        clean,
        test_size=test_size,
        random_state=random_state,
        stratify=_stratify_target(clean[target], resolved_type),
    )
    relative_validation = validation_size / (1 - test_size)
    train, validation = train_test_split(
        train_val,
        test_size=relative_validation,
        random_state=random_state,
        stratify=_stratify_target(train_val[target], resolved_type),
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
        "problem_type": resolved_type,
    }


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    max_categories=100,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def _classification_metrics(y_true, prediction) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision_weighted": float(
            precision_score(y_true, prediction, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, prediction, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, prediction, average="weighted", zero_division=0)
        ),
    }


def _regression_metrics(y_true, prediction) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
    }


def _feature_importance(pipeline: Pipeline, limit: int = 20) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        names = preprocessor.get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            coefficients = np.asarray(model.coef_)
            values = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
        else:
            return pd.DataFrame()
        order = np.argsort(values)[::-1][:limit]
        return pd.DataFrame(
            {"feature": names[order], "importance": values[order]}
        ).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def run_automl(
    df: pd.DataFrame,
    target: str,
    problem_type: str = "auto",
    random_state: int = 42,
    max_rows: int = 50000,
    exclude_features: Optional[list] = None,
) -> Dict[str, Any]:
    """Compare compact baseline models and evaluate the winner on held-out test data."""
    clean = df.dropna(subset=[target]).copy()
    if len(clean) > max_rows:
        clean = clean.sample(max_rows, random_state=random_state)
    splits = split_dataset(
        clean, target, problem_type=problem_type, random_state=random_state
    )
    resolved_type = splits["problem_type"]
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    excluded = [
        column
        for column in (exclude_features or [])
        if column in clean.columns and column != target
    ]
    feature_columns = [
        column for column in clean.columns if column != target and column not in excluded
    ]
    X_train, y_train = train[feature_columns].copy(), train[target]
    X_validation, y_validation = (
        validation[feature_columns].copy(),
        validation[target],
    )
    X_test, y_test = test[feature_columns].copy(), test[target]
    if X_train.shape[1] == 0:
        raise ValueError("At least one feature column is required.")
    categorical = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(X_train[column])
    ]
    for frame in (X_train, X_validation, X_test):
        for column in categorical:
            frame[column] = frame[column].map(
                lambda value: np.nan if pd.isna(value) else str(value)
            )

    if resolved_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, class_weight="balanced"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=random_state
            ),
        }
        metric_function = _classification_metrics
        selection_metric = "f1_weighted"
        maximize = True
    else:
        models = {
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=150, random_state=random_state, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=random_state
            ),
        }
        metric_function = _regression_metrics
        selection_metric = "rmse"
        maximize = False

    evaluations = []
    fitted = {}
    for name, model in models.items():
        pipeline = Pipeline(
            [("preprocess", _preprocessor(X_train)), ("model", model)]
        )
        try:
            pipeline.fit(X_train, y_train)
            metrics = metric_function(y_validation, pipeline.predict(X_validation))
            evaluations.append({"model": name, **metrics, "error": None})
            fitted[name] = pipeline
        except Exception as exc:
            evaluations.append({"model": name, "error": str(exc)})

    successful = [
        row for row in evaluations if row.get(selection_metric) is not None
    ]
    if not successful:
        raise RuntimeError("All candidate models failed. Review the feature columns.")
    best_row = sorted(
        successful, key=lambda row: row[selection_metric], reverse=maximize
    )[0]
    best_name = best_row["model"]
    best_pipeline = fitted[best_name]
    test_metrics = metric_function(y_test, best_pipeline.predict(X_test))

    model_buffer = io.BytesIO()
    joblib.dump(best_pipeline, model_buffer)
    return {
        "problem_type": resolved_type,
        "target": target,
        "results": pd.DataFrame(evaluations),
        "best_model_name": best_name,
        "validation_metrics": {
            key: value for key, value in best_row.items() if key not in {"model", "error"}
        },
        "test_metrics": test_metrics,
        "feature_importance": _feature_importance(best_pipeline),
        "model_bytes": model_buffer.getvalue(),
        "splits": splits,
        "sampled_rows": len(clean),
        "excluded_features": excluded,
    }
