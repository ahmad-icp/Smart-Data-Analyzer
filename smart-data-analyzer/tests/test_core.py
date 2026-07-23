import io
import zipfile

import numpy as np
import pandas as pd

from modules.advanced_statistics import (
    chi_square_test,
    mann_whitney_test,
    normality_test,
    one_way_anova,
)
from modules.ai_export_writer import generate_data_card
from modules.automl import run_automl, split_dataset
from modules.data_cleaning import (
    clean_string_column,
    normalize_column,
    remove_outliers_iqr,
    remove_outliers_zscore,
    standardize_categories,
)
from modules.export_tools import build_export_package
from modules.ml_readiness import (
    analyze_ml_readiness,
    build_data_dictionary,
    detect_leakage_risks,
    detect_sensitive_columns,
)
from modules.natural_language_cleaning import apply_plan, parse_instruction
from modules.report_generator import generate_pdf_report
from modules.visualization import create_plot


def test_constant_numeric_columns_are_safe():
    df = pd.DataFrame({"value": [4, 4, 4]})
    assert normalize_column(df, "value", "minmax")["value"].tolist() == [0, 0, 0]
    assert normalize_column(df, "value", "zscore")["value"].tolist() == [0, 0, 0]
    assert len(remove_outliers_zscore(df, "value")) == 3
    assert len(remove_outliers_iqr(df, "value")) == 3


def test_string_cleaning_preserves_missing_values():
    df = pd.DataFrame({"city": [" Islamabad! ", None, np.nan]})
    cleaned = clean_string_column(df, "city")
    standardized = standardize_categories(df, "city")
    assert cleaned.loc[0, "city"] == "Islamabad"
    assert cleaned["city"].isna().sum() == 2
    assert standardized["city"].isna().sum() == 2


def test_ml_readiness_and_dictionary_are_explainable():
    df = pd.DataFrame(
        {"id": range(30), "target": ["yes"] * 28 + ["no"] * 2, "x": [1] * 30}
    )
    report = analyze_ml_readiness(df, "target")
    dictionary = build_data_dictionary(df)
    assert 0 <= report["score"] < 100
    assert any("imbalanced" in issue for issue in report["issues"])
    assert set(["column", "dtype", "missing", "unique"]).issubset(dictionary.columns)


def test_writer_always_has_a_no_api_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    df = pd.DataFrame({"age": [20, 21], "city": ["A", "B"]})
    quality = analyze_ml_readiness(df)
    text, provider, warning = generate_data_card(
        df, quality, "Example Dataset", provider="template"
    )
    assert text.startswith("# Example Dataset")
    assert "Column guide" in text
    assert provider == "Built-in professional writer"
    assert warning is None


def test_export_package_is_self_documenting():
    df = pd.DataFrame({"value": [1, 2]})
    package = build_export_package(
        df,
        "# Example",
        build_data_dictionary(df),
        [{"step": 1, "action": "test"}],
        analyze_ml_readiness(df),
    )
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        names = set(archive.namelist())
    assert "README.md" in names
    assert "data/cleaned_dataset.csv" in names
    assert "documentation/data_dictionary.csv" in names
    assert "documentation/transformation_history.json" in names
    assert "reproduce.py" in names


def test_current_pandas_bar_chart_and_pdf_export():
    df = pd.DataFrame({"category": ["A", "B", "A"], "value": [1, 2, 3]})
    figure = create_plot(df, "Bar", "category")
    assert list(figure.data[0].y) == [2, 1]
    scatter = create_plot(df, "Scatter", "value", "value")
    assert len(scatter.data[0].x) == 3
    pdf = generate_pdf_report("# Test\n\n## Overview\nAll good.", [])
    assert pdf.startswith(b"%PDF")


def test_natural_language_plan_is_validated_before_execution():
    df = pd.DataFrame(
        {
            "Age": [20, None, 30, 30],
            "City": [" ISLAMABAD ", "lahore", "Lahore", "Lahore"],
        }
    )
    plan = parse_instruction(
        "Remove duplicates, fill missing values in Age with median, "
        "and standardize categories in City.",
        df.columns.tolist(),
    )
    cleaned = apply_plan(df, plan)
    assert cleaned["Age"].isna().sum() == 0
    assert set(cleaned["City"]) == {"Islamabad", "Lahore"}
    assert all(item["action"] in {
        "remove_duplicates", "fill_missing", "standardize_categories"
    } for item in plan)


def test_sensitive_data_and_leakage_checks():
    df = pd.DataFrame(
        {
            "email": [f"user{i}@example.com" for i in range(30)],
            "target": [0, 1] * 15,
            "copied_target": [0, 1] * 15,
        }
    )
    sensitive = detect_sensitive_columns(df)
    leakage = detect_leakage_risks(df, "target")
    report = analyze_ml_readiness(df, "target")
    assert sensitive["email"]
    assert any("copied_target" in risk for risk in leakage)
    assert report["score"] < 100


def test_advanced_statistics_return_effect_sizes():
    df = pd.DataFrame(
        {
            "value": [1, 2, 3, 4, 5, 6, 7, 8],
            "group": ["A"] * 4 + ["B"] * 4,
            "category": ["X", "X", "Y", "Y", "X", "Y", "X", "Y"],
        }
    )
    assert "p_value" in normality_test(df["value"])
    assert "effect_size_rank_biserial" in mann_whitney_test(
        df, "value", "group"
    )
    assert "eta_squared" in one_way_anova(df, "value", "group")
    assert "cramers_v" in chi_square_test(df, "group", "category")


def test_automl_creates_reproducible_splits_and_downloadable_model():
    rng = np.random.default_rng(42)
    size = 120
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, size),
            "income": rng.normal(50000, 10000, size),
            "city": np.where(np.arange(size) % 2 == 0, "A", "B"),
            "mixed_category": [1 if index % 3 == 0 else "unknown" for index in range(size)],
        }
    )
    df["target"] = (df["age"] + df["income"] / 10000 > 42).astype(int)
    splits = split_dataset(df, "target", random_state=7)
    result = run_automl(df, "target", random_state=7)
    assert sum(len(splits[name]) for name in ("train", "validation", "test")) == size
    assert result["best_model_name"]
    assert result["model_bytes"]
    assert "f1_weighted" in result["test_metrics"]
    package = build_export_package(
        df,
        "# AutoML Dataset",
        build_data_dictionary(df),
        [],
        analyze_ml_readiness(df, "target"),
        splits=result["splits"],
        automl_summary={"best_model": result["best_model_name"]},
        model_bytes=result["model_bytes"],
    )
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        names = set(archive.namelist())
    assert "data/train.csv" in names
    assert "data/validation.csv" in names
    assert "data/test.csv" in names
    assert "model/best_model.joblib" in names
