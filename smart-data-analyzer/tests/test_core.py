import io
import importlib
import zipfile

import numpy as np
import pandas as pd

from modules.advanced_statistics import (
    adjust_p_values,
    bootstrap_mean_interval,
    chi_square_test,
    independent_t_test,
    kruskal_wallis_test,
    levene_test,
    mann_whitney_test,
    normality_test,
    one_sample_t_test,
    one_way_anova,
    paired_t_test,
    wilcoxon_test,
)
from modules.ai_export_writer import generate_data_card
from modules.automl import run_automl, split_dataset
from modules.data_cleaning import (
    clean_string_column,
    convert_column_type,
    log_transform,
    normalize_column,
    remove_outliers_iqr,
    remove_outliers_zscore,
    standardize_categories,
)
from modules.dashboard_builder import apply_filters, create_dashboard_chart
from modules.data_loader import load_dataset_bytes
from modules.export_tools import (
    dataframe_to_excel,
    dataframe_to_json,
    dataframe_to_parquet,
)
from modules.export_tools import build_export_package, plot_to_image_bytes
from modules.ml_readiness import (
    analyze_ml_readiness,
    build_data_dictionary,
    detect_leakage_risks,
    detect_sensitive_columns,
)
from modules.natural_language_cleaning import apply_plan, parse_instruction
from modules.report_generator import generate_pdf_report
from modules.statistics_tools import (
    confidence_interval_mean,
    correlation_test,
    descriptive_statistics,
    detect_outliers_iqr,
    detect_outliers_zscore,
    frequency_table,
    multiple_linear_regression,
)
from modules.visualization import create_plot


def test_advertised_runtime_dependencies_are_installed():
    for package in ("xlrd", "openpyxl", "pyarrow", "kaleido", "statsmodels"):
        assert importlib.import_module(package)


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


def test_type_conversion_and_log_transform_preserve_invalid_values():
    df = pd.DataFrame({"label": ["A", None], "value": [1, -2]})
    converted = convert_column_type(df, "label", "str")
    transformed = log_transform(df, "value")
    assert converted["label"].isna().sum() == 1
    assert transformed.loc[0, "value_log"] == np.log(2)
    assert pd.isna(transformed.loc[1, "value_log"])


def test_advertised_data_formats_round_trip():
    source = pd.DataFrame({"value": [1, 2], "label": ["A", "B"]})
    csv_loaded = load_dataset_bytes(source.to_csv(index=False).encode(), "data.csv")
    json_loaded = load_dataset_bytes(dataframe_to_json(source), "data.json")
    excel_loaded = load_dataset_bytes(dataframe_to_excel(source), "data.xlsx")
    parquet_loaded = load_dataset_bytes(dataframe_to_parquet(source), "data.parquet")
    for loaded in (csv_loaded, json_loaded, excel_loaded, parquet_loaded):
        assert loaded.shape == source.shape
        assert loaded["value"].tolist() == [1, 2]


def test_dashboard_filters_and_charts_are_functional():
    df = pd.DataFrame(
        {"group": ["A", "A", "B"], "value": [1, 3, 7], "other": [2, 4, 8]}
    )
    ranged = apply_filters(
        df, [{"column": "value", "op": "range", "value": (2, 8)}]
    )
    selected = apply_filters(
        df, [{"column": "group", "op": "in", "value": ["A"]}]
    )
    assert ranged["value"].tolist() == [3, 7]
    assert len(selected) == 2
    assert create_dashboard_chart(df, "Scatter", "value", "other") is not None
    assert create_dashboard_chart(df, "Heatmap") is not None


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


def test_basic_statistics_are_complete_and_missing_safe():
    df = pd.DataFrame(
        {
            "value": [1, 2, 3, 4, 100, np.nan],
            "other": [2, 4, 6, 8, 200, 12],
            "category": ["A", "A", "B", "B", "B", None],
        }
    )
    summary = descriptive_statistics(df)
    assert {"missing_%", "iqr", "skewness", "kurtosis"}.issubset(summary.columns)
    assert summary.loc["value", "missing"] == 1
    assert frequency_table(df, "category")["count"].sum() == len(df)
    interval = confidence_interval_mean(df["value"])
    assert interval["lower"] < interval["mean"] < interval["upper"]
    association = correlation_test(df, "value", "other", "spearman")
    assert association["coefficient"] > 0.9
    assert list(detect_outliers_zscore(df, "value", threshold=1.5).index) == [4]
    assert list(detect_outliers_iqr(df, "value").index) == [4]


def test_parametric_nonparametric_and_robust_statistics():
    df = pd.DataFrame(
        {
            "before": [10, 12, 13, 15, 16, 18, 20, 22],
            "after": [11, 13, 15, 16, 18, 20, 23, 25],
            "value": [1, 2, 3, 4, 10, 11, 12, 14],
            "group": ["A"] * 4 + ["B"] * 4,
        }
    )
    assert "cohens_d" in independent_t_test(df, "value", "group")
    assert "cohens_dz" in paired_t_test(df, "before", "after")
    assert "cohens_d" in one_sample_t_test(df["value"], 5)
    assert "effect_size_r" in wilcoxon_test(df, "before", "after")
    assert "epsilon_squared" in kruskal_wallis_test(df, "value", "group")
    assert "equal_variance_supported" in levene_test(df, "value", "group")
    bootstrap = bootstrap_mean_interval(df["value"], resamples=500)
    assert bootstrap["lower"] < bootstrap["estimate"] < bootstrap["upper"]
    adjusted = adjust_p_values([0.01, 0.03, 0.2])
    assert len(adjusted) == 3
    assert all(0 <= value <= 1 for value in adjusted)


def test_multiple_ols_returns_inference_and_diagnostics():
    rng = np.random.default_rng(7)
    x1 = np.arange(1, 61, dtype=float)
    x2 = rng.normal(size=60)
    y = 4 + 2.5 * x1 - 1.2 * x2 + rng.normal(scale=0.25, size=60)
    result = multiple_linear_regression(
        pd.DataFrame({"x1": x1, "x2": x2, "target": y}),
        ["x1", "x2"],
        "target",
    )
    assert result["metrics"]["r2"] > 0.99
    assert {"coefficient", "p_value", "ci_95_lower"}.issubset(
        result["coefficients"].columns
    )
    assert len(result["predictions"]) == 60


def test_chart_png_export_works_without_external_browser():
    figure = create_plot(
        pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 8]}),
        "Scatter",
        "x",
        "y",
    )
    image = plot_to_image_bytes(figure, width=400, height=300)
    assert image.startswith(b"\x89PNG\r\n\x1a\n")


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


def test_automl_regression_path_is_functional():
    rng = np.random.default_rng(21)
    size = 100
    first = rng.normal(size=size)
    second = rng.normal(size=size)
    target = 3.5 * first - 1.7 * second + rng.normal(scale=0.15, size=size)
    result = run_automl(
        pd.DataFrame({"first": first, "second": second, "target": target}),
        "target",
        problem_type="regression",
        random_state=21,
    )
    assert result["problem_type"] == "regression"
    assert result["best_model_name"]
    assert result["test_metrics"]["rmse"] >= 0
    assert result["model_bytes"]
