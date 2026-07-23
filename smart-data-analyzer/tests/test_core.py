import io
import zipfile

import numpy as np
import pandas as pd

from modules.ai_export_writer import generate_data_card
from modules.data_cleaning import (
    clean_string_column,
    normalize_column,
    remove_outliers_iqr,
    remove_outliers_zscore,
    standardize_categories,
)
from modules.export_tools import build_export_package
from modules.ml_readiness import analyze_ml_readiness, build_data_dictionary
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
