import io
import json
import warnings
import zipfile
from typing import Any, Dict, List, Optional

import pandas as pd


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def statistics_to_csv(stats_df: pd.DataFrame) -> bytes:
    """Convert a statistics DataFrame to CSV bytes for download."""
    return stats_df.to_csv(index=True).encode("utf-8")


def dataframe_to_excel(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned Data")
    return buffer.getvalue()


def dataframe_to_json(df: pd.DataFrame) -> bytes:
    return df.to_json(orient="records", indent=2, date_format="iso").encode("utf-8")


def dataframe_to_parquet(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


def build_export_package(
    df: pd.DataFrame,
    readme: str,
    data_dictionary: pd.DataFrame,
    history: List[Dict[str, Any]],
    quality_report: Dict[str, Any],
    original_df: pd.DataFrame = None,
    splits: Optional[Dict[str, Any]] = None,
    automl_summary: Optional[Dict[str, Any]] = None,
    model_bytes: Optional[bytes] = None,
) -> bytes:
    """Create a self-documenting, reproducible dataset export ZIP."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.md", readme)
        archive.writestr("data/cleaned_dataset.csv", dataframe_to_csv(df))
        archive.writestr("data/cleaned_dataset.json", dataframe_to_json(df))
        if original_df is not None:
            archive.writestr("data/original_dataset.csv", dataframe_to_csv(original_df))
        archive.writestr(
            "documentation/data_dictionary.csv",
            data_dictionary.to_csv(index=False).encode("utf-8"),
        )
        archive.writestr(
            "documentation/quality_report.json",
            json.dumps(quality_report, indent=2, default=str),
        )
        archive.writestr(
            "documentation/transformation_history.json",
            json.dumps(history, indent=2, default=str),
        )
        if splits:
            for name in ("train", "validation", "test"):
                split = splits.get(name)
                if isinstance(split, pd.DataFrame):
                    archive.writestr(
                        f"data/{name}.csv", dataframe_to_csv(split)
                    )
        if automl_summary:
            archive.writestr(
                "model/automl_results.json",
                json.dumps(automl_summary, indent=2, default=str),
            )
        if model_bytes:
            archive.writestr("model/best_model.joblib", model_bytes)
        archive.writestr(
            "reproduce.py",
            (
                '"""Load the exported cleaned dataset."""\n'
                "import pandas as pd\n\n"
                'df = pd.read_csv("data/cleaned_dataset.csv")\n'
                'print(f"Loaded {len(df)} rows and {len(df.columns)} columns")\n'
            ),
        )
        try:
            archive.writestr("data/cleaned_dataset.xlsx", dataframe_to_excel(df))
        except Exception:
            pass
        try:
            archive.writestr("data/cleaned_dataset.parquet", dataframe_to_parquet(df))
        except Exception:
            pass
    return buffer.getvalue()


def plot_to_image_bytes(fig, width: int = 800, height: int = 600) -> bytes:
    """Render a Plotly figure to PNG bytes using the bundled Kaleido renderer."""
    try:
        import plotly.io as pio

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return pio.to_image(fig, format="png", width=width, height=height)
    except Exception as e:
        raise RuntimeError(
            "Failed to render the chart. Reinstall the dependencies from "
            f"requirements.txt. Details: {e}"
        )
