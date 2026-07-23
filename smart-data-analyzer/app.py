import hashlib
from pathlib import Path

import streamlit as st
import pandas as pd

from modules.ai_cleaning import analyze_dataset, apply_suggestion
from modules.advanced_statistics import (
    chi_square_test,
    mann_whitney_test,
    normality_test,
    one_way_anova,
)
from modules.ai_export_writer import generate_cleaning_plan, generate_data_card
from modules.automl import run_automl, split_dataset
from modules.data_loader import load_dataset_bytes
from modules.data_cleaning import (
    clean_string_column,
    convert_column_type,
    extract_date_parts,
    fill_missing,
    log_transform,
    normalize_column,
    parse_dates,
    remove_duplicates,
    remove_missing,
    replace_values,
)
from modules.data_profiling import (
    column_statistics,
    correlation_heatmap,
    dataset_overview,
    missing_heatmap,
    numeric_distribution_plots,
)
from modules.dashboard_builder import apply_filters, create_dashboard_chart
from modules.export_tools import (
    build_export_package,
    dataframe_to_csv,
    dataframe_to_excel,
    dataframe_to_json,
    dataframe_to_parquet,
    plot_to_image_bytes,
    statistics_to_csv,
)
from modules.ml_readiness import analyze_ml_readiness, build_data_dictionary
from modules.natural_language_cleaning import (
    apply_plan,
    parse_instruction,
    plan_summary,
)
from modules.report_generator import (
    generate_markdown_report,
    generate_pdf_report,
    markdown_to_html,
    build_chart_image_base64,
)
from modules.statistics_tools import (
    covariance_matrix,
    descriptive_statistics,
    linear_regression,
    t_test_independent,
)
from modules.visualization import create_plot, suggest_chart_types


st.set_page_config(
    page_title="Smart Data Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state():
    if "df" not in st.session_state:
        st.session_state.df = None
    if "original_df" not in st.session_state:
        st.session_state.original_df = None
    if "last_plot" not in st.session_state:
        st.session_state.last_plot = None
    if "ai_suggestions" not in st.session_state:
        st.session_state.ai_suggestions = []
    if "dashboard_charts" not in st.session_state:
        st.session_state.dashboard_charts = []
    if "dashboard_filters" not in st.session_state:
        st.session_state.dashboard_filters = []
    if "undo_stack" not in st.session_state:
        st.session_state.undo_stack = []
    if "transformation_history" not in st.session_state:
        st.session_state.transformation_history = []
    if "loaded_file_id" not in st.session_state:
        st.session_state.loaded_file_id = None
    if "generated_data_card" not in st.session_state:
        st.session_state.generated_data_card = None
    if "data_card_provider" not in st.session_state:
        st.session_state.data_card_provider = None
    if "nl_cleaning_plan" not in st.session_state:
        st.session_state.nl_cleaning_plan = []
    if "nl_cleaning_preview" not in st.session_state:
        st.session_state.nl_cleaning_preview = None
    if "automl_result" not in st.session_state:
        st.session_state.automl_result = None
    if "dataset_splits" not in st.session_state:
        st.session_state.dataset_splits = None


def commit_dataframe(new_df, action):
    """Save an auditable mutation and preserve a bounded undo snapshot."""
    current = st.session_state.df
    if current is not None:
        st.session_state.undo_stack.append(current.copy(deep=True))
        st.session_state.undo_stack = st.session_state.undo_stack[-10:]
    before_rows = len(current) if current is not None else 0
    st.session_state.df = new_df.copy()
    st.session_state.transformation_history.append(
        {
            "step": len(st.session_state.transformation_history) + 1,
            "action": action,
            "rows_before": before_rows,
            "rows_after": len(new_df),
            "columns_after": len(new_df.columns),
            "missing_before": (
                int(current.isna().sum().sum()) if current is not None else 0
            ),
            "missing_after": int(new_df.isna().sum().sum()),
            "duplicates_after": int(new_df.duplicated().sum()),
        }
    )
    st.session_state.dashboard_charts = []
    st.session_state.generated_data_card = None
    st.session_state.nl_cleaning_plan = []
    st.session_state.nl_cleaning_preview = None
    st.session_state.automl_result = None
    st.session_state.dataset_splits = None
    refresh_suggestions()


def undo_last_change():
    if not st.session_state.undo_stack:
        return False
    st.session_state.df = st.session_state.undo_stack.pop()
    if st.session_state.transformation_history:
        st.session_state.transformation_history.pop()
    st.session_state.dashboard_charts = []
    st.session_state.generated_data_card = None
    st.session_state.automl_result = None
    st.session_state.dataset_splits = None
    refresh_suggestions()
    return True


def load_dataset_from_uploader(uploaded_file):
    if uploaded_file is None:
        return None

    try:
        content = uploaded_file.getvalue()
        df = load_dataset_bytes(content, uploaded_file.name)
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None


def refresh_suggestions():
    if st.session_state.df is not None:
        st.session_state.ai_suggestions = analyze_dataset(st.session_state.df)
    else:
        st.session_state.ai_suggestions = []


def sidebar_upload():
    st.sidebar.title("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV, Excel, JSON, TSV or Parquet",
        type=["csv", "tsv", "xlsx", "xls", "json", "jsonl", "parquet"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        file_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        if file_id != st.session_state.loaded_file_id:
            df = load_dataset_from_uploader(uploaded_file)
            if df is not None:
                st.session_state.df = df.copy()
                st.session_state.original_df = df.copy()
                st.session_state.loaded_file_id = file_id
                st.session_state.undo_stack = []
                st.session_state.transformation_history = []
                st.session_state.dashboard_charts = []
                st.session_state.dashboard_filters = []
                st.session_state.generated_data_card = None
                st.session_state.nl_cleaning_plan = []
                st.session_state.nl_cleaning_preview = None
                st.session_state.automl_result = None
                st.session_state.dataset_splits = None
                refresh_suggestions()
                st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    if st.session_state.df is not None:
        if st.sidebar.button("Reset to original"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.undo_stack = []
            st.session_state.transformation_history = []
            st.session_state.dashboard_charts = []
            st.session_state.generated_data_card = None
            st.session_state.automl_result = None
            st.session_state.dataset_splits = None
            refresh_suggestions()
            st.sidebar.success("Reset to original dataset.")
        if st.sidebar.button(
            "Undo last change", disabled=not st.session_state.undo_stack
        ):
            if undo_last_change():
                st.sidebar.success("Last transformation was undone.")

    st.sidebar.markdown("---")


def sidebar_cleaning_tools():
    st.sidebar.header("Cleaning & Transformation")
    if st.session_state.df is None:
        st.sidebar.info("Upload a dataset to enable cleaning tools.")
        return

    df = st.session_state.df
    cols = list(df.columns)

    with st.sidebar.expander("Quick Cleaning"):
        if st.button("Remove missing rows"):
            commit_dataframe(remove_missing(df), "Remove rows containing missing values")
            st.success("Removed rows with missing values.")

        if st.button("Remove duplicate rows"):
            commit_dataframe(remove_duplicates(df), "Remove duplicate rows")
            st.success("Removed duplicate rows.")

    with st.sidebar.expander("Missing Value Handling"):
        fill_method = st.selectbox("Fill missing using", ["Mean", "Median", "Mode", "Custom"], index=0)
        if fill_method == "Custom":
            custom_value = st.text_input("Custom fill value", value="")
        else:
            custom_value = None

        selected_cols = st.multiselect("Columns", cols, default=cols)
        if st.button("Apply fill"):
            updated = fill_missing(
                st.session_state.df,
                columns=selected_cols,
                method=fill_method.lower(),
                custom_value=custom_value,
            )
            commit_dataframe(updated, f"Fill missing values using {fill_method.lower()}")
            st.success("Filled missing values.")

    with st.sidebar.expander("Value replacement"):
        replacement_text = st.text_area(
            "Replacement rules (old:new per line)",
            value="NA: , N/A: , null: ",
            help="Enter key:value pairs separated by newlines.",
        )
        if st.button("Apply replacements"):
            mapping = {}
            for line in replacement_text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    mapping[k.strip()] = v.strip()
            commit_dataframe(
                replace_values(st.session_state.df, mapping), "Apply value replacements"
            )
            st.success("Applied value replacements.")

    with st.sidebar.expander("String Cleaning"):
        str_col = st.selectbox("Column", [None] + cols, key="clean_str_col")
        if str_col:
            trim = st.checkbox("Trim whitespace", value=True)
            remove_special = st.checkbox("Remove special characters", value=True)
            case = st.selectbox("Case", ["none", "lower", "upper", "title"], index=0)
            if st.button("Clean strings"):
                updated = clean_string_column(
                    st.session_state.df,
                    str_col,
                    trim=trim,
                    remove_special=remove_special,
                    case=case if case != "none" else None,
                )
                commit_dataframe(updated, f"Clean strings in {str_col}")
                st.success(f"Cleaned strings in '{str_col}'.")

    with st.sidebar.expander("Numeric Transformations"):
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        num_col = st.selectbox("Numeric column", [None] + num_cols, key="num_trans_col")
        if num_col:
            transform = st.selectbox("Transformation", ["Log", "Normalize (0-1)", "Standardize (z)"])
            if st.button("Apply transformation"):
                if transform == "Log":
                    updated = log_transform(st.session_state.df, num_col)
                elif transform == "Normalize (0-1)":
                    updated = normalize_column(st.session_state.df, num_col, method="minmax")
                else:
                    updated = normalize_column(st.session_state.df, num_col, method="zscore")
                commit_dataframe(updated, f"{transform}: {num_col}")
                st.success(f"Applied {transform} to {num_col}.")

    with st.sidebar.expander("Date Parsing"):
        date_col = st.selectbox("Date column", [None] + cols, key="date_col")
        date_format = st.text_input("Format (optional)", value="")
        if date_col and st.button("Parse dates"):
            updated = parse_dates(
                st.session_state.df, date_col, format=date_format or None
            )
            updated = extract_date_parts(updated, date_col)
            commit_dataframe(updated, f"Parse date and extract parts: {date_col}")
            st.success(f"Parsed dates in '{date_col}'.")

    with st.sidebar.expander("Type Conversion"):
        col_type = st.selectbox("Column", [None] + cols, key="convert_col")
        if col_type:
            target_type = st.selectbox(
                "Type", ["int", "float", "str", "datetime", "category"], key="convert_type"
            )
            if st.button("Convert type"):
                updated = convert_column_type(
                    st.session_state.df, col_type, target_type
                )
                commit_dataframe(updated, f"Convert {col_type} to {target_type}")
                st.success(f"Converted {col_type} to {target_type}.")

    with st.sidebar.expander("Filters"):
        st.write("Create a filter for interactive dashboards")
        filter_col = st.selectbox("Filter column", [None] + cols, key="filter_column")
        if filter_col:
            unique_vals = st.session_state.df[filter_col].dropna().unique().tolist()
            if st.session_state.df[filter_col].dtype.name in ["int64", "float64"]:
                min_val = float(st.session_state.df[filter_col].min())
                max_val = float(st.session_state.df[filter_col].max())
                lo, hi = st.slider(
                    "Range", min_val, max_val, (min_val, max_val)
                )
                if st.button("Apply range filter"):
                    st.session_state.dashboard_filters.append(
                        {"column": filter_col, "op": "range", "value": (lo, hi)}
                    )
            else:
                selected = st.multiselect("Values", unique_vals)
                if st.button("Apply value filter"):
                    st.session_state.dashboard_filters.append(
                        {"column": filter_col, "op": "in", "value": selected}
                    )

    st.sidebar.markdown("---")


def data_preview_tab():
    st.header("Data Preview")
    if st.session_state.df is None:
        st.info("Upload a dataset to start analyzing.")
        return

    df = st.session_state.df
    overview = dataset_overview(df)

    st.metric("Rows", overview.get("rows", 0))
    st.metric("Columns", overview.get("columns", 0))

    st.subheader("Data Quality")
    cols = st.columns(3)
    cols[0].metric("Missing cells", overview.get("missing_values", 0))
    cols[1].metric("Duplicate rows", overview.get("duplicate_rows", 0))
    cols[2].metric(
        "Numeric columns", overview.get("numeric_columns", 0)
    )

    st.subheader("Schema")
    schema = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum()})
    st.dataframe(schema, width="stretch")

    st.subheader("Preview")
    st.dataframe(df.head(10), width="stretch")


def ai_suggestions_tab():
    st.header("AI-Powered Cleaning Suggestions")
    if st.session_state.df is None:
        st.info("Upload a dataset to get AI cleaning suggestions.")
        return

    refresh_suggestions()
    suggestions = st.session_state.ai_suggestions

    if not suggestions:
        st.success("No suggestions available — dataset looks clean!")
        return

    for idx, suggestion in enumerate(suggestions):
        with st.expander(f"Suggestion {idx + 1}: {suggestion.get('suggestion')}"):
            st.write(suggestion.get("suggestion"))
            if st.button(f"Apply suggestion {idx + 1}"):
                updated = apply_suggestion(
                    st.session_state.df, suggestion
                )
                commit_dataframe(updated, suggestion.get("suggestion", "Smart suggestion"))
                st.success("Suggestion applied."
                            "Review data preview to see changes.")

    if st.button("Apply all suggestions"):
        updated = st.session_state.df
        for suggestion in suggestions:
            updated = apply_suggestion(updated, suggestion)
        commit_dataframe(updated, f"Apply {len(suggestions)} smart suggestions")
        st.success("All suggestions applied.")


def natural_language_cleaning_tab():
    st.header("Natural-Language Cleaning")
    if st.session_state.df is None:
        st.info("Upload a dataset to describe cleaning actions in plain English.")
        return

    st.caption(
        "The app converts your request into allow-listed Pandas operations. "
        "Nothing is applied until you review and approve the preview."
    )
    instruction = st.text_area(
        "Describe the cleaning you want",
        placeholder=(
            "Example: Remove duplicates, fill missing values in Age with the median, "
            "standardize categories in City, and normalize Income."
        ),
        key="nl_cleaning_instruction",
    )
    engine_label = st.selectbox(
        "Instruction engine",
        ["Automatic", "Rules only", "Local Ollama / Qwen", "Gemini", "Groq"],
        key="nl_cleaning_engine",
    )
    provider_map = {
        "Automatic": "auto",
        "Rules only": "rules",
        "Local Ollama / Qwen": "ollama",
        "Gemini": "gemini",
        "Groq": "groq",
    }

    if st.button("Build safe cleaning plan", key="build_nl_plan"):
        if not instruction.strip():
            st.warning("Enter a cleaning instruction first.")
        else:
            try:
                plan = parse_instruction(
                    instruction, st.session_state.df.columns.astype(str).tolist()
                )
                provider = "Built-in safe parser"
                warning = None
                selected_provider = provider_map[engine_label]
                if not plan and selected_provider != "rules":
                    plan, provider, warning = generate_cleaning_plan(
                        instruction,
                        st.session_state.df.columns.astype(str).tolist(),
                        {
                            str(column): str(dtype)
                            for column, dtype in st.session_state.df.dtypes.items()
                        },
                        selected_provider,
                    )
                if not plan:
                    st.session_state.nl_cleaning_plan = []
                    st.session_state.nl_cleaning_preview = None
                    st.warning(
                        warning
                        or "The request could not be mapped to supported safe actions."
                    )
                else:
                    preview = apply_plan(st.session_state.df, plan)
                    st.session_state.nl_cleaning_plan = plan
                    st.session_state.nl_cleaning_preview = preview
                    st.session_state.nl_cleaning_provider = provider
            except Exception as exc:
                st.session_state.nl_cleaning_plan = []
                st.session_state.nl_cleaning_preview = None
                st.error(f"Could not build the cleaning plan: {exc}")

    plan = st.session_state.nl_cleaning_plan
    preview = st.session_state.nl_cleaning_preview
    if plan and preview is not None:
        st.subheader("Review plan")
        st.caption(
            f"Interpreter: {st.session_state.get('nl_cleaning_provider', 'Safe parser')}"
        )
        for line in plan_summary(plan):
            st.write(line)

        current = st.session_state.df
        metrics = st.columns(4)
        metrics[0].metric("Rows before", len(current))
        metrics[1].metric("Rows after", len(preview), len(preview) - len(current))
        metrics[2].metric("Columns before", len(current.columns))
        metrics[3].metric(
            "Columns after",
            len(preview.columns),
            len(preview.columns) - len(current.columns),
        )
        st.dataframe(preview.head(20), width="stretch")
        approve, discard = st.columns(2)
        if approve.button(
            "Approve and apply plan", type="primary", key="approve_nl_plan"
        ):
            commit_dataframe(preview, f"Natural-language plan: {instruction[:120]}")
            st.session_state.nl_cleaning_plan = []
            st.session_state.nl_cleaning_preview = None
            st.success("The reviewed cleaning plan was applied.")
            st.rerun()
        if discard.button("Discard plan", key="discard_nl_plan"):
            st.session_state.nl_cleaning_plan = []
            st.session_state.nl_cleaning_preview = None
            st.rerun()


def data_cleaning_tab():
    st.header("Data Cleaning")
    st.write(
        "Use the sidebar tools for one-click cleaning, or use the options below for manual operations."
    )

    if st.session_state.df is None:
        st.info("Upload a dataset to start cleaning.")
        return

    st.subheader("Current dataset snapshot")
    st.dataframe(st.session_state.df.head(5), width="stretch")

    if st.button("Reset dataset to original"):
        st.session_state.df = st.session_state.original_df.copy()
        st.session_state.undo_stack = []
        st.session_state.transformation_history = []
        st.session_state.dashboard_charts = []
        st.session_state.generated_data_card = None
        st.session_state.automl_result = None
        st.session_state.dataset_splits = None
        refresh_suggestions()
        st.success("Dataset reset to original.")

    if st.session_state.transformation_history:
        st.subheader("Transformation History")
        st.dataframe(
            pd.DataFrame(st.session_state.transformation_history),
            width="stretch",
            hide_index=True,
        )


def profiling_tab():
    st.header("Dataset Profiling")
    if st.session_state.df is None:
        st.info("Upload a dataset to profile.")
        return

    df = st.session_state.df

    overview = dataset_overview(df)
    st.subheader("Overview")
    st.write(overview)

    st.subheader("Missing Value Heatmap")
    heatmap = missing_heatmap(df)
    if heatmap is not None:
        st.plotly_chart(heatmap, width="stretch")

    st.subheader("Correlation Heatmap")
    corr_fig = correlation_heatmap(df)
    if corr_fig is not None:
        st.plotly_chart(corr_fig, width="stretch")

    st.subheader("Column Statistics")
    stats = column_statistics(df)
    if not stats.empty:
        st.dataframe(stats, width="stretch")

    st.subheader("Numeric Distributions")
    dist_plots = numeric_distribution_plots(df)
    for name, fig in dist_plots.items():
            st.plotly_chart(fig, width="stretch")

def visualization_tab():
    st.header("Visualization")
    if st.session_state.df is None:
        st.info("Upload a dataset to visualize.")
        return

    df = st.session_state.df
    all_cols = df.columns.tolist()

    chart_type = st.selectbox(
        "Chart type",
        [
            "Auto",
            "Histogram",
            "Bar",
            "Line",
            "Scatter",
            "Box",
            "Heatmap (correlation)",
            "Distribution",
        ],
    )

    if chart_type == "Auto":
        suggestions = suggest_chart_types(df)
        st.info(f"Suggested charts: {', '.join(suggestions[:3])}")

    x_col = st.selectbox("X axis", [None] + all_cols, index=0)
    y_col = st.selectbox("Y axis", [None] + all_cols, index=0)

    if st.button("Create chart"):
        try:
            fig = create_plot(df, chart_type, x_col, y_col)
            st.session_state.last_plot = fig
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.error(f"Could not create chart: {e}")

    if st.session_state.last_plot is not None:
        try:
            img_bytes = plot_to_image_bytes(st.session_state.last_plot)
            st.download_button(
                "Download chart as PNG",
                data=img_bytes,
                file_name="chart.png",
                mime="image/png",
            )
        except Exception as e:
            st.error(f"Export error: {e}")


def dashboard_tab():
    st.header("Interactive Dashboard")

    if st.session_state.df is None:
        st.info("Upload a dataset to build dashboards.")
        return

    df = st.session_state.df
    cols = df.columns.tolist()

    st.sidebar.header("Dashboard Builder")
    chart_type = st.sidebar.selectbox(
        "Chart type",
        ["Bar", "Line", "Scatter", "Histogram", "Box", "Heatmap"],
        key="dash_chart_type",
    )
    x = st.sidebar.selectbox("X column", [None] + cols, key="dash_x")
    y = st.sidebar.selectbox("Y column", [None] + cols, key="dash_y")
    color = st.sidebar.selectbox("Color", [None] + cols, key="dash_color")

    if st.sidebar.button("Add chart"):
        filtered = apply_filters(df, st.session_state.dashboard_filters)
        fig = create_dashboard_chart(
            filtered, chart_type, x, y, color, title=f"{chart_type}: {y or x}"
        )
        if fig is not None:
            st.session_state.dashboard_charts.append(
                {"type": chart_type, "x": x, "y": y, "color": color}
            )

    if st.session_state.dashboard_filters:
        st.write("### Active Filters")
        for f in st.session_state.dashboard_filters:
            st.write(f"- {f}")
        if st.button("Clear dashboard filters"):
            st.session_state.dashboard_filters = []
            st.rerun()

    filtered = apply_filters(df, st.session_state.dashboard_filters)

    st.write("### Dashboard")
    if not st.session_state.dashboard_charts:
        st.info("Add charts using the sidebar to build your dashboard.")
    else:
        for idx, entry in enumerate(st.session_state.dashboard_charts, start=1):
            st.subheader(f"Chart {idx}: {entry.get('type')}")
            figure = create_dashboard_chart(
                filtered,
                entry.get("type"),
                entry.get("x"),
                entry.get("y"),
                entry.get("color"),
                title=f"{entry.get('type')}: {entry.get('y') or entry.get('x')}",
            )
            if figure is not None:
                st.plotly_chart(figure, width="stretch")
            if st.button(f"Remove chart {idx}"):
                st.session_state.dashboard_charts.pop(idx - 1)
                st.rerun()


def statistics_tab():
    st.header("Statistics")
    if st.session_state.df is None:
        st.info("Upload a dataset to run statistical analysis.")
        return

    df = st.session_state.df

    st.subheader("Descriptive Statistics")
    desc = descriptive_statistics(df)
    st.dataframe(desc, width="stretch")

    st.subheader("Correlation & Covariance")
    corr = df.select_dtypes(include=["number"]).corr()
    cov = covariance_matrix(df)
    st.write("Correlation")
    st.dataframe(corr, width="stretch")
    st.write("Covariance")
    st.dataframe(cov, width="stretch")

    st.subheader("Regression")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) >= 2:
        reg_x = st.selectbox("Regression X (predictor)", [None] + numeric_cols)
        reg_y = st.selectbox("Regression Y (target)", [None] + numeric_cols)
        if reg_x and reg_y and st.button("Run linear regression"):
            result = linear_regression(df, reg_x, reg_y)
            st.write("**Linear regression results**")
            st.json(result)
    else:
        st.info("Need at least 2 numeric columns for regression.")

    st.subheader("Hypothesis testing")
    numeric_test_cols = df.select_dtypes(include=["number"]).columns.tolist()
    tcol1 = st.selectbox("Sample 1 column", [None] + numeric_test_cols, key="t1")
    tcol2 = st.selectbox("Sample 2 column", [None] + numeric_test_cols, key="t2")
    if tcol1 and tcol2:
        alpha = st.number_input(
            "Significance level (alpha)", min_value=0.001, max_value=0.2, value=0.05, step=0.01
        )
        if st.button("Run t-test"):
            try:
                res = t_test_independent(df, tcol1, tcol2)
                st.write(res)
                st.write(
                    "✅ Reject null hypothesis"
                    if res["p_value"] < alpha
                    else "✅ Fail to reject null hypothesis"
                )
            except Exception as e:
                st.error(f"T-test error: {e}")

    st.subheader("Advanced statistical tests")
    advanced_test = st.selectbox(
        "Test",
        ["Normality", "Mann-Whitney U", "One-way ANOVA", "Chi-square independence"],
        key="advanced_test",
    )
    categorical_cols = [
        column
        for column in df.columns
        if not pd.api.types.is_numeric_dtype(df[column])
        or df[column].nunique(dropna=True) <= 20
    ]
    try:
        if advanced_test == "Normality":
            value_col = st.selectbox(
                "Numeric column", [None] + numeric_cols, key="normality_value"
            )
            if value_col and st.button("Run normality test", key="run_normality"):
                st.json(normality_test(df[value_col]))
        elif advanced_test in {"Mann-Whitney U", "One-way ANOVA"}:
            value_col = st.selectbox(
                "Numeric outcome", [None] + numeric_cols, key="group_test_value"
            )
            group_col = st.selectbox(
                "Group column", [None] + categorical_cols, key="group_test_group"
            )
            if (
                value_col
                and group_col
                and st.button("Run group comparison", key="run_group_test")
            ):
                result = (
                    mann_whitney_test(df, value_col, group_col)
                    if advanced_test == "Mann-Whitney U"
                    else one_way_anova(df, value_col, group_col)
                )
                st.json(result)
        else:
            first = st.selectbox(
                "First categorical column",
                [None] + categorical_cols,
                key="chi_first",
            )
            second = st.selectbox(
                "Second categorical column",
                [None] + categorical_cols,
                key="chi_second",
            )
            if (
                first
                and second
                and first != second
                and st.button("Run chi-square test", key="run_chi")
            ):
                st.json(chi_square_test(df, first, second))
    except Exception as exc:
        st.error(f"Statistical test error: {exc}")


def report_generator_tab():
    st.header("Report Generator")
    if st.session_state.df is None:
        st.info("Upload a dataset to generate reports.")
        return

    df = st.session_state.df
    overview = dataset_overview(df)
    stats = column_statistics(df)
    suggestions = st.session_state.ai_suggestions

    charts = []
    if st.session_state.last_plot is not None:
        charts.append(
            {
                "title": "Latest chart",
                "image_base64": build_chart_image_base64(st.session_state.last_plot),
            }
        )

    report_md = generate_markdown_report(df, overview, stats, suggestions, charts)

    st.subheader("Preview")
    st.markdown(report_md)

    st.subheader("Download")
    st.download_button(
        "Download Markdown", data=report_md, file_name="report.md", mime="text/markdown"
    )
    html_report = markdown_to_html(report_md)
    st.download_button(
        "Download HTML",
        data=html_report,
        file_name="report.html",
        mime="text/html",
    )

    try:
        pdf_bytes = generate_pdf_report(report_md, charts)
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="report.pdf",
            mime="application/pdf",
        )
    except Exception as exc:
        st.warning(f"PDF generation is unavailable: {exc}")


def ml_readiness_tab():
    st.header("ML Readiness")
    if st.session_state.df is None:
        st.info("Upload a dataset to assess ML readiness.")
        return

    df = st.session_state.df
    target = st.selectbox("Optional target column", [None] + df.columns.tolist())
    report = analyze_ml_readiness(df, target)
    st.metric("ML-readiness score", f"{report['score']}/100")

    left, right = st.columns(2)
    with left:
        st.subheader("Issues")
        if report["issues"]:
            for issue in report["issues"]:
                st.warning(issue)
        else:
            st.success("No major automated readiness issues detected.")
    with right:
        st.subheader("Recommendations")
        for recommendation in report["recommendations"]:
            st.info(recommendation)

    st.subheader("Data Dictionary")
    st.dataframe(build_data_dictionary(df), width="stretch", hide_index=True)


def automl_tab():
    st.header("AutoML Workbench")
    if st.session_state.df is None:
        st.info("Upload a dataset to prepare splits and compare baseline models.")
        return

    df = st.session_state.df
    if len(df.columns) < 2:
        st.warning("AutoML requires at least one feature and one target column.")
        return

    target = st.selectbox(
        "Target column", df.columns.tolist(), key="automl_target_select"
    )
    problem_label = st.selectbox(
        "Problem type",
        ["Automatic", "Classification", "Regression"],
        key="automl_problem_type",
    )
    problem_type = problem_label.lower()
    if problem_type == "automatic":
        problem_type = "auto"
    readiness = analyze_ml_readiness(df, target)
    suggested_exclusions = list(
        dict.fromkeys(
            readiness.get("potential_identifiers", [])
            + list(readiness.get("sensitive_columns", {}).keys())
        )
    )
    excluded_features = st.multiselect(
        "Exclude features from modelling",
        [column for column in df.columns if column != target],
        default=[
            column for column in suggested_exclusions if column != target
        ],
        help="Potential identifiers and sensitive fields are excluded by default.",
        key="automl_excluded_features",
    )
    random_state = int(
        st.number_input(
            "Random seed",
            min_value=0,
            max_value=100000,
            value=42,
            step=1,
            key="automl_seed",
        )
    )

    split_col, run_col = st.columns(2)
    if split_col.button("Create train/validation/test splits", key="create_splits"):
        try:
            splits = split_dataset(
                df,
                target,
                problem_type=problem_type,
                random_state=random_state,
            )
            st.session_state.dataset_splits = splits
            st.success(
                f"Created {len(splits['train'])} training, "
                f"{len(splits['validation'])} validation and "
                f"{len(splits['test'])} test rows."
            )
        except Exception as exc:
            st.error(f"Could not create splits: {exc}")

    if run_col.button("Compare baseline models", type="primary", key="run_automl"):
        try:
            with st.spinner("Training and comparing compact baseline models..."):
                result = run_automl(
                    df,
                    target,
                    problem_type=problem_type,
                    random_state=random_state,
                    exclude_features=excluded_features,
                )
            st.session_state.automl_result = result
            st.session_state.dataset_splits = result["splits"]
            st.success(f"Best validation model: {result['best_model_name']}")
        except Exception as exc:
            st.session_state.automl_result = None
            st.error(f"AutoML could not complete: {exc}")

    splits = st.session_state.dataset_splits
    if splits:
        st.subheader("Reproducible splits")
        split_metrics = st.columns(3)
        for column, name in zip(split_metrics, ("train", "validation", "test")):
            column.metric(name.title(), len(splits[name]))
            column.download_button(
                f"Download {name}",
                data=dataframe_to_csv(splits[name]),
                file_name=f"{name}.csv",
                mime="text/csv",
                key=f"download_{name}_split",
            )

    result = st.session_state.automl_result
    if result:
        st.subheader("Model comparison")
        st.caption(
            f"Problem: {result['problem_type']} · Target: {result['target']} · "
            f"Rows used: {result['sampled_rows']:,}"
        )
        if result["excluded_features"]:
            st.caption(
                "Excluded features: " + ", ".join(result["excluded_features"])
            )
        st.dataframe(result["results"], width="stretch", hide_index=True)
        left, right = st.columns(2)
        left.write("**Best validation metrics**")
        left.json(result["validation_metrics"])
        right.write("**Held-out test metrics**")
        right.json(result["test_metrics"])

        if not result["feature_importance"].empty:
            st.subheader("Most influential features")
            importance = result["feature_importance"].set_index("feature")
            st.bar_chart(importance)

        st.download_button(
            "Download trained preprocessing and model pipeline",
            data=result["model_bytes"],
            file_name="best_model.joblib",
            mime="application/octet-stream",
            key="download_automl_model",
        )
        st.caption(
            "AutoML results are baselines, not proof of real-world validity. "
            "Review leakage, fairness, sampling and domain assumptions."
        )


def export_tab():
    st.header("AI Export Studio")

    if st.session_state.df is None:
        st.info("Upload and clean a dataset before exporting.")
        return

    df = st.session_state.df
    st.subheader("1. Generate professional dataset documentation")
    dataset_title = st.text_input("Dataset title", value="Smart Dataset")
    provider_label = st.selectbox(
        "Writing engine",
        [
            "Automatic (local AI → configured cloud AI → professional fallback)",
            "Local Ollama / Qwen",
            "Gemini",
            "Groq",
            "Built-in professional writer",
        ],
    )
    provider_map = {
        "Automatic (local AI → configured cloud AI → professional fallback)": "auto",
        "Local Ollama / Qwen": "ollama",
        "Gemini": "gemini",
        "Groq": "groq",
        "Built-in professional writer": "template",
    }
    automl_result = st.session_state.automl_result
    quality_target = automl_result["target"] if automl_result else None
    quality = analyze_ml_readiness(df, quality_target)

    if st.button("Generate export documentation", type="primary"):
        with st.spinner("Writing the dataset documentation..."):
            card, used_provider, warning = generate_data_card(
                df, quality, dataset_title, provider_map[provider_label]
            )
        st.session_state.generated_data_card = card
        st.session_state.data_card_provider = used_provider
        if warning:
            st.warning(
                "An AI provider was unavailable, so the reliable fallback was used. "
                f"Details: {warning}"
            )
        st.success(f"Documentation generated with {used_provider}.")

    if st.session_state.generated_data_card:
        st.caption(f"Writer: {st.session_state.data_card_provider}")
        st.markdown(st.session_state.generated_data_card)
        st.download_button(
            "Download data card",
            data=st.session_state.generated_data_card,
            file_name="README.md",
            mime="text/markdown",
        )

        automl_summary = None
        model_bytes = None
        if automl_result:
            automl_summary = {
                "problem_type": automl_result["problem_type"],
                "target": automl_result["target"],
                "best_model": automl_result["best_model_name"],
                "excluded_features": automl_result["excluded_features"],
                "validation_metrics": automl_result["validation_metrics"],
                "test_metrics": automl_result["test_metrics"],
                "comparison": automl_result["results"].to_dict(orient="records"),
            }
            model_bytes = automl_result["model_bytes"]
        package = build_export_package(
            df,
            st.session_state.generated_data_card,
            build_data_dictionary(df),
            st.session_state.transformation_history,
            quality,
            st.session_state.original_df,
            st.session_state.dataset_splits,
            automl_summary,
            model_bytes,
        )
        st.download_button(
            "Download complete publishing package",
            data=package,
            file_name="smart_dataset_export.zip",
            mime="application/zip",
            type="primary",
        )

    st.subheader("2. Individual data exports")
    csv_bytes = dataframe_to_csv(st.session_state.df)
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )
    if len(df) <= 1_048_575:
        st.download_button(
            "Download Excel",
            data=dataframe_to_excel(df),
            file_name="cleaned_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info(
            "Excel export is unavailable above 1,048,575 data rows; use CSV, "
            "JSON, or the Parquet file in the publishing package."
        )
    st.download_button(
        "Download JSON",
        data=dataframe_to_json(df),
        file_name="cleaned_dataset.json",
        mime="application/json",
    )
    st.download_button(
        "Download Parquet",
        data=dataframe_to_parquet(df),
        file_name="cleaned_dataset.parquet",
        mime="application/octet-stream",
    )

    st.subheader("3. Statistics")
    stats_csv = statistics_to_csv(descriptive_statistics(st.session_state.df))
    st.download_button(
        "Download stats CSV",
        data=stats_csv,
        file_name="statistics_summary.csv",
        mime="text/csv",
    )


def main():
    init_state()

    # Use path relative to the app file for assets (works regardless of working directory)
    app_dir = Path(__file__).resolve().parent
    logo_path = app_dir / "assets" / "logo.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=220)
    else:
        st.sidebar.write("**Smart Data Analyzer Pro**")

    st.title("Smart Data Analyzer Pro")

    sidebar_upload()
    sidebar_cleaning_tools()

    tabs = st.tabs(
        ["Understand", "Clean", "Visualize", "Analyze", "AutoML", "Publish"]
    )

    with tabs[0]:
        preview_tab, profile_tab = st.tabs(["Preview", "Profiling"])
        with preview_tab:
            data_preview_tab()
        with profile_tab:
            profiling_tab()

    with tabs[1]:
        suggestions_tab, language_tab, history_tab = st.tabs(
            ["Smart Suggestions", "Natural Language", "History"]
        )
        with suggestions_tab:
            ai_suggestions_tab()
        with language_tab:
            natural_language_cleaning_tab()
        with history_tab:
            data_cleaning_tab()

    with tabs[2]:
        chart_tab, dashboard_workspace = st.tabs(["Charts", "Dashboard"])
        with chart_tab:
            visualization_tab()
        with dashboard_workspace:
            dashboard_tab()

    with tabs[3]:
        statistics_workspace, readiness_workspace = st.tabs(
            ["Statistics", "ML Readiness"]
        )
        with statistics_workspace:
            statistics_tab()
        with readiness_workspace:
            ml_readiness_tab()

    with tabs[4]:
        automl_tab()

    with tabs[5]:
        report_tab, export_workspace = st.tabs(["Reports", "AI Export Studio"])
        with report_tab:
            report_generator_tab()
        with export_workspace:
            export_tab()

    st.markdown("---")
    st.caption(
        "Privacy-aware data cleaning, statistical analysis, AutoML, and publishing."
    )


if __name__ == "__main__":
    main()
