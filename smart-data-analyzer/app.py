import streamlit as st
import pandas as pd

from modules.ai_cleaning import analyze_dataset, apply_suggestion
from modules.data_loader import load_dataset_bytes
from modules.data_cleaning import (
    clean_string_column,
    convert_column_type,
    drop_columns,
    extract_date_parts,
    fill_missing,
    log_transform,
    normalize_column,
    parse_dates,
    remove_duplicates,
    remove_missing,
    replace_values,
    sort_dataframe,
    standardize_categories,
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
    dataframe_to_csv,
    plot_to_image_bytes,
    statistics_to_csv,
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
        "Upload CSV or Excel", type=["csv", "xlsx"], accept_multiple_files=False
    )

    if uploaded_file is not None:
        df = load_dataset_from_uploader(uploaded_file)
        if df is not None:
            st.session_state.df = df.copy()
            st.session_state.original_df = df.copy()
            refresh_suggestions()
            st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    if st.session_state.df is not None:
        if st.sidebar.button("Reset to original"):
            st.session_state.df = st.session_state.original_df.copy()
            refresh_suggestions()
            st.sidebar.success("Reset to original dataset.")

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
            st.session_state.df = remove_missing(df)
            refresh_suggestions()
            st.success("Removed rows with missing values.")

        if st.button("Remove duplicate rows"):
            st.session_state.df = remove_duplicates(df)
            refresh_suggestions()
            st.success("Removed duplicate rows.")

    with st.sidebar.expander("Missing Value Handling"):
        fill_method = st.selectbox("Fill missing using", ["Mean", "Median", "Mode", "Custom"], index=0)
        if fill_method == "Custom":
            custom_value = st.text_input("Custom fill value", value="")
        else:
            custom_value = None

        selected_cols = st.multiselect("Columns", cols, default=cols)
        if st.button("Apply fill"):
            st.session_state.df = fill_missing(
                st.session_state.df,
                columns=selected_cols,
                method=fill_method.lower(),
                custom_value=custom_value,
            )
            refresh_suggestions()
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
            st.session_state.df = replace_values(st.session_state.df, mapping)
            refresh_suggestions()
            st.success("Applied value replacements.")

    with st.sidebar.expander("String Cleaning"):
        str_col = st.selectbox("Column", [None] + cols, key="clean_str_col")
        if str_col:
            trim = st.checkbox("Trim whitespace", value=True)
            remove_special = st.checkbox("Remove special characters", value=True)
            case = st.selectbox("Case", ["none", "lower", "upper", "title"], index=0)
            if st.button("Clean strings"):
                st.session_state.df = clean_string_column(
                    st.session_state.df,
                    str_col,
                    trim=trim,
                    remove_special=remove_special,
                    case=case if case != "none" else None,
                )
                refresh_suggestions()
                st.success(f"Cleaned strings in '{str_col}'.")

    with st.sidebar.expander("Numeric Transformations"):
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        num_col = st.selectbox("Numeric column", [None] + num_cols, key="num_trans_col")
        if num_col:
            transform = st.selectbox("Transformation", ["Log", "Normalize (0-1)", "Standardize (z)"])
            if st.button("Apply transformation"):
                if transform == "Log":
                    st.session_state.df = log_transform(st.session_state.df, num_col)
                elif transform == "Normalize (0-1)":
                    st.session_state.df = normalize_column(st.session_state.df, num_col, method="minmax")
                else:
                    st.session_state.df = normalize_column(st.session_state.df, num_col, method="zscore")
                refresh_suggestions()
                st.success(f"Applied {transform} to {num_col}.")

    with st.sidebar.expander("Date Parsing"):
        date_col = st.selectbox("Date column", [None] + cols, key="date_col")
        date_format = st.text_input("Format (optional)", value="")
        if date_col and st.button("Parse dates"):
            st.session_state.df = parse_dates(
                st.session_state.df, date_col, format=date_format or None
            )
            st.session_state.df = extract_date_parts(st.session_state.df, date_col)
            refresh_suggestions()
            st.success(f"Parsed dates in '{date_col}'.")

    with st.sidebar.expander("Type Conversion"):
        col_type = st.selectbox("Column", [None] + cols, key="convert_col")
        if col_type:
            target_type = st.selectbox(
                "Type", ["int", "float", "str", "datetime", "category"], key="convert_type"
            )
            if st.button("Convert type"):
                st.session_state.df = convert_column_type(
                    st.session_state.df, col_type, target_type
                )
                refresh_suggestions()
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
    st.dataframe(schema, use_container_width=True)

    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)


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
                st.session_state.df = apply_suggestion(
                    st.session_state.df, suggestion
                )
                refresh_suggestions()
                st.success("Suggestion applied."
                            "Review data preview to see changes.")

    if st.button("Apply all suggestions"):
        for suggestion in suggestions:
            st.session_state.df = apply_suggestion(
                st.session_state.df, suggestion
            )
        refresh_suggestions()
        st.success("All suggestions applied.")


def data_cleaning_tab():
    st.header("Data Cleaning")
    st.write(
        "Use the sidebar tools for one-click cleaning, or use the options below for manual operations."
    )

    if st.session_state.df is None:
        st.info("Upload a dataset to start cleaning.")
        return

    st.subheader("Current dataset snapshot")
    st.dataframe(st.session_state.df.head(5), use_container_width=True)

    if st.button("Reset dataset to original"):
        st.session_state.df = st.session_state.original_df.copy()
        refresh_suggestions()
        st.success("Dataset reset to original.")


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
        st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr_fig = correlation_heatmap(df)
    if corr_fig is not None:
        st.plotly_chart(corr_fig, use_container_width=True)

    st.subheader("Column Statistics")
    stats = column_statistics(df)
    if not stats.empty:
        st.dataframe(stats, use_container_width=True)

    st.subheader("Numeric Distributions")
    dist_plots = numeric_distribution_plots(df)
    for name, fig in dist_plots.items():
        st.plotly_chart(fig, use_container_width=True)


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
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not create chart: {e}")

    if st.session_state.last_plot is not None:
        if st.button("Download chart as PNG"):
            try:
                img_bytes = plot_to_image_bytes(st.session_state.last_plot)
                st.download_button(
                    "Download PNG",
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
        fig = create_dashboard_chart(df, chart_type, x, y, color, title=f"{chart_type}: {y or x}")
        if fig is not None:
            st.session_state.dashboard_charts.append(
                {"type": chart_type, "x": x, "y": y, "color": color, "figure": fig}
            )

    if st.session_state.dashboard_filters:
        st.write("### Active Filters")
        for f in st.session_state.dashboard_filters:
            st.write(f"- {f}")

    filtered = apply_filters(df, st.session_state.dashboard_filters)

    st.write("### Dashboard")
    if not st.session_state.dashboard_charts:
        st.info("Add charts using the sidebar to build your dashboard.")
    else:
        for idx, entry in enumerate(st.session_state.dashboard_charts, start=1):
            st.subheader(f"Chart {idx}: {entry.get('type')}")
            st.plotly_chart(entry.get("figure"), use_container_width=True)
            if st.button(f"Remove chart {idx}"):
                st.session_state.dashboard_charts.pop(idx - 1)
                st.experimental_rerun()


def statistics_tab():
    st.header("Statistics")
    if st.session_state.df is None:
        st.info("Upload a dataset to run statistical analysis.")
        return

    df = st.session_state.df

    st.subheader("Descriptive Statistics")
    desc = descriptive_statistics(df)
    st.dataframe(desc, use_container_width=True)

    st.subheader("Correlation & Covariance")
    corr = df.select_dtypes(include=["number"]).corr()
    cov = covariance_matrix(df)
    st.write("Correlation")
    st.dataframe(corr, use_container_width=True)
    st.write("Covariance")
    st.dataframe(cov, use_container_width=True)

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
    cols = df.columns.tolist()
    tcol1 = st.selectbox("Sample 1 column", [None] + cols, key="t1")
    tcol2 = st.selectbox("Sample 2 column", [None] + cols, key="t2")
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
    st.markdown(markdown_to_html(report_md), unsafe_allow_html=True)

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

    if st.button("Download PDF"):
        pdf_bytes = generate_pdf_report(report_md, charts)
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="report.pdf",
            mime="application/pdf",
        )


def export_tab():
    st.header("Export")

    if st.session_state.df is None:
        st.info("Upload and clean a dataset before exporting.")
        return

    st.subheader("Export Cleaned Dataset")
    csv_bytes = dataframe_to_csv(st.session_state.df)
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )

    st.subheader("Export Statistics")
    stats_csv = statistics_to_csv(descriptive_statistics(st.session_state.df))
    st.download_button(
        "Download stats CSV",
        data=stats_csv,
        file_name="statistics_summary.csv",
        mime="text/csv",
    )


def main():
    init_state()
    st.sidebar.image("assets/logo.png", use_column_width=True)
    st.title("Smart Data Analyzer Pro")

    sidebar_upload()
    sidebar_cleaning_tools()

    tabs = st.tabs(
        [
            "Data Preview",
            "AI Suggestions",
            "Data Cleaning",
            "Dataset Profiling",
            "Visualization",
            "Interactive Dashboard",
            "Statistics",
            "Report Generator",
            "Export",
        ]
    )

    with tabs[0]:
        data_preview_tab()
    with tabs[1]:
        ai_suggestions_tab()
    with tabs[2]:
        data_cleaning_tab()
    with tabs[3]:
        profiling_tab()
    with tabs[4]:
        visualization_tab()
    with tabs[5]:
        dashboard_tab()
    with tabs[6]:
        statistics_tab()
    with tabs[7]:
        report_generator_tab()
    with tabs[8]:
        export_tab()

    st.markdown("---")
    st.caption(
        "Built with Streamlit, pandas, Plotly, SciPy, scikit-learn, and optimized for interactive data analysis."
    )


if __name__ == "__main__":
    main()
