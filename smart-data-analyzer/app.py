import streamlit as st
import pandas as pd
import numpy as np

from modules.data_loader import load_dataset
from modules.data_cleaning import (
    remove_missing,
    fill_missing,
    remove_duplicates,
    drop_columns,
    rename_columns,
    convert_column_type,
    filter_rows,
    sort_dataframe,
)
from modules.visualization import create_plot, suggest_chart_types
from modules.statistics_tools import (
    descriptive_statistics,
    correlation_matrix,
    covariance_matrix,
    linear_regression,
    detect_outliers_zscore,
    detect_outliers_iqr,
    t_test_independent,
)
from modules.export_tools import (
    dataframe_to_csv,
    plot_to_image_bytes,
    statistics_to_csv,
)


st.set_page_config(
    page_title="Smart Data Analyzer",
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


def sidebar_upload():
    st.sidebar.title("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        df = load_dataset(uploaded_file)
        if df is not None:
            st.session_state.df = df.copy()
            st.session_state.original_df = df.copy()
            st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    if st.session_state.df is not None:
        if st.sidebar.button("Reset to original"):
            st.session_state.df = st.session_state.original_df.copy()
            st.sidebar.success("Reset to original dataset.")

    st.sidebar.markdown("---")


def sidebar_cleaning_tools():
    st.sidebar.header("Cleaning Tools")
    if st.session_state.df is None:
        st.sidebar.info("Upload a dataset to enable cleaning tools.")
        return

    df = st.session_state.df
    columns = list(df.columns)

    with st.sidebar.expander("Missing Value Handling"):
        if st.button("Remove rows with missing values"):
            st.session_state.df = remove_missing(df)
            st.success("Removed rows with missing values.")

        fill_method = st.selectbox(
            "Fill missing values using", ["Mean", "Median", "Mode", "Custom"], index=0
        )
        if fill_method == "Custom":
            custom_value = st.text_input("Custom fill value", value="")
        else:
            custom_value = None

        selected_cols = st.multiselect("Columns", columns, default=columns)
        if st.button("Apply fill"):
            st.session_state.df = fill_missing(
                st.session_state.df,
                columns=selected_cols,
                method=fill_method.lower(),
                custom_value=custom_value,
            )
            st.success("Filled missing values.")

    with st.sidebar.expander("Duplicates"):
        if st.button("Remove duplicate rows"):
            st.session_state.df = remove_duplicates(st.session_state.df)
            st.success("Removed duplicate rows.")

    with st.sidebar.expander("Columns"):
        drop_cols = st.multiselect("Drop columns", columns)
        if st.button("Drop selected columns"):
            st.session_state.df = drop_columns(st.session_state.df, drop_cols)
            st.success("Dropped selected columns.")

        rename_from = st.selectbox("Rename column", [None] + columns)
        if rename_from:
            rename_to = st.text_input("New name", value=rename_from)
            if st.button("Apply rename"):
                st.session_state.df = rename_columns(
                    st.session_state.df, {rename_from: rename_to}
                )
                st.success(f"Renamed {rename_from} -> {rename_to}.")

        st.markdown("---")
        st.write("**Convert column types**")
        col_type = st.selectbox("Column", [None] + columns, key="convert_col")
        if col_type:
            target_type = st.selectbox(
                "Type", ["int", "float", "str", "datetime", "category"], key="convert_type"
            )
            if st.button("Convert type"):
                st.session_state.df = convert_column_type(
                    st.session_state.df, col_type, target_type
                )
                st.success(f"Converted {col_type} to {target_type}.")

    with st.sidebar.expander("Filter & Sort"):
        st.write("Filter rows by condition")
        filter_col = st.selectbox(
            "Column", [None] + columns, key="filter_col"
        )
        if filter_col:
            filter_op = st.selectbox(
                "Operator", ["==", "!=", ">", "<", ">=", "<=", "contains"]
            )
            filter_value = st.text_input("Value")
            if st.button("Apply filter"):
                try:
                    st.session_state.df = filter_rows(
                        st.session_state.df,
                        column=filter_col,
                        op=filter_op,
                        value=filter_value,
                    )
                    st.success("Filter applied.")
                except Exception as e:
                    st.error(f"Filter error: {e}")

        st.write("Sort dataset")
        sort_col = st.selectbox("Sort by", [None] + columns, key="sort_col")
        if sort_col:
            ascending = st.checkbox("Ascending", value=True)
            if st.button("Sort"):
                st.session_state.df = sort_dataframe(
                    st.session_state.df, sort_col, ascending=ascending
                )
                st.success("Dataset sorted.")


def data_preview_tab():
    st.header("Data Preview")
    if st.session_state.df is None:
        st.info("Upload a dataset to start analyzing.")
        return

    df = st.session_state.df
    st.subheader("Quick Overview")
    cols1, cols2 = st.columns(2)
    cols1.metric("Rows", df.shape[0])
    cols1.metric("Columns", df.shape[1])
    cols2.metric("Missing cells", int(df.isna().sum().sum()))
    cols2.metric("Duplicate rows", int(df.duplicated().sum()))

    st.subheader("Schema")
    schema = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum()})
    st.dataframe(schema, use_container_width=True)

    st.subheader("Preview (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Automatic Insights")
    with st.expander("Dataset quality"):
        missing_pct = (df.isna().sum() / len(df) * 100).round(2)
        st.write("Missing Value % by column")
        st.dataframe(missing_pct.to_frame("% missing"))

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        st.write(f"Detected numeric columns: {numeric_cols}")
        st.write("Column counts by type:")
        st.dataframe(df.dtypes.value_counts().to_frame("count"))


def visualization_tab():
    st.header("Visualization")
    if st.session_state.df is None:
        st.info("Upload a dataset to visualize.")
        return

    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
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
            "Distribution (seaborn)",
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
    corr = correlation_matrix(df)
    cov = covariance_matrix(df)
    st.write("Correlation")
    st.dataframe(corr, use_container_width=True)
    st.write("Covariance")
    st.dataframe(cov, use_container_width=True)

    st.subheader("Regression & Outliers")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) >= 2:
        reg_x = st.selectbox("Regression X (predictor)", [None] + numeric_cols)
        reg_y = st.selectbox("Regression Y (target)", [None] + numeric_cols)
        if reg_x and reg_y and st.button("Run linear regression"):
            result = linear_regression(df, reg_x, reg_y)
            st.write("**Linear regression results**")
            st.write(result)
    else:
        st.info("Need at least 2 numeric columns for regression.")

    st.subheader("Outlier detection")
    if len(numeric_cols) > 0:
        outlier_col = st.selectbox("Outlier column", [None] + numeric_cols)
        if outlier_col:
            if st.button("Detect outliers (Z-score)"):
                outliers = detect_outliers_zscore(df, outlier_col)
                st.write(f"{len(outliers)} outliers detected")
                st.dataframe(outliers.head())
            if st.button("Detect outliers (IQR)"):
                outliers = detect_outliers_iqr(df, outlier_col)
                st.write(f"{len(outliers)} outliers detected")
                st.dataframe(outliers.head())
    else:
        st.info("No numeric columns available for outlier detection.")

    st.subheader("Hypothesis testing")
    cols = df.columns.tolist()
    tcol1 = st.selectbox("Sample 1 column", [None] + cols, key="t1")
    tcol2 = st.selectbox("Sample 2 column", [None] + cols, key="t2")
    if tcol1 and tcol2:
        alpha = st.number_input("Significance level (alpha)", min_value=0.001, max_value=0.2, value=0.05, step=0.01)
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


def export_tab():
    st.header("Export")
    if st.session_state.df is None:
        st.info("Upload and clean a dataset before exporting.")
        return

    st.subheader("Download Cleaned Dataset")
    csv_bytes = dataframe_to_csv(st.session_state.df)
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )

    st.subheader("Download Statistical Summary")
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
    st.title("Smart Data Analyzer")
    st.markdown(
        "A beginner-friendly yet powerful data analysis app. Upload a dataset, clean it, explore patterns, and export results."
    )

    sidebar_upload()
    sidebar_cleaning_tools()

    tabs = st.tabs(["Data Preview", "Data Cleaning", "Visualization", "Statistics", "Export"])
    with tabs[0]:
        data_preview_tab()
    with tabs[1]:
        st.write("Use the sidebar cleaning tools to update the dataset. Preview changes in the Data Preview tab.")
    with tabs[2]:
        visualization_tab()
    with tabs[3]:
        statistics_tab()
    with tabs[4]:
        export_tab()

    st.markdown("---")
    st.caption("Built with Streamlit, pandas, Plotly, SciPy, and scikit-learn.")


if __name__ == "__main__":
    main()
