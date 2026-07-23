import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def suggest_chart_types(df: pd.DataFrame) -> list:
    """Suggest chart types based on data types in the dataframe."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [
        column
        for column in df.columns
        if pd.api.types.is_string_dtype(df[column])
        or isinstance(df[column].dtype, pd.CategoricalDtype)
    ]
    suggestions = []
    if num_cols:
        suggestions.append("Histogram")
        suggestions.append("Scatter")
        suggestions.append("Line")
    if cat_cols:
        suggestions.append("Bar")
    suggestions.append("Correlation")
    return suggestions


def create_plot(df: pd.DataFrame, chart_type: str, x: str = None, y: str = None):
    """Create an interactive Plotly chart based on user's selection."""

    chart_type = (chart_type or "").lower()
    if chart_type in ["auto", ""]:
        if x and y:
            chart_type = "scatter"
        elif x:
            chart_type = "histogram"
        else:
            chart_type = "histogram"

    if chart_type in ["histogram"]:
        if x is None:
            raise ValueError("Select an X column for histogram")
        return px.histogram(df, x=x, nbins=30, title=f"Histogram: {x}")

    if chart_type in ["bar"]:
        if x is None:
            raise ValueError("Select an X column for bar chart")
        if y is None:
            # show counts
            counts = (
                df[x]
                .value_counts(dropna=False)
                .rename_axis(x)
                .reset_index(name="count")
            )
            return px.bar(counts, x=x, y="count", title=f"Bar chart: {x}")
        return px.bar(df, x=x, y=y, title=f"Bar chart: {y} by {x}")

    if chart_type in ["line"]:
        if x is None or y is None:
            raise ValueError("Select X and Y columns for line chart")
        return px.line(df, x=x, y=y, title=f"Line chart: {y} vs {x}")

    if chart_type in ["scatter"]:
        if x is None or y is None:
            raise ValueError("Select X and Y columns for scatter plot")
        return px.scatter(df, x=x, y=y, title=f"Scatter: {y} vs {x}")

    if chart_type in ["box"]:
        if y is None:
            raise ValueError("Select a Y column for box plot")
        return px.box(df, y=y, title=f"Box plot: {y}")

    if chart_type in ["heatmap", "correlation", "heatmap (correlation)"]:
        corr = df.select_dtypes(include=["number"]).corr()
        return px.imshow(
            corr,
            title="Correlation heatmap",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
        )

    if chart_type in ["distribution", "distribution (seaborn)"]:
        if x is None:
            raise ValueError("Select a column for distribution plot")
        return px.histogram(df, x=x, nbins=40, marginal="violin", title=f"Distribution: {x}")

    # Fallback
    if x and y:
        return px.scatter(df, x=x, y=y, title=f"Scatter: {y} vs {x}")
    if x:
        return px.histogram(df, x=x, title=f"Histogram: {x}")

    raise ValueError("Unable to create chart with the given parameters")
