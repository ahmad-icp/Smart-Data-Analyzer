import pandas as pd
import plotly.express as px
from typing import Any, Dict, List, Optional


def apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    """Apply a list of filters to a DataFrame."""
    if df is None or df.empty or not filters:
        return df

    filtered = df.copy()
    for f in filters:
        col = f.get("column")
        op = f.get("op")
        value = f.get("value")
        if col not in filtered.columns or value is None:
            continue

        if op == "==":
            filtered = filtered[filtered[col] == value]
        elif op == "!=":
            filtered = filtered[filtered[col] != value]
        elif op == ">":
            filtered = filtered[filtered[col] > value]
        elif op == "<":
            filtered = filtered[filtered[col] < value]
        elif op == ">=":
            filtered = filtered[filtered[col] >= value]
        elif op == "<=":
            filtered = filtered[filtered[col] <= value]
        elif op == "in":
            filtered = filtered[filtered[col].isin(value if isinstance(value, list) else [value])]
        elif op == "range":
            low, high = value
            filtered = filtered[(filtered[col] >= low) & (filtered[col] <= high)]

    return filtered


def create_dashboard_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[Any]:
    """Create a chart for the dashboard."""

    if df is None or df.empty:
        return None

    chart = (chart_type or "").lower()
    title = title or f"{chart_type}"

    if chart in ["bar", "bar chart"]:
        return px.bar(df, x=x, y=y, color=color, title=title)
    if chart in ["line", "line chart"]:
        return px.line(df, x=x, y=y, color=color, title=title)
    if chart in ["scatter", "scatter plot"]:
        return px.scatter(df, x=x, y=y, color=color, title=title)
    if chart in ["histogram", "hist"]:
        return px.histogram(df, x=x, color=color, title=title)
    if chart in ["box", "box plot"]:
        return px.box(df, x=x, y=y, color=color, title=title)
    if chart in ["heatmap", "correlation"]:
        numeric = df.select_dtypes(include=["number"])
        corr = numeric.corr()
        return px.imshow(corr, title=title, zmin=-1, zmax=1, color_continuous_scale="RdBu")

    return None
