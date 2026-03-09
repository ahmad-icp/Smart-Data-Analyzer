import pandas as pd


def remove_missing(df: pd.DataFrame, axis: int = 0, how: str = "any") -> pd.DataFrame:
    """Remove rows (axis=0) or columns (axis=1) with missing values."""
    return df.dropna(axis=axis, how=how).copy()


def fill_missing(
    df: pd.DataFrame,
    columns=None,
    method: str = "mean",
    custom_value=None,
) -> pd.DataFrame:
    """Fill missing values in columns using a method (mean/median/mode/custom)."""
    result = df.copy()

    if columns is None or len(columns) == 0:
        columns = result.columns.tolist()

    for col in columns:
        if col not in result.columns:
            continue

        if method == "mean":
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].mean())
        elif method == "median":
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].median())
        elif method == "mode":
            mode_val = result[col].mode()
            if not mode_val.empty:
                result[col] = result[col].fillna(mode_val.iloc[0])
        elif method == "custom":
            result[col] = result[col].fillna(custom_value)
        else:
            raise ValueError(f"Unknown fill method: {method}")

    return result


def remove_duplicates(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame."""
    return df.drop_duplicates(subset=subset).copy()


def drop_columns(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Drop selected columns from the dataset."""
    if not columns:
        return df
    return df.drop(columns=columns, errors="ignore").copy()


def rename_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Rename columns using a mapping dict."""
    return df.rename(columns=rename_map).copy()


def convert_column_type(df: pd.DataFrame, column: str, target_type: str) -> pd.DataFrame:
    """Convert a column to a different dtype."""
    result = df.copy()
    if column not in result.columns:
        return result

    try:
        if target_type == "int":
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
        elif target_type == "float":
            result[column] = pd.to_numeric(result[column], errors="coerce")
        elif target_type == "str":
            result[column] = result[column].astype(str)
        elif target_type == "datetime":
            result[column] = pd.to_datetime(result[column], errors="coerce")
        elif target_type == "category":
            result[column] = result[column].astype("category")
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
    except Exception:
        raise

    return result


def filter_rows(df: pd.DataFrame, column: str, op: str, value) -> pd.DataFrame:
    """Filter rows using a simple comparison operator."""
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    series = df[column]

    if op == "contains":
        return df[series.astype(str).str.contains(str(value), na=False)]

    # Try numeric conversion when possible
    try:
        comp_value = float(value)
    except Exception:
        comp_value = value

    if op == "==":
        return df[series == comp_value]
    if op == "!=":
        return df[series != comp_value]
    if op == ">":
        return df[series.astype(float) > float(comp_value)]
    if op == "<":
        return df[series.astype(float) < float(comp_value)]
    if op == ">=":
        return df[series.astype(float) >= float(comp_value)]
    if op == "<=":
        return df[series.astype(float) <= float(comp_value)]

    raise ValueError(f"Unsupported operator: {op}")


def sort_dataframe(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
    """Sort a DataFrame by a column."""
    if column not in df.columns:
        return df
    return df.sort_values(by=column, ascending=ascending).copy()
