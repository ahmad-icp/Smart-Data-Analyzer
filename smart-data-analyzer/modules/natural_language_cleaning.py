import json
import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from .data_cleaning import (
    clean_string_column,
    convert_column_type,
    drop_columns,
    fill_missing,
    normalize_column,
    remove_duplicates,
    remove_missing,
    remove_outliers_iqr,
    remove_outliers_zscore,
    rename_columns,
    standardize_categories,
)


ALLOWED_ACTIONS = {
    "remove_duplicates",
    "remove_missing_rows",
    "fill_missing",
    "drop_columns",
    "rename_column",
    "convert_type",
    "clean_strings",
    "standardize_categories",
    "normalize",
    "standardize",
    "remove_outliers_iqr",
    "remove_outliers_zscore",
}


def _resolve_column(name: str, columns: List[str]) -> str:
    cleaned = name.strip().strip("`'\" ")
    exact = {str(column).lower(): str(column) for column in columns}
    if cleaned.lower() in exact:
        return exact[cleaned.lower()]
    raise ValueError(f"Column not found: {cleaned}")


def parse_instruction(instruction: str, columns: List[str]) -> List[Dict[str, Any]]:
    """Parse common natural-language cleaning commands without an AI dependency."""
    text = instruction.strip()
    lowered = text.lower()
    actions: List[Dict[str, Any]] = []

    if re.search(r"\b(remove|delete)\b.*\bduplicates?\b", lowered):
        actions.append({"action": "remove_duplicates", "params": {}})

    if re.search(r"\b(remove|delete|drop)\b.*\b(rows?)\b.*\bmissing\b", lowered):
        actions.append({"action": "remove_missing_rows", "params": {}})

    fill_pattern = re.compile(
        r"fill\s+(?:the\s+)?missing(?:\s+values?)?(?:\s+in)?\s+"
        r"[`'\"]?([^,.;]+?)[`'\"]?\s+(?:with|using)\s+"
        r"(mean|median|mode|[^\s,.;]+)",
        re.IGNORECASE,
    )
    for match in fill_pattern.finditer(text):
        column = _resolve_column(match.group(1), columns)
        value = match.group(2).strip()
        method = value.lower() if value.lower() in {"mean", "median", "mode"} else "custom"
        actions.append(
            {
                "action": "fill_missing",
                "params": {
                    "columns": [column],
                    "method": method,
                    "custom_value": None if method != "custom" else value,
                },
            }
        )

    for match in re.finditer(
        r"rename\s+(?:column\s+)?[`'\"]?(.+?)[`'\"]?\s+to\s+[`'\"]?([^,.;]+)",
        text,
        re.IGNORECASE,
    ):
        actions.append(
            {
                "action": "rename_column",
                "params": {
                    "column": _resolve_column(match.group(1), columns),
                    "new_name": match.group(2).strip().strip("`'\" "),
                },
            }
        )

    for match in re.finditer(
        r"(?:drop|remove|delete)\s+(?:the\s+)?column\s+[`'\"]?([^,.;]+)",
        text,
        re.IGNORECASE,
    ):
        actions.append(
            {
                "action": "drop_columns",
                "params": {"columns": [_resolve_column(match.group(1), columns)]},
            }
        )

    for match in re.finditer(
        r"convert\s+[`'\"]?(.+?)[`'\"]?\s+to\s+"
        r"(integer|int|float|numeric|string|str|date|datetime|category)",
        text,
        re.IGNORECASE,
    ):
        target = {
            "integer": "int",
            "numeric": "float",
            "string": "str",
            "date": "datetime",
        }.get(match.group(2).lower(), match.group(2).lower())
        actions.append(
            {
                "action": "convert_type",
                "params": {
                    "column": _resolve_column(match.group(1), columns),
                    "target_type": target,
                },
            }
        )

    operation_patterns: List[Tuple[str, str]] = [
        ("standardize categories", "standardize_categories"),
        ("clean strings", "clean_strings"),
        ("normalize", "normalize"),
        ("standardize", "standardize"),
    ]
    for phrase, action in operation_patterns:
        if phrase == "standardize" and "standardize categories" in lowered:
            continue
        match = re.search(
            rf"\b{phrase}\b(?:\s+(?:in|for|column))?\s+[`'\"]?([^,.;]+)",
            text,
            re.IGNORECASE,
        )
        if match:
            actions.append(
                {
                    "action": action,
                    "params": {"column": _resolve_column(match.group(1), columns)},
                }
            )

    for method, action in [
        ("iqr", "remove_outliers_iqr"),
        ("z-score", "remove_outliers_zscore"),
        ("zscore", "remove_outliers_zscore"),
    ]:
        match = re.search(
            rf"remove\s+outliers?(?:\s+(?:in|from))?\s+[`'\"]?(.+?)[`'\"]?"
            rf"(?:\s+(?:using|with)\s+{re.escape(method)})?(?:[,.;]|$)",
            text,
            re.IGNORECASE,
        )
        if match and (method in lowered or method == "iqr"):
            actions.append(
                {
                    "action": action,
                    "params": {"column": _resolve_column(match.group(1), columns)},
                }
            )
            break

    return validate_plan(actions, columns)


def parse_ai_plan(response: str, columns: List[str]) -> List[Dict[str, Any]]:
    """Extract and validate an AI-produced JSON cleaning plan."""
    match = re.search(r"\[[\s\S]*\]", response)
    if not match:
        raise ValueError("AI response did not contain a JSON action list.")
    try:
        plan = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError("AI returned invalid JSON.") from exc
    return validate_plan(plan, columns)


def validate_plan(plan: List[Dict[str, Any]], columns: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(plan, list) or len(plan) > 20:
        raise ValueError("Cleaning plan must contain between 0 and 20 actions.")

    validated = []
    current_columns = list(columns)
    for item in plan:
        if not isinstance(item, dict) or item.get("action") not in ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported cleaning action: {item}")
        action = item["action"]
        params = dict(item.get("params") or {})

        if "column" in params:
            params["column"] = _resolve_column(str(params["column"]), current_columns)
        if "columns" in params:
            params["columns"] = [
                _resolve_column(str(column), current_columns)
                for column in params["columns"]
            ]
        if action == "rename_column":
            new_name = str(params.get("new_name", "")).strip()
            if not new_name or new_name in current_columns:
                raise ValueError("Renamed column must have a new, unique name.")
            old_name = params["column"]
            current_columns[current_columns.index(old_name)] = new_name
            params["new_name"] = new_name
        elif action == "drop_columns":
            current_columns = [
                column for column in current_columns if column not in params["columns"]
            ]
        elif action == "convert_type" and params.get("target_type") not in {
            "int",
            "float",
            "str",
            "datetime",
            "category",
        }:
            raise ValueError("Unsupported target data type.")

        validated.append({"action": action, "params": params})
    return validated


def apply_plan(df: pd.DataFrame, plan: List[Dict[str, Any]]) -> pd.DataFrame:
    result = df.copy()
    for item in plan:
        action = item["action"]
        params = item.get("params", {})
        if action == "remove_duplicates":
            result = remove_duplicates(result)
        elif action == "remove_missing_rows":
            result = remove_missing(result)
        elif action == "fill_missing":
            result = fill_missing(result, **params)
        elif action == "drop_columns":
            result = drop_columns(result, params["columns"])
        elif action == "rename_column":
            result = rename_columns(
                result, {params["column"]: params["new_name"]}
            )
        elif action == "convert_type":
            result = convert_column_type(result, **params)
        elif action == "clean_strings":
            result = clean_string_column(result, params["column"])
        elif action == "standardize_categories":
            result = standardize_categories(result, params["column"])
        elif action == "normalize":
            result = normalize_column(result, params["column"], "minmax")
        elif action == "standardize":
            result = normalize_column(result, params["column"], "zscore")
        elif action == "remove_outliers_iqr":
            result = remove_outliers_iqr(result, params["column"])
        elif action == "remove_outliers_zscore":
            result = remove_outliers_zscore(result, params["column"])
    return result


def plan_summary(plan: List[Dict[str, Any]]) -> List[str]:
    return [
        f"{index}. {item['action'].replace('_', ' ').title()} — {item.get('params') or 'all rows'}"
        for index, item in enumerate(plan, start=1)
    ]
