import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .ml_readiness import build_data_dictionary


def _secret(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        return None


def _dataset_context(df: pd.DataFrame, quality: Dict[str, Any], title: str) -> Dict[str, Any]:
    dictionary = build_data_dictionary(df)
    private_dictionary = dictionary.drop(columns=["examples"], errors="ignore")
    return {
        "requested_title": title or "Smart Dataset",
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": private_dictionary.to_dict(orient="records"),
        "quality": quality,
    }


def _prompt(context: Dict[str, Any]) -> str:
    return (
        "You are a dataset documentation specialist. Using only the supplied metadata, "
        "write a polished Markdown data card. Include: title, subtitle, overview, dataset "
        "structure, data quality, column guide, recommended analysis and ML tasks, "
        "limitations, responsible-use notes, and eight searchable Kaggle tags. Never "
        "invent provenance, collection methods, causal claims, or results. Clearly label "
        "unknown information.\n\nDATASET METADATA:\n"
        + json.dumps(context, default=str)
    )


def _post_json(
    url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int = 45
) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _ollama(prompt: str, model: str) -> str:
    base_url = (_secret("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
    data = _post_json(
        f"{base_url}/api/chat",
        {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.2},
        },
        {},
        timeout=90,
    )
    return data["message"]["content"]


def _gemini(prompt: str, model: str) -> str:
    key = _secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    data = _post_json(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}",
        {"contents": [{"parts": [{"text": prompt}]}]},
        {},
    )
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _groq(prompt: str, model: str) -> str:
    key = _secret("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    data = _post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {
            "model": model,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
        },
        {"Authorization": f"Bearer {key}"},
    )
    return data["choices"][0]["message"]["content"]


def template_data_card(
    df: pd.DataFrame, quality: Dict[str, Any], title: str
) -> str:
    title = title.strip() or "Smart Dataset"
    dictionary = build_data_dictionary(df)
    issues = quality.get("issues") or [
        "No major automated quality warnings were detected."
    ]
    recommendations = quality.get("recommendations") or [
        "Review the data before modelling."
    ]
    column_lines = [
        f"- **{row.column}** (`{row.dtype}`): {row.missing} missing; "
        f"{row.unique} unique. Examples: {row.examples or 'not available'}"
        for row in dictionary.itertuples()
    ]
    return "\n".join(
        [
            f"# {title}",
            "",
            f"**A documented dataset with {len(df):,} rows and "
            f"{len(df.columns):,} columns, prepared for reproducible analysis.**",
            "",
            "## Overview",
            "",
            "This data card was generated from the dataset schema and quality checks. "
            "Source and collection details were not provided and should be added by the publisher.",
            "",
            "## Data quality",
            "",
            f"- ML-readiness score: **{quality.get('score', 'N/A')}/100**",
            *[f"- {item}" for item in issues],
            "",
            "## Column guide",
            "",
            *column_lines,
            "",
            "## Recommended next steps",
            "",
            *[f"- {item}" for item in recommendations],
            "",
            "## Suitable tasks",
            "",
            "- Exploratory data analysis and visualization",
            "- Supervised learning when an appropriate target is selected",
            "- Data-cleaning and feature-engineering practice",
            "",
            "## Limitations and responsible use",
            "",
            "- Verify provenance, consent, licensing, sampling and domain assumptions.",
            "- Automated checks do not establish accuracy, fairness or causal validity.",
            "- Review potentially identifying columns before sharing the dataset.",
            "",
            "## Suggested Kaggle tags",
            "",
            "`data-cleaning`, `eda`, `machine-learning`, `tabular-data`, "
            "`data-visualization`, `beginner`, `research`, `analytics`",
        ]
    )


def generate_data_card(
    df: pd.DataFrame,
    quality: Dict[str, Any],
    title: str = "",
    provider: str = "auto",
) -> Tuple[str, str, Optional[str]]:
    """Generate documentation with local/cloud AI and a deterministic fallback."""
    prompt = _prompt(_dataset_context(df, quality, title))
    requested = provider.lower()
    attempts = []

    if requested in ("auto", "ollama"):
        attempts.append(
            (
                "Ollama / Qwen",
                lambda: _ollama(prompt, _secret("OLLAMA_MODEL") or "qwen3:4b"),
            )
        )
    if requested in ("auto", "gemini") and _secret("GEMINI_API_KEY"):
        attempts.append(
            (
                "Gemini",
                lambda: _gemini(
                    prompt, _secret("GEMINI_MODEL") or "gemini-3-flash-preview"
                ),
            )
        )
    if requested in ("auto", "groq") and _secret("GROQ_API_KEY"):
        attempts.append(
            (
                "Groq",
                lambda: _groq(
                    prompt, _secret("GROQ_MODEL") or "llama-3.3-70b-versatile"
                ),
            )
        )

    errors = []
    for name, writer in attempts:
        try:
            text = writer().strip()
            if len(text) >= 200:
                return text, name, None
            errors.append(f"{name} returned incomplete text")
        except (
            RuntimeError,
            KeyError,
            IndexError,
            urllib.error.URLError,
            TimeoutError,
        ) as exc:
            errors.append(f"{name}: {exc}")

    warning = "; ".join(errors) if errors else None
    return template_data_card(df, quality, title), "Built-in professional writer", warning
