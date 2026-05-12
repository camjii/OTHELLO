"""Text cleaning and JSON parsing helpers used across pipelines."""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, List, Optional


def strip_code_fences(text: str) -> str:
    """Strip ```...``` markdown fences from an LLM response."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    parts = text.split("```")
    if len(parts) < 2:
        return text
    inner = parts[1].strip()
    if inner.lower().startswith("json"):
        inner = inner[4:].lstrip()
    return inner.strip()


def safe_json_list(text: str, fallback: Optional[List[str]] = None) -> List[str]:
    """Parse an LLM response as a JSON list of strings.

    Falls back to the supplied default (or empty list) on any failure.
    """
    fallback = list(fallback or [])
    cleaned = strip_code_fences(text)
    try:
        value = json.loads(cleaned)
    except (ValueError, TypeError):
        return fallback
    if not isinstance(value, list):
        return fallback
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out or fallback


def normalize_label(text: str) -> str:
    """Collapse whitespace and strip surrounding quotes from a label."""
    if not isinstance(text, str):
        return ""
    text = text.strip().strip('"').strip("'")
    return re.sub(r"\s+", " ", text)


def safe_text(value: Any) -> str:
    """Coerce arbitrary values into a clean string, treating NaN as empty."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def clean_text(text: Any) -> Any:
    """Normalize free-form text that may have arrived as a list or stringified list."""
    if isinstance(text, list):
        text = text[0] if text else ""
    elif isinstance(text, str):
        stripped = text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list) and parsed:
                    text = parsed[0]
            except (ValueError, SyntaxError):
                pass

    if not isinstance(text, str):
        return text

    text = re.sub(r"^(AAVE|SAE)\s*Question[:\s-]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[\s\[\]\'\"]+|[\s\[\]\'\"]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()
