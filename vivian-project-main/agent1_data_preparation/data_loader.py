from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import Settings


@dataclass
class NormalizedThoughtRow:
    row_id: str
    source_row_number: int
    original_data: Dict[str, Any]
    video_key_raw: Any
    thought_text_raw: Any
    original_timestamp: Any
    thought_text: str
    timestamp_seconds: Optional[float]
    normalized_fields: Dict[str, Any]


def normalize_column_name(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "_")


def resolve_column_mapping(
    dataframe: pd.DataFrame, settings: Settings
) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    normalized_to_actual = {
        normalize_column_name(column): str(column) for column in dataframe.columns
    }
    mapping: Dict[str, str] = {}
    issues: List[Dict[str, str]] = []

    for field_name, configured_name in settings.column_mapping.items():
        if configured_name:
            normalized_name = normalize_column_name(configured_name)
            if normalized_name in normalized_to_actual:
                mapping[field_name] = normalized_to_actual[normalized_name]
                continue
            issues.append(
                {
                    "issue_type": "configured_column_missing",
                    "issue_details": f"Configured column '{configured_name}' was not found for field '{field_name}'.",
                    "severity": "warning",
                }
            )

        for alias in settings.column_aliases.get(field_name, []):
            normalized_alias = normalize_column_name(alias)
            if normalized_alias in normalized_to_actual:
                mapping[field_name] = normalized_to_actual[normalized_alias]
                break

        if field_name not in mapping:
            issues.append(
                {
                    "issue_type": "required_column_missing",
                    "issue_details": f"Could not resolve a column for required field '{field_name}'.",
                    "severity": "error",
                }
            )

    return mapping, issues


def parse_timestamp_to_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, pd.Timestamp):
        return float(
            value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000
        )

    if isinstance(value, timedelta):
        return float(value.total_seconds())

    if isinstance(value, time):
        return float(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000)

    if isinstance(value, datetime):
        return float(
            value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000
        )

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        pass

    pieces = text.split(":")
    if len(pieces) in (2, 3):
        try:
            numeric_pieces = [float(piece) for piece in pieces]
        except ValueError:
            numeric_pieces = []
        if numeric_pieces:
            if len(numeric_pieces) == 2:
                minutes, seconds = numeric_pieces
                return minutes * 60 + seconds
            hours, minutes, seconds = numeric_pieces
            return hours * 3600 + minutes * 60 + seconds

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return float(
            parsed.hour * 3600 + parsed.minute * 60 + parsed.second + parsed.microsecond / 1_000_000
        )

    return None


def load_thought_reports(
    settings: Settings,
) -> Tuple[pd.DataFrame, Dict[str, str], List[NormalizedThoughtRow], List[Dict[str, Any]]]:
    sheet_name = settings.excel_sheet_name if settings.excel_sheet_name is not None else 0
    dataframe = pd.read_excel(settings.resolved_excel_path, sheet_name=sheet_name)
    column_mapping, mapping_issues = resolve_column_mapping(dataframe, settings)
    if any(issue["severity"] == "error" for issue in mapping_issues):
        return dataframe, column_mapping, [], mapping_issues

    normalized_rows: List[NormalizedThoughtRow] = []

    for idx, row in dataframe.iterrows():
        row_id_value = row.get(column_mapping["row_id"])
        row_id = str(row_id_value).strip() if pd.notna(row_id_value) else f"row_{idx + 1}"

        thought_value = row.get(column_mapping["thought_text"])
        thought_text = "" if pd.isna(thought_value) else str(thought_value).strip()

        timestamp_value = row.get(column_mapping["timestamp"])
        timestamp_seconds = parse_timestamp_to_seconds(timestamp_value)

        normalized_rows.append(
            NormalizedThoughtRow(
                row_id=row_id,
                source_row_number=idx + 2,
                original_data=row.to_dict(),
                video_key_raw=row.get(column_mapping["video"]),
                thought_text_raw=thought_value,
                original_timestamp=timestamp_value,
                thought_text=thought_text,
                timestamp_seconds=timestamp_seconds,
                normalized_fields={
                    "row_id": row_id,
                    "video": row.get(column_mapping["video"]),
                    "timestamp": timestamp_value,
                    "thought_text": thought_text,
                },
            )
        )

    return dataframe, column_mapping, normalized_rows, mapping_issues
