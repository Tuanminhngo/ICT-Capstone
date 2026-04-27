from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from config import Settings

LOGGER = logging.getLogger(__name__)

# Columns Agent 1 guarantees to produce
REQUIRED_COLUMNS = {"row_id", "thought_text", "clip_path", "window_size_seconds", "validation_status", "clip_exists"}


@dataclass
class PairedRow:
    row_id: str
    thought_text: str
    clip_path: Path
    window_size_seconds: int
    validation_status: str
    clip_exists: bool
    source_row_number: int
    extra_fields: Dict[str, Any]


def load_paired_dataset(
    settings: Settings,
) -> Tuple[pd.DataFrame, List[PairedRow], List[Dict[str, str]]]:
    
    csv_path = settings.resolved_paired_dataset_path
    issues: List[Dict[str, str]] = []

    if not csv_path.exists():
        issues.append(
            {
                "row_id": "__schema__",
                "issue_type": "missing_input_file",
                "issue_description": f"paired_dataset.csv not found at '{csv_path}'.",
                "severity": "error",
            }
        )
        return pd.DataFrame(), [], issues

    dataframe = pd.read_csv(csv_path)
    LOGGER.info("Loaded paired_dataset.csv: %d rows from '%s'.", len(dataframe), csv_path)

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        issues.append(
            {
                "row_id": "__schema__",
                "issue_type": "missing_required_columns",
                "issue_description": f"Required columns absent from paired_dataset.csv: {sorted(missing_columns)}.",
                "severity": "error",
            }
        )
        return dataframe, [], issues

    rows: List[PairedRow] = []
    for idx, raw in dataframe.iterrows():
        row_id = str(raw["row_id"]).strip()
        thought_text = str(raw["thought_text"]).strip() if pd.notna(raw["thought_text"]) else ""
        clip_path = Path(str(raw["clip_path"]).strip())
        window_size = int(raw["window_size_seconds"])
        validation_status = str(raw["validation_status"]).strip().lower()
        clip_exists_raw = raw["clip_exists"]
        clip_exists = bool(clip_exists_raw) if not isinstance(clip_exists_raw, str) else clip_exists_raw.strip().lower() == "true"

        extra = {
            col: raw[col]
            for col in dataframe.columns
            if col not in REQUIRED_COLUMNS
        }

        rows.append(
            PairedRow(
                row_id=row_id,
                thought_text=thought_text,
                clip_path=clip_path,
                window_size_seconds=window_size,
                validation_status=validation_status,
                clip_exists=clip_exists,
                source_row_number=int(idx) + 2,
                extra_fields=extra,
            )
        )

    LOGGER.info("Parsed %d PairedRow objects.", len(rows))
    return dataframe, rows, issues
