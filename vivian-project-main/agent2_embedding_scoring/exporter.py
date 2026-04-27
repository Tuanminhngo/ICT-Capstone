from __future__ import annotations



import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


SCORE_COLUMNS = [
    "row_id",
    "thought_text",
    "window_size_seconds",
    "model_name",
    "similarity_score",
    "video_embedding_source",
    "scoring_status",
    "scoring_message",
    "clip_path",
]


def ensure_output_directories(output_dir: Path, cache_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)


def export_similarity_scores(
    records: List[Dict[str, Any]],
    output_dir: Path,
    csv_name: str,
    json_name: str,
) -> None:
    dataframe = pd.DataFrame(records, columns=SCORE_COLUMNS)
    dataframe.to_csv(output_dir / csv_name, index=False)
    with (output_dir / json_name).open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2, default=str)


def export_model_comparison(
    records: List[Dict[str, Any]],
    output_dir: Path,
    csv_name: str,
) -> None:
    """Write the model-comparison subset (CLIP vs ImageBind on sample rows)."""
    if not records:
        return
    dataframe = pd.DataFrame(records, columns=SCORE_COLUMNS)
    dataframe.to_csv(output_dir / csv_name, index=False)


def export_validation_report(
    issues: List[Dict[str, Any]],
    output_dir: Path,
    report_name: str,
) -> None:
    dataframe = pd.DataFrame(
        issues,
        columns=["row_id", "issue_type", "issue_description", "severity"],
    )
    dataframe.to_csv(output_dir / report_name, index=False)


def export_summary_report(
    summary: Dict[str, Any],
    output_dir: Path,
    report_name: str,
) -> None:
    with (output_dir / report_name).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
