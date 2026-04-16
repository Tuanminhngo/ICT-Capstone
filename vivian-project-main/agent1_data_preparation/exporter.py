from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


def ensure_output_directories(base_output_dir: Path, window_sizes: Iterable[int]) -> None:
    (base_output_dir / "clips").mkdir(parents=True, exist_ok=True)
    for window in window_sizes:
        (base_output_dir / "clips" / f"{window}s").mkdir(parents=True, exist_ok=True)


def export_dataset_rows(
    records: List[Dict[str, Any]],
    output_dir: Path,
    csv_name: str,
    json_name: str,
) -> None:
    dataframe = pd.DataFrame(records)
    dataframe.to_csv(output_dir / csv_name, index=False)
    with (output_dir / json_name).open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2, default=str)


def export_validation_report(
    issues: List[Dict[str, Any]],
    output_dir: Path,
    validation_report_name: str,
) -> None:
    dataframe = pd.DataFrame(
        issues,
        columns=["row_id", "issue_type", "issue_description", "severity"],
    )
    dataframe.to_csv(output_dir / validation_report_name, index=False)


def export_summary_report(
    summary: Dict[str, Any],
    output_dir: Path,
    summary_report_name: str,
) -> None:
    with (output_dir / summary_report_name).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
