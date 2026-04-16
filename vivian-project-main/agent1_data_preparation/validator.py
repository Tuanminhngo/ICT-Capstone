from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from data_loader import NormalizedThoughtRow
from video_utils import VideoMatch


@dataclass
class ValidationIssue:
    row_id: str
    issue_type: str
    issue_description: str
    severity: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "row_id": self.row_id,
            "issue_type": self.issue_type,
            "issue_description": self.issue_description,
            "severity": self.severity,
        }


def detect_duplicate_keys(rows: List[NormalizedThoughtRow]) -> Set[Tuple[str, Optional[float], str]]:
    seen: Set[Tuple[str, Optional[float], str]] = set()
    duplicates: Set[Tuple[str, Optional[float], str]] = set()
    for row in rows:
        key = (
            str(row.video_key_raw).strip().lower(),
            row.timestamp_seconds,
            row.thought_text.strip().lower(),
        )
        if key in seen:
            duplicates.add(key)
        else:
            seen.add(key)
    return duplicates


def validate_row(
    row: NormalizedThoughtRow,
    matched_video: Optional[VideoMatch],
    duration_seconds: Optional[float],
    duplicate_keys: Set[Tuple[str, Optional[float], str]],
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    raw_values = [value for value in row.original_data.values() if value not in (None, "")]
    if not raw_values:
        issues.append(
            ValidationIssue(
                row_id=row.row_id,
                issue_type="empty_row",
                issue_description="The row does not contain any usable values.",
                severity="error",
            )
        )

    if not row.thought_text:
        issues.append(
            ValidationIssue(
                row_id=row.row_id,
                issue_type="missing_thought_text",
                issue_description="Thought text is missing or blank.",
                severity="error",
            )
        )

    if row.timestamp_seconds is None:
        issues.append(
            ValidationIssue(
                row_id=row.row_id,
                issue_type="invalid_timestamp",
                issue_description=f"Could not parse timestamp value '{row.original_timestamp}'.",
                severity="error",
            )
        )

    duplicate_key = (
        str(row.video_key_raw).strip().lower(),
        row.timestamp_seconds,
        row.thought_text.strip().lower(),
    )
    if duplicate_key in duplicate_keys:
        issues.append(
            ValidationIssue(
                row_id=row.row_id,
                issue_type="duplicate_row",
                issue_description="A duplicate row with the same video, timestamp, and thought text was found.",
                severity="warning",
            )
        )

    if matched_video is None:
        issues.append(
            ValidationIssue(
                row_id=row.row_id,
                issue_type="missing_source_video",
                issue_description=f"No source video could be matched for '{row.video_key_raw}'.",
                severity="error",
            )
        )

    if row.timestamp_seconds is not None and row.timestamp_seconds < 0:
        issues.append(
            ValidationIssue(
                row_id=row.row_id,
                issue_type="negative_timestamp",
                issue_description=f"Timestamp {row.timestamp_seconds:.3f}s is negative.",
                severity="error",
            )
        )

    if row.timestamp_seconds is not None and duration_seconds is not None and row.timestamp_seconds > duration_seconds:
        issues.append(
            ValidationIssue(
                row_id=row.row_id,
                issue_type="timestamp_beyond_duration",
                issue_description=f"Timestamp {row.timestamp_seconds:.3f}s exceeds video duration {duration_seconds:.3f}s.",
                severity="error",
            )
        )

    return issues


def summarize_validation_status(issues: List[ValidationIssue]) -> Tuple[str, str]:
    if not issues:
        return "valid", ""

    severities = {issue.severity for issue in issues}
    status = "invalid" if "error" in severities else "warning"
    message = " | ".join(issue.issue_description for issue in issues)
    return status, message
