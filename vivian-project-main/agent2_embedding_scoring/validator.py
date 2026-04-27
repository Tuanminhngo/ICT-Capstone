from __future__ import annotations



from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from data_loader import PairedRow


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


def validate_row(row: PairedRow, skip_invalid: bool, skip_missing: bool) -> List[ValidationIssue]:
   
    issues: List[ValidationIssue] = []

    if not row.thought_text:
        issues.append(
            ValidationIssue(
                row_id=row.row_id,
                issue_type="missing_thought_text",
                issue_description="Thought text is empty; cannot encode.",
                severity="error",
            )
        )

    if skip_invalid and row.validation_status == "invalid":
        issues.append(
            ValidationIssue(
                row_id=row.row_id,
                issue_type="agent1_invalid_row",
                issue_description=(
                    f"Row was marked invalid by Agent 1 (validation_status='{row.validation_status}')."
                ),
                severity="error",
            )
        )

    if skip_missing and not row.clip_exists:
        issues.append(
            ValidationIssue(
                row_id=row.row_id,
                issue_type="clip_file_missing",
                issue_description=f"Agent 1 reports clip_exists=False for '{row.clip_path}'.",
                severity="error",
            )
        )

    if row.clip_exists and not row.clip_path.exists():
        issues.append(
            ValidationIssue(
                row_id=row.row_id,
                issue_type="clip_not_found_on_disk",
                issue_description=f"Clip file not found on disk: '{row.clip_path}'.",
                severity="error",
            )
        )

    if row.window_size_seconds not in (5, 10, 15):
        issues.append(
            ValidationIssue(
                row_id=row.row_id,
                issue_type="unexpected_window_size",
                issue_description=(
                    f"Window size {row.window_size_seconds}s is outside the expected set {{5, 10, 15}}."
                ),
                severity="warning",
            )
        )

    return issues


def summarize_validation_status(issues: List[ValidationIssue]) -> Tuple[str, str]:
    """Collapse issue list to (status, message) — mirrors Agent 1 helper."""
    if not issues:
        return "valid", ""
    severities = {issue.severity for issue in issues}
    status = "invalid" if "error" in severities else "warning"
    message = " | ".join(issue.issue_description for issue in issues)
    return status, message
