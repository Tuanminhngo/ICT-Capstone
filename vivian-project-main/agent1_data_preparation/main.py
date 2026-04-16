from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

from config import Settings
from data_loader import load_thought_reports
from exporter import ensure_output_directories, export_dataset_rows, export_summary_report, export_validation_report
from validator import ValidationIssue, detect_duplicate_keys, summarize_validation_status, validate_row
from video_utils import (
    VideoCatalog,
    build_clip_output_path,
    build_clip_time_range,
    clamp_clip_time_range,
    extract_clip,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def iter_rows(rows: List[Any], settings: Settings) -> Iterable[Any]:
    if settings.enable_progress_bar and tqdm is not None:
        return tqdm(rows, desc="Processing rows", unit="row")
    return rows


def build_dataset_record(
    row: Any,
    matched_video: Any,
    window_size: int,
    clip_path: str,
    clip_result: Any,
    validation_status: str,
    validation_message: str,
) -> Dict[str, Any]:
    return {
        "row_id": row.row_id,
        "video_group": matched_video.video_group,
        "source_video_file": matched_video.source_video_file,
        "source_video_path": str(matched_video.source_video_path),
        "thought_text": row.thought_text,
        "original_timestamp": row.original_timestamp,
        "timestamp_seconds": row.timestamp_seconds,
        "window_size_seconds": window_size,
        "clip_start_seconds": clip_result.clip_start_seconds,
        "clip_end_seconds": clip_result.clip_end_seconds,
        "clip_path": clip_path,
        "clip_exists": clip_result.clip_exists,
        "clip_extraction_mode": clip_result.extraction_mode_used,
        "expected_clip_duration_seconds": clip_result.expected_duration_seconds,
        "actual_clip_duration_seconds": clip_result.actual_duration_seconds,
        "duration_within_tolerance": clip_result.duration_within_tolerance,
        "playback_valid": clip_result.playback_valid,
        "used_fallback": clip_result.used_fallback,
        "debug_frame_path": clip_result.debug_frame_path,
        "validation_status": validation_status,
        "validation_message": validation_message,
    }


def run_pipeline(settings: Settings) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    ensure_output_directories(settings.resolved_output_dir, settings.clip_window_sizes)

    dataframe, column_mapping, rows, mapping_issues = load_thought_reports(settings)
    video_catalog = VideoCatalog(settings)
    duplicate_keys = detect_duplicate_keys(rows)

    dataset_rows: List[Dict[str, Any]] = []
    validation_issues: List[ValidationIssue] = [
        ValidationIssue(
            row_id="__schema__",
            issue_type=issue["issue_type"],
            issue_description=issue["issue_details"],
            severity=issue["severity"],
        )
        for issue in mapping_issues
    ]

    summary: Dict[str, Any] = {
        "total_rows_read": len(dataframe),
        "valid_rows": 0,
        "invalid_rows": 0,
        "rows_with_warnings": 0,
        "clips_generated": 0,
        "clips_skipped": 0,
        "failures": 0,
        "fallback_reencodes": 0,
        "dry_run": settings.dry_run,
        "clip_extraction_mode": settings.clip_extraction_mode,
        "duration_tolerance_seconds": settings.duration_tolerance_seconds,
    }

    if any(issue.severity == "error" for issue in validation_issues):
        export_validation_report(
            [issue.as_dict() for issue in validation_issues],
            settings.resolved_output_dir,
            settings.validation_report_name,
        )
        export_summary_report(summary, settings.resolved_output_dir, settings.summary_report_name)
        logger.error("Pipeline stopped due to unresolved schema errors.")
        return summary

    logger.info("Resolved column mapping: %s", column_mapping)
    logger.info("Indexed %s video files.", len(video_catalog.videos))
    logger.info("Clip extraction mode: %s", settings.clip_extraction_mode)
    if settings.clip_extraction_mode == "fast_mode":
        logger.warning(
            "fast_mode is approximate. Stream copy may align to keyframes, preserve frozen first frames, or start slightly off the requested timestamp."
        )

    for row in iter_rows(rows, settings):
        matched_video, match_error = video_catalog.match_video(row.video_key_raw)
        duration_seconds = (
            video_catalog.get_duration_seconds(matched_video.source_video_path)
            if matched_video is not None
            else None
        )
        row_issues = validate_row(row, matched_video, duration_seconds, duplicate_keys)
        if match_error and matched_video is None:
            row_issues.append(
                ValidationIssue(
                    row_id=row.row_id,
                    issue_type="missing_source_video",
                    issue_description=match_error,
                    severity="error",
                )
            )

        row_status, row_message = summarize_validation_status(row_issues)
        if row_status == "invalid":
            summary["invalid_rows"] += 1
            validation_issues.extend(row_issues)
            continue

        summary["valid_rows"] += 1
        if row_issues:
            validation_issues.extend(row_issues)
            summary["rows_with_warnings"] += 1

        for window_size in settings.clip_window_sizes:
            raw_start, raw_end = build_clip_time_range(
                row.timestamp_seconds,
                window_size,
                settings.clip_mode,
            )
            clip_start_seconds, clip_end_seconds = clamp_clip_time_range(raw_start, raw_end, duration_seconds)

            if duration_seconds is not None and raw_end > duration_seconds:
                validation_issues.append(
                    ValidationIssue(
                        row_id=row.row_id,
                        issue_type="clip_window_trimmed",
                        issue_description=(
                            f"Window {window_size}s was trimmed to fit source duration {duration_seconds:.3f}s."
                        ),
                        severity="warning",
                    )
                )

            if clip_end_seconds <= clip_start_seconds:
                issue = ValidationIssue(
                    row_id=row.row_id,
                    issue_type="invalid_clip_window",
                    issue_description=(
                        f"Window {window_size}s produced an invalid clip range "
                        f"[{clip_start_seconds:.3f}, {clip_end_seconds:.3f}] after duration clamping."
                    ),
                    severity="error",
                )
                validation_issues.append(issue)
                summary["failures"] += 1
                summary["invalid_rows"] += 1
                continue

            clip_path = build_clip_output_path(
                settings,
                row.row_id,
                str(row.video_key_raw),
                row.timestamp_seconds,
                window_size,
            )
            clip_result = extract_clip(
                matched_video.source_video_path,
                clip_path,
                clip_start_seconds,
                clip_end_seconds,
                settings,
            )

            if clip_result.used_fallback:
                summary["fallback_reencodes"] += 1

            if clip_result.success:
                if clip_result.was_regenerated:
                    summary["clips_generated"] += 1
                else:
                    summary["clips_skipped"] += 1
            else:
                summary["failures"] += 1
                validation_issues.append(
                    ValidationIssue(
                        row_id=row.row_id,
                        issue_type="failed_clip_generation",
                        issue_description=clip_result.message,
                        severity="error",
                    )
                )

            validation_status = row_status
            validation_message = row_message
            if not clip_result.success:
                validation_status = "invalid"
                validation_message = clip_result.message if not row_message else f"{row_message} | {clip_result.message}"
            elif row_message:
                validation_status = "warning"

            dataset_rows.append(
                build_dataset_record(
                    row,
                    matched_video,
                    window_size,
                    str(clip_path),
                    clip_result,
                    validation_status,
                    validation_message,
                )
            )

    export_dataset_rows(
        dataset_rows,
        settings.resolved_output_dir,
        settings.paired_csv_name,
        settings.paired_json_name,
    )
    export_validation_report(
        [issue.as_dict() for issue in validation_issues],
        settings.resolved_output_dir,
        settings.validation_report_name,
    )
    export_summary_report(summary, settings.resolved_output_dir, settings.summary_report_name)

    logger.info("Pipeline complete.")
    logger.info("Successful clips: %s", summary["clips_generated"])
    logger.info("Skipped clips: %s", summary["clips_skipped"])
    logger.info("Failures: %s", summary["failures"])
    logger.info("Fallback re-encodes: %s", summary["fallback_reencodes"])
    return summary


if __name__ == "__main__":
    settings = Settings()
    configure_logging(settings.log_level)
    run_pipeline(settings)
