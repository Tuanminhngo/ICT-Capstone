from __future__ import annotations

"""main.py — Agent 2: Embedding & Scoring pipeline.

Workflow
--------
1.  Load paired_dataset.csv produced by Agent 1.
2.  Pre-flight validate every row (skip invalid / missing-clip rows).
3.  Run model comparison on a small sample:
        - Score ``model_comparison_sample_size`` unique row_ids with both
          the primary model (CLIP) and each comparison model (ImageBind).
        - Export model_comparison.csv.
4.  Score the full dataset with the primary model.
        - For each (row_id, window_size) pair: encode the clip and the
          thought text, compute cosine similarity.
        - Cache video embeddings to disk to speed up re-runs.
5.  Export:
        - similarity_scores.csv  /  similarity_scores.json   (long format)
        - model_comparison.csv
        - validation_report.csv
        - summary_report.json
"""

import logging
import random
from typing import Any, Dict, Iterable, List

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from config import Settings
from data_loader import PairedRow, load_paired_dataset
from encoders import get_encoder
from exporter import (
    ensure_output_directories,
    export_model_comparison,
    export_similarity_scores,
    export_summary_report,
    export_validation_report,
)
from scorer import score_row
from validator import ValidationIssue, summarize_validation_status, validate_row


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def iter_rows(rows: List[Any], desc: str, settings: Settings) -> Iterable[Any]:
    if settings.enable_progress_bar and tqdm is not None:
        return tqdm(rows, desc=desc, unit="row")
    return rows


def _make_score_record(
    row: PairedRow,
    model_name: str,
    similarity_score: Any,
    video_embedding_source: Any,
    scoring_status: str,
    scoring_message: str,
) -> Dict[str, Any]:
    return {
        "row_id": row.row_id,
        "thought_text": row.thought_text,
        "window_size_seconds": row.window_size_seconds,
        "model_name": model_name,
        "similarity_score": similarity_score,
        "video_embedding_source": video_embedding_source,
        "scoring_status": scoring_status,
        "scoring_message": scoring_message,
        "clip_path": str(row.clip_path),
    }


def run_pipeline(settings: Settings) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    ensure_output_directories(settings.resolved_output_dir, settings.resolved_cache_dir)

    #  Load input 
    _, rows, load_issues = load_paired_dataset(settings)
    validation_issues: List[ValidationIssue] = [
        ValidationIssue(
            row_id=issue["row_id"],
            issue_type=issue["issue_type"],
            issue_description=issue["issue_description"],
            severity=issue["severity"],
        )
        for issue in load_issues
    ]

    summary: Dict[str, Any] = {
        "total_rows_loaded": len(rows),
        "rows_skipped": 0,
        "rows_scored": 0,
        "rows_failed": 0,
        "primary_model": settings.primary_model,
        "clip_model_variant": settings.clip_model_variant,
        "frames_per_clip": settings.frames_per_clip,
        "similarity_metric": settings.similarity_metric,
        "model_comparison_sample_size": settings.model_comparison_sample_size,
        "dry_run": settings.dry_run,
        "cache_embeddings": settings.cache_embeddings,
    }

    if any(issue.severity == "error" for issue in validation_issues):
        export_validation_report(
            [i.as_dict() for i in validation_issues],
            settings.resolved_output_dir,
            settings.validation_report_name,
        )
        export_summary_report(summary, settings.resolved_output_dir, settings.summary_report_name)
        logger.error("Pipeline stopped due to input file errors (see validation_report.csv).")
        return summary

    #  Pre-flight validation 
    valid_rows: List[PairedRow] = []
    for row in rows:
        row_issues = validate_row(row, settings.skip_invalid_rows, settings.skip_missing_clips)
        status, _ = summarize_validation_status(row_issues)
        if status == "invalid":
            summary["rows_skipped"] += 1
            validation_issues.extend(row_issues)
        else:
            valid_rows.append(row)
            if row_issues:
                validation_issues.extend(row_issues)

    logger.info(
        "Pre-flight: %d valid rows, %d skipped.", len(valid_rows), summary["rows_skipped"]
    )

    #  Model comparison (optional) 
    comparison_records: List[Dict[str, Any]] = []

    if settings.model_comparison_sample_size > 0 and settings.comparison_models:
        unique_row_ids = list({row.row_id for row in valid_rows})
        sample_ids = set(
            random.sample(
                unique_row_ids,
                min(settings.model_comparison_sample_size, len(unique_row_ids)),
            )
        )
        sample_rows = [r for r in valid_rows if r.row_id in sample_ids]
        logger.info(
            "Model comparison: %d rows × %d models.",
            len(sample_rows),
            len(settings.comparison_models) + 1,
        )

        # Score sample with primary model first
        primary_encoder = get_encoder(settings.primary_model, settings)
        for row in iter_rows(sample_rows, f"Comparison/{settings.primary_model}", settings):
            score, source, reason = score_row(row.clip_path, row.thought_text, primary_encoder, settings)
            status = "scored" if score is not None else ("dry_run" if settings.dry_run else "failed")
            comparison_records.append(
                _make_score_record(row, primary_encoder.model_name, score, source, status, reason or "")
            )

        # Score sample with each comparison model
        for model_name in settings.comparison_models:
            try:
                comp_encoder = get_encoder(model_name, settings)
            except (ImportError, ValueError) as exc:
                logger.warning("Skipping comparison model '%s': %s", model_name, exc)
                validation_issues.append(
                    ValidationIssue(
                        row_id="__model_comparison__",
                        issue_type="comparison_model_unavailable",
                        issue_description=f"Could not load comparison model '{model_name}': {exc}",
                        severity="warning",
                    )
                )
                continue

            for row in iter_rows(sample_rows, f"Comparison/{model_name}", settings):
                score, source, reason = score_row(row.clip_path, row.thought_text, comp_encoder, settings)
                status = "scored" if score is not None else ("dry_run" if settings.dry_run else "failed")
                comparison_records.append(
                    _make_score_record(row, comp_encoder.model_name, score, source, status, reason or "")
                )

        export_model_comparison(
            comparison_records,
            settings.resolved_output_dir,
            settings.model_comparison_csv_name,
        )
        logger.info("Model comparison exported to '%s'.", settings.model_comparison_csv_name)

    #   Full-dataset scoring with primary model 
    # Re-use encoder if already loaded during comparison, otherwise load fresh.
    try:
        primary_encoder  # type: ignore[has-type]
    except NameError:
        primary_encoder = get_encoder(settings.primary_model, settings)

    score_records: List[Dict[str, Any]] = []

    for row in iter_rows(valid_rows, f"Scoring/{settings.primary_model}", settings):
        score, source, reason = score_row(row.clip_path, row.thought_text, primary_encoder, settings)

        if score is not None:
            summary["rows_scored"] += 1
            scoring_status = "scored"
            scoring_message = ""
        elif settings.dry_run:
            scoring_status = "dry_run"
            scoring_message = "dry_run"
        else:
            summary["rows_failed"] += 1
            scoring_status = "failed"
            scoring_message = reason or "unknown_error"
            validation_issues.append(
                ValidationIssue(
                    row_id=row.row_id,
                    issue_type="scoring_failed",
                    issue_description=scoring_message,
                    severity="error",
                )
            )

        score_records.append(
            _make_score_record(
                row,
                primary_encoder.model_name,
                score,
                source,
                scoring_status,
                scoring_message,
            )
        )

    #   Export 
    export_similarity_scores(
        score_records,
        settings.resolved_output_dir,
        settings.similarity_scores_csv_name,
        settings.similarity_scores_json_name,
    )
    export_validation_report(
        [i.as_dict() for i in validation_issues],
        settings.resolved_output_dir,
        settings.validation_report_name,
    )
    export_summary_report(summary, settings.resolved_output_dir, settings.summary_report_name)

    logger.info("Pipeline complete.")
    logger.info("Rows scored:   %d", summary["rows_scored"])
    logger.info("Rows skipped:  %d", summary["rows_skipped"])
    logger.info("Rows failed:   %d", summary["rows_failed"])
    return summary


if __name__ == "__main__":
    settings = Settings()
    configure_logging(settings.log_level)
    run_pipeline(settings)
