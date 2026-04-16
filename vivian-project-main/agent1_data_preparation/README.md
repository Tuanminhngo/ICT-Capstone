# Agent 1 Data Preparation

Production-oriented Python pipeline for preparing paired video-thought-report data for downstream multimodal processing.

## What it does

- Reads `thought_reports.xlsx` with configurable column mapping.
- Resolves source videos across `Video 1/` and `Video 2/`.
- Parses timestamps into seconds.
- Validates rows and records data quality issues.
- Extracts configurable video clips with `ffmpeg`.
- Re-validates generated clip duration and playback health.
- Exports `paired_dataset.csv`, `paired_dataset.json`, `validation_report.csv`, and `summary_report.json`.

## Expected project layout

```text
Vivian Project/
├── Video 1/
├── Video 2/
├── thought_reports.xlsx
└── agent1_data_preparation/
    ├── main.py
    ├── config.py
    ├── data_loader.py
    ├── video_utils.py
    ├── validator.py
    ├── exporter.py
    ├── requirements.txt
    ├── README.md
    └── output/
        ├── clips/
        │   ├── 5s/
        │   ├── 10s/
        │   └── 15s/
        ├── paired_dataset.csv
        ├── paired_dataset.json
        └── validation_report.csv
```

## Install

```bash
cd agent1_data_preparation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pipeline also requires `ffmpeg` and `ffprobe` to be installed and available on `PATH`.

## Run

```bash
cd agent1_data_preparation
python3 main.py
```

## Default workbook mapping

The current defaults match the provided workbook:

- `row_id` -> `uuid`
- `video` -> `video`
- `timestamp` -> `time_in_video`
- `thought_text` -> `thought`

You can change these in `config.py`.

## Clipping modes

The pipeline supports two extraction modes:

- `accurate_mode`: slower but frame-accurate. This re-encodes with `libx264` and is the default.
- `fast_mode`: faster but approximate. This uses stream copy and may align to keyframes instead of the exact requested timestamp.

Important limitation:

- `fast_mode` can produce slightly misaligned start times or frozen first frames because stream copy is keyframe-based.
- When `enable_reencode_fallback` is enabled, the pipeline retries risky or invalid `fast_mode` clips in `accurate_mode`.
- Generated clips are validated after export against an expected duration with a tolerance controlled by `duration_tolerance_seconds`.
- If `reset_timestamps` is enabled, clips are normalized to start near time `0`.

Useful settings in `config.py`:

- `clip_extraction_mode`
- `duration_tolerance_seconds`
- `enable_reencode_fallback`
- `validate_clip_playback`
- `dry_run`
- `enable_progress_bar`
- `extract_debug_frame`

## Validation behavior

- Invalid timestamps are logged and skipped without stopping the run.
- Missing or ambiguous video matches are written to the validation report.
- Duplicate rows are flagged as warnings.
- Timestamps beyond video duration are marked invalid.
- Failed clip generation is recorded as an error.
- Trimmed clip windows are logged as warnings.
- Validation output columns are `row_id`, `issue_type`, `issue_description`, and `severity`.

## Notes

- Clip windows default to `5`, `10`, and `15` seconds using the range `[t-window, t]`.
- Existing clips are reused unless `overwrite_existing_clips` is enabled.
- Video metadata is cached to avoid repeated `ffprobe` calls.
- Processing order is deterministic because rows and video file discovery are handled in a stable order.
