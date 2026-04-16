from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Settings:
    project_root: Path = Path(__file__).resolve().parent.parent
    agent_root: Path = Path(__file__).resolve().parent
    excel_path: Path = Path("thought_reports.xlsx")
    video_directories: Dict[str, Path] = field(
        default_factory=lambda: {
            "Video 1": Path("Video 1"),
            "Video 2": Path("Video 2"),
        }
    )
    output_dir: Path = Path("agent1_data_preparation/output")
    clips_dirname: str = "clips"
    clip_window_sizes: List[int] = field(default_factory=lambda: [5, 10, 15])
    clip_mode: str = "preceding"
    clip_extraction_mode: str = "accurate_mode"
    fast_mode_warning_enabled: bool = True
    video_extensions: List[str] = field(
        default_factory=lambda: [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]
    )
    overwrite_existing_clips: bool = False
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    reset_timestamps: bool = True
    duration_tolerance_seconds: float = 0.5
    short_clip_threshold_seconds: float = 1.0
    enable_reencode_fallback: bool = True
    validate_clip_playback: bool = True
    dry_run: bool = False
    enable_progress_bar: bool = True
    extract_debug_frame: bool = False
    excel_sheet_name: Optional[str] = None

    column_mapping: Dict[str, Optional[str]] = field(
        default_factory=lambda: {
            "row_id": "uuid",
            "video": "video",
            "timestamp": "time_in_video",
            "thought_text": "thought",
        }
    )
    column_aliases: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "row_id": ["uuid", "id", "row_id", "report_id"],
            "video": ["video_id", "video", "video_name", "source_video", "media_name"],
            "timestamp": ["timestamp", "time", "probe_time", "report_time", "time_in_video"],
            "thought_text": ["thought", "thought_text", "report_text", "response", "text"],
        }
    )

    validation_report_name: str = "validation_report.csv"
    paired_csv_name: str = "paired_dataset.csv"
    paired_json_name: str = "paired_dataset.json"
    summary_report_name: str = "summary_report.json"
    log_level: str = "INFO"

    def resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.project_root / path)

    @property
    def resolved_excel_path(self) -> Path:
        return self.resolve_path(self.excel_path)

    @property
    def resolved_output_dir(self) -> Path:
        return self.resolve_path(self.output_dir)
