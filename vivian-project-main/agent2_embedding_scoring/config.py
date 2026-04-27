from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Settings:
    #  Paths 
    project_root: Path = Path(__file__).resolve().parent.parent
    agent_root: Path = Path(__file__).resolve().parent

    # Input: paired_dataset.csv produced by Agent 1
    paired_dataset_path: Path = Path("agent1_data_preparation/output/paired_dataset.csv")

    # Output directory for Agent 2 artefacts
    output_dir: Path = Path("agent2_embedding_scoring/output")

    #  Model selection 
    # Primary model used for the full dataset run.
    # Supported values: "clip", "imagebind"
    primary_model: str = "clip"

    # CLIP variant.  Supported values: "ViT-B/32", "ViT-B/16", "ViT-L/14"
    clip_model_variant: str = "ViT-B/32"

    # Comparison models evaluated on a small sample subset only.
    # Set model_comparison_sample_size to 0 to skip comparison.
    comparison_models: List[str] = field(default_factory=lambda: ["imagebind"])
    model_comparison_sample_size: int = 20

    #  Video frame sampling 
    # Number of evenly-spaced frames sampled from each clip.
    # CLIP image embeddings for all frames are averaged into one clip embedding.
    frames_per_clip: int = 8

    #  Scoring 
    # Similarity metric.  Currently only "cosine" is supported.
    similarity_metric: str = "cosine"

    #  Input filtering 
    # Skip rows whose Agent 1 validation_status is "invalid".
    skip_invalid_rows: bool = True
    # Skip rows where Agent 1 reports the clip file does not exist on disk.
    skip_missing_clips: bool = True

    #  Performance 
    # Frames are encoded in batches.  Reduce if CPU RAM is constrained.
    frame_batch_size: int = 32

    # Persist computed embeddings to disk so re-runs skip already-processed clips.
    cache_embeddings: bool = True
    embeddings_cache_dirname: str = "embeddings_cache"
    overwrite_cache: bool = False

    #  Output file names 
    similarity_scores_csv_name: str = "similarity_scores.csv"
    similarity_scores_json_name: str = "similarity_scores.json"
    model_comparison_csv_name: str = "model_comparison.csv"
    validation_report_name: str = "validation_report.csv"
    summary_report_name: str = "summary_report.json"

    #  Misc 
    enable_progress_bar: bool = True
    log_level: str = "INFO"
    dry_run: bool = False

    #  Path helpers 
    def resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.project_root / path)

    @property
    def resolved_paired_dataset_path(self) -> Path:
        return self.resolve_path(self.paired_dataset_path)

    @property
    def resolved_output_dir(self) -> Path:
        return self.resolve_path(self.output_dir)

    @property
    def resolved_cache_dir(self) -> Path:
        return self.resolved_output_dir / self.embeddings_cache_dirname
