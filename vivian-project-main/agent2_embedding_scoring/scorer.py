from __future__ import annotations



import logging
from pathlib import Path
from typing import Optional

import numpy as np

from config import Settings
from encoders import BaseEncoder

LOGGER = logging.getLogger(__name__)


#  Cache helpers 

def _cache_path(cache_dir: Path, model_name: str, clip_path: Path) -> Path:
    model_dir = cache_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / (clip_path.stem + ".npy")


def load_cached_embedding(cache_dir: Path, model_name: str, clip_path: Path) -> Optional[np.ndarray]:
    path = _cache_path(cache_dir, model_name, clip_path)
    if path.exists():
        try:
            return np.load(str(path))
        except Exception as exc:  
            LOGGER.warning("Could not load cached embedding '%s': %s", path, exc)
    return None


def save_cached_embedding(
    cache_dir: Path, model_name: str, clip_path: Path, embedding: np.ndarray
) -> None:
    path = _cache_path(cache_dir, model_name, clip_path)
    try:
        np.save(str(path), embedding)
    except Exception as exc:  
        LOGGER.warning("Could not save embedding cache to '%s': %s", path, exc)


#  Similarity 

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors.

    Both vectors are expected to be L2-normalised (as produced by the encoders),
    in which case this reduces to a simple dot product.  An explicit
    normalisation guard is retained for safety.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_similarity(a: np.ndarray, b: np.ndarray, metric: str = "cosine") -> float:
    if metric == "cosine":
        return cosine_similarity(a, b)
    raise ValueError(f"Unsupported similarity metric '{metric}'.  Only 'cosine' is currently supported.")


#  Per-row scoring 

def score_row(
    clip_path: Path,
    thought_text: str,
    encoder: BaseEncoder,
    settings: Settings,
) -> tuple[Optional[float], Optional[str], Optional[str]]:
    
    #  Video embedding  
    video_embedding: Optional[np.ndarray] = None
    video_embedding_source: Optional[str] = None

    if settings.cache_embeddings and not settings.overwrite_cache:
        video_embedding = load_cached_embedding(
            settings.resolved_cache_dir, encoder.model_name, clip_path
        )
        if video_embedding is not None:
            video_embedding_source = "cache"

    if video_embedding is None:
        if settings.dry_run:
            return None, None, "dry_run"

        video_embedding = encoder.encode_video(clip_path)
        if video_embedding is None:
            return None, None, f"video_encoding_failed: {clip_path.name}"

        video_embedding_source = "computed"
        if settings.cache_embeddings:
            save_cached_embedding(
                settings.resolved_cache_dir, encoder.model_name, clip_path, video_embedding
            )

    #  Text embedding 
    text_embedding = encoder.encode_text(thought_text)
    if text_embedding is None:
        return None, video_embedding_source, "text_encoding_failed"

    #  Similarity 
    score = compute_similarity(video_embedding, text_embedding, settings.similarity_metric)
    return score, video_embedding_source, None
