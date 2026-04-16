from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import Settings


LOGGER = logging.getLogger(__name__)


@dataclass
class VideoMatch:
    video_group: str
    source_video_file: str
    source_video_path: Path


@dataclass
class ClipResult:
    success: bool
    message: str
    extraction_mode_used: str
    clip_exists: bool
    was_regenerated: bool
    expected_duration_seconds: float
    actual_duration_seconds: Optional[float]
    duration_within_tolerance: Optional[bool]
    playback_valid: Optional[bool]
    used_fallback: bool
    clip_start_seconds: float
    clip_end_seconds: float
    debug_frame_path: Optional[str] = None


class VideoCatalog:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._videos: List[VideoMatch] = []
        self._duration_cache: Dict[Path, Optional[float]] = {}
        self._build_index()

    def _build_index(self) -> None:
        valid_extensions = {ext.lower() for ext in self.settings.video_extensions}
        for group_name, directory in sorted(self.settings.video_directories.items()):
            resolved_dir = self.settings.resolve_path(directory)
            if not resolved_dir.exists():
                LOGGER.warning("Video directory does not exist: %s", resolved_dir)
                continue

            for path in sorted(resolved_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in valid_extensions:
                    continue
                self._videos.append(
                    VideoMatch(
                        video_group=group_name,
                        source_video_file=path.name,
                        source_video_path=path.resolve(),
                    )
                )

    @property
    def videos(self) -> List[VideoMatch]:
        return self._videos

    def match_video(self, raw_video_key: object) -> Tuple[Optional[VideoMatch], Optional[str]]:
        if raw_video_key is None:
            return None, "Video key is missing."

        video_key = str(raw_video_key).strip()
        if not video_key:
            return None, "Video key is empty."

        requested_norm = normalize_video_key(video_key)
        exact_matches = [
            video
            for video in self._videos
            if normalize_video_key(video.source_video_file) == requested_norm
            or normalize_video_key(video.source_video_path.stem) == requested_norm
        ]
        if len(exact_matches) == 1:
            return exact_matches[0], None
        if len(exact_matches) > 1:
            return None, f"Multiple exact video matches found for '{video_key}'."

        partial_matches = [
            video
            for video in self._videos
            if requested_norm in normalize_video_key(video.source_video_file)
            or requested_norm in normalize_video_key(video.source_video_path.stem)
        ]
        if len(partial_matches) == 1:
            return partial_matches[0], None
        if len(partial_matches) > 1:
            return None, f"Multiple partial video matches found for '{video_key}'."

        return None, f"No matching video found for '{video_key}'."

    def get_duration_seconds(self, media_path: Path) -> Optional[float]:
        if media_path in self._duration_cache:
            return self._duration_cache[media_path]

        duration = probe_media_duration_seconds(media_path, self.settings)
        self._duration_cache[media_path] = duration
        return duration


def normalize_video_key(value: str) -> str:
    stem = Path(value).stem
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def build_clip_time_range(
    timestamp_seconds: float,
    window_size_seconds: int,
    clip_mode: str,
) -> Tuple[float, float]:
    if clip_mode == "preceding":
        return max(0.0, timestamp_seconds - window_size_seconds), max(0.0, timestamp_seconds)

    if clip_mode == "centered":
        half = window_size_seconds / 2
        return max(0.0, timestamp_seconds - half), max(0.0, timestamp_seconds + half)

    raise ValueError(f"Unsupported clip mode: {clip_mode}")


def clamp_clip_time_range(
    clip_start_seconds: float,
    clip_end_seconds: float,
    video_duration_seconds: Optional[float],
) -> Tuple[float, float]:
    start = max(0.0, clip_start_seconds)
    end = max(start, clip_end_seconds)
    if video_duration_seconds is not None:
        end = min(end, video_duration_seconds)
        start = min(start, end)
    return start, end


def safe_filename_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned[:120] or "unknown"


def build_clip_output_path(
    settings: Settings,
    row_id: str,
    video_key: str,
    timestamp_seconds: float,
    window_size_seconds: int,
) -> Path:
    clip_dir = settings.resolved_output_dir / settings.clips_dirname / f"{window_size_seconds}s"
    clip_dir.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"row_{safe_filename_component(row_id)}"
        f"_video_{safe_filename_component(video_key)}"
        f"_t_{int(round(timestamp_seconds * 1000))}"
        f"_w_{window_size_seconds}.mp4"
    )
    return clip_dir / file_name


def run_subprocess(command: List[str]) -> Tuple[bool, str]:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True, (result.stderr or result.stdout or "").strip()
    except FileNotFoundError as exc:
        return False, str(exc)
    except subprocess.CalledProcessError as exc:
        return False, (exc.stderr or exc.stdout or str(exc)).strip()


def build_ffmpeg_command(
    source_video_path: Path,
    clip_output_path: Path,
    clip_start_seconds: float,
    clip_duration_seconds: float,
    settings: Settings,
    extraction_mode: str,
) -> List[str]:
    overwrite_flag = "-y" if settings.overwrite_existing_clips else "-n"
    command = [settings.ffmpeg_binary, overwrite_flag]

    if extraction_mode == "fast_mode":
        command.extend(
            [
                "-ss",
                f"{clip_start_seconds:.3f}",
                "-i",
                str(source_video_path),
                "-t",
                f"{clip_duration_seconds:.3f}",
                "-c",
                "copy",
                "-map",
                "0",
            ]
        )
    elif extraction_mode == "accurate_mode":
        command.extend(
            [
                "-i",
                str(source_video_path),
                "-ss",
                f"{clip_start_seconds:.3f}",
                "-t",
                f"{clip_duration_seconds:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
            ]
        )
    else:
        raise ValueError(f"Unsupported clip extraction mode: {extraction_mode}")

    if settings.reset_timestamps:
        command.extend(["-reset_timestamps", "1", "-avoid_negative_ts", "make_zero"])

    command.append(str(clip_output_path))
    return command


def probe_media_duration_seconds(media_path: Path, settings: Settings) -> Optional[float]:
    command = [
        settings.ffprobe_binary,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(media_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout or "{}")
        return float(payload["format"]["duration"])
    except FileNotFoundError:
        LOGGER.error("ffprobe binary '%s' was not found.", settings.ffprobe_binary)
        return None
    except (subprocess.CalledProcessError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("Failed to probe media duration for %s: %s", media_path, exc)
        return None


def validate_clip_duration(
    clip_output_path: Path,
    expected_duration_seconds: float,
    settings: Settings,
) -> Tuple[Optional[float], Optional[bool], str]:
    actual_duration_seconds = probe_media_duration_seconds(clip_output_path, settings)
    if actual_duration_seconds is None:
        return None, None, "Could not measure generated clip duration."

    deviation = abs(actual_duration_seconds - expected_duration_seconds)
    within_tolerance = deviation <= settings.duration_tolerance_seconds
    if within_tolerance:
        return actual_duration_seconds, True, f"Clip duration validated at {actual_duration_seconds:.3f}s."

    return (
        actual_duration_seconds,
        False,
        (
            f"Clip duration {actual_duration_seconds:.3f}s deviates from expected "
            f"{expected_duration_seconds:.3f}s by {deviation:.3f}s."
        ),
    )


def validate_clip_playback(clip_output_path: Path, settings: Settings) -> Tuple[Optional[bool], str]:
    if not settings.validate_clip_playback:
        return None, "Clip playback validation disabled."

    command = [
        settings.ffmpeg_binary,
        "-v",
        "error",
        "-i",
        str(clip_output_path),
        "-f",
        "null",
        "-",
    ]
    success, message = run_subprocess(command)
    if success:
        return True, "Clip playback validation passed."
    return False, message or "Clip playback validation failed."


def maybe_extract_debug_frame(clip_output_path: Path, settings: Settings) -> Optional[str]:
    if not settings.extract_debug_frame or not clip_output_path.exists():
        return None

    debug_frame_path = clip_output_path.with_suffix(".jpg")
    command = [
        settings.ffmpeg_binary,
        "-y",
        "-i",
        str(clip_output_path),
        "-frames:v",
        "1",
        str(debug_frame_path),
    ]
    success, _ = run_subprocess(command)
    return str(debug_frame_path) if success and debug_frame_path.exists() else None


def should_retry_with_accurate_mode(
    extraction_mode: str,
    clip_duration_seconds: float,
    duration_within_tolerance: Optional[bool],
    playback_valid: Optional[bool],
    settings: Settings,
) -> bool:
    if extraction_mode != "fast_mode" or not settings.enable_reencode_fallback:
        return False
    if clip_duration_seconds <= settings.short_clip_threshold_seconds:
        return True
    if duration_within_tolerance is False:
        return True
    if playback_valid is False:
        return True
    return False


def perform_clip_generation(
    source_video_path: Path,
    clip_output_path: Path,
    clip_start_seconds: float,
    clip_end_seconds: float,
    settings: Settings,
    extraction_mode: str,
) -> ClipResult:
    expected_duration_seconds = max(0.0, clip_end_seconds - clip_start_seconds)
    if expected_duration_seconds <= 0:
        return ClipResult(
            success=False,
            message="Clip duration is zero or negative.",
            extraction_mode_used=extraction_mode,
            clip_exists=False,
            was_regenerated=False,
            expected_duration_seconds=expected_duration_seconds,
            actual_duration_seconds=None,
            duration_within_tolerance=None,
            playback_valid=None,
            used_fallback=False,
            clip_start_seconds=clip_start_seconds,
            clip_end_seconds=clip_end_seconds,
        )

    command = build_ffmpeg_command(
        source_video_path,
        clip_output_path,
        clip_start_seconds,
        expected_duration_seconds,
        settings,
        extraction_mode,
    )
    success, message = run_subprocess(command)
    if not success:
        LOGGER.error("ffmpeg failed for %s: %s", clip_output_path, message)
        return ClipResult(
            success=False,
            message=message,
            extraction_mode_used=extraction_mode,
            clip_exists=clip_output_path.exists(),
            was_regenerated=True,
            expected_duration_seconds=expected_duration_seconds,
            actual_duration_seconds=None,
            duration_within_tolerance=None,
            playback_valid=None,
            used_fallback=False,
            clip_start_seconds=clip_start_seconds,
            clip_end_seconds=clip_end_seconds,
        )

    actual_duration_seconds, duration_within_tolerance, duration_message = validate_clip_duration(
        clip_output_path,
        expected_duration_seconds,
        settings,
    )
    playback_valid, playback_message = validate_clip_playback(clip_output_path, settings)
    debug_frame_path = maybe_extract_debug_frame(clip_output_path, settings)
    combined_message = " | ".join(
        part
        for part in [message, duration_message, playback_message]
        if part
    )
    return ClipResult(
        success=(
            clip_output_path.exists()
            and duration_within_tolerance is not False
            and playback_valid is not False
        ),
        message=combined_message,
        extraction_mode_used=extraction_mode,
        clip_exists=clip_output_path.exists(),
        was_regenerated=True,
        expected_duration_seconds=expected_duration_seconds,
        actual_duration_seconds=actual_duration_seconds,
        duration_within_tolerance=duration_within_tolerance,
        playback_valid=playback_valid,
        used_fallback=False,
        clip_start_seconds=clip_start_seconds,
        clip_end_seconds=clip_end_seconds,
        debug_frame_path=debug_frame_path,
    )


def extract_clip(
    source_video_path: Path,
    clip_output_path: Path,
    clip_start_seconds: float,
    clip_end_seconds: float,
    settings: Settings,
) -> ClipResult:
    expected_duration_seconds = max(0.0, clip_end_seconds - clip_start_seconds)
    if clip_output_path.exists() and not settings.overwrite_existing_clips:
        actual_duration_seconds, duration_within_tolerance, duration_message = validate_clip_duration(
            clip_output_path,
            expected_duration_seconds,
            settings,
        )
        playback_valid, playback_message = validate_clip_playback(clip_output_path, settings)
        return ClipResult(
            success=(duration_within_tolerance is not False and playback_valid is not False),
            message=" | ".join(
                part
                for part in [
                    "Clip already exists; skipped regeneration.",
                    duration_message,
                    playback_message,
                ]
                if part
            ),
            extraction_mode_used=settings.clip_extraction_mode,
            clip_exists=True,
            was_regenerated=False,
            expected_duration_seconds=expected_duration_seconds,
            actual_duration_seconds=actual_duration_seconds,
            duration_within_tolerance=duration_within_tolerance,
            playback_valid=playback_valid,
            used_fallback=False,
            clip_start_seconds=clip_start_seconds,
            clip_end_seconds=clip_end_seconds,
            debug_frame_path=maybe_extract_debug_frame(clip_output_path, settings),
        )

    if settings.dry_run:
        return ClipResult(
            success=True,
            message="Dry run enabled; clip generation skipped.",
            extraction_mode_used=settings.clip_extraction_mode,
            clip_exists=False,
            was_regenerated=False,
            expected_duration_seconds=expected_duration_seconds,
            actual_duration_seconds=None,
            duration_within_tolerance=None,
            playback_valid=None,
            used_fallback=False,
            clip_start_seconds=clip_start_seconds,
            clip_end_seconds=clip_end_seconds,
        )

    if settings.clip_extraction_mode == "fast_mode" and settings.fast_mode_warning_enabled:
        LOGGER.warning(
            "Using fast_mode for %s. Stream copy is approximate and may align to keyframes instead of exact timestamps.",
            clip_output_path.name,
        )

    first_result = perform_clip_generation(
        source_video_path,
        clip_output_path,
        clip_start_seconds,
        clip_end_seconds,
        settings,
        settings.clip_extraction_mode,
    )
    if first_result.success:
        LOGGER.info("Generated clip %s with %s", clip_output_path, first_result.extraction_mode_used)
        return first_result

    if should_retry_with_accurate_mode(
        settings.clip_extraction_mode,
        expected_duration_seconds,
        first_result.duration_within_tolerance,
        first_result.playback_valid,
        settings,
    ):
        LOGGER.warning(
            "Retrying %s in accurate_mode after fast_mode validation failure or short-clip risk.",
            clip_output_path.name,
        )
        retry_result = perform_clip_generation(
            source_video_path,
            clip_output_path,
            clip_start_seconds,
            clip_end_seconds,
            settings,
            "accurate_mode",
        )
        retry_result.used_fallback = True
        if retry_result.success:
            LOGGER.info("Generated clip %s with accurate_mode fallback", clip_output_path)
        return retry_result

    return first_result
