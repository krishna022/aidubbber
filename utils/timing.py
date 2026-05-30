"""Audio timing utilities – stretch / compress dubbed clips to match original timing."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

from config import SAMPLE_RATE, MAX_TIME_STRETCH_RATIO

log = logging.getLogger(__name__)


def adjust_audio_duration(
    audio_path: str,
    target_duration: float,
    out_path: str,
    sr: int = SAMPLE_RATE,
) -> str:
    """
    Time-stretch `audio_path` to exactly `target_duration` seconds.

    Uses rubberband (via ffmpeg atempo) for high-quality stretching.
    Falls back to librosa if ffmpeg rubberband is unavailable.
    If the required stretch ratio exceeds MAX_TIME_STRETCH_RATIO, the audio
    is not stretched; instead silence is appended / the clip is truncated.
    """
    y, orig_sr = librosa.load(audio_path, sr=sr, mono=True)
    actual_duration = len(y) / sr

    if actual_duration < 0.01:
        _write_silence(out_path, target_duration, sr)
        return out_path

    ratio = actual_duration / target_duration  # > 1 means speed-up needed

    if abs(ratio - 1.0) < 0.02:
        # Already close enough
        sf.write(out_path, y, sr)
        return out_path

    if ratio > MAX_TIME_STRETCH_RATIO or ratio < (1 / MAX_TIME_STRETCH_RATIO):
        log.warning(
            "Stretch ratio %.2f out of range for '%s'; padding/trimming instead.",
            ratio, audio_path,
        )
        y = _pad_or_trim(y, target_duration, sr)
        sf.write(out_path, y, sr)
        return out_path

    # Try ffmpeg atempo (chains filters for ratios outside [0.5, 2.0])
    success = _ffmpeg_atempo(audio_path, out_path, ratio, sr)
    if not success:
        # Fallback: librosa phase vocoder
        y_stretched = librosa.effects.time_stretch(y, rate=ratio)
        y_stretched = _pad_or_trim(y_stretched, target_duration, sr)
        sf.write(out_path, y_stretched, sr)

    return out_path


def merge_audio_segments(
    segments: list[tuple[float, str]],   # (start_seconds, wav_path)
    total_duration: float,
    out_path: str,
    sr: int = SAMPLE_RATE,
) -> str:
    """
    Composite all segment WAV files onto a silent canvas of `total_duration` seconds.
    Overlapping segments are mixed by addition (clipped to [-1, 1]).
    """
    canvas = np.zeros(int(total_duration * sr), dtype=np.float32)

    for start_sec, wav_path in segments:
        try:
            y, _ = librosa.load(wav_path, sr=sr, mono=True)
        except Exception as exc:
            log.warning("Cannot load '%s': %s", wav_path, exc)
            continue

        start_frame = int(start_sec * sr)
        end_frame = start_frame + len(y)

        # Trim if segment runs past canvas
        if end_frame > len(canvas):
            y = y[: len(canvas) - start_frame]
            end_frame = len(canvas)

        if len(y) > 0:
            canvas[start_frame:end_frame] += y

    canvas = np.clip(canvas, -1.0, 1.0)
    sf.write(out_path, canvas, sr)
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffmpeg_atempo(in_path: str, out_path: str, ratio: float, sr: int) -> bool:
    """Use ffmpeg atempo filter to time-stretch audio."""
    # atempo supports [0.5, 100] by chaining
    filters = _build_atempo_chain(ratio)
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-filter:a", filters,
        "-ar", str(sr), "-ac", "1",
        out_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0
    except Exception as exc:
        log.debug("ffmpeg atempo failed: %s", exc)
        return False


def _build_atempo_chain(ratio: float) -> str:
    """Build chained atempo filter string for arbitrary ratios."""
    filters = []
    r = ratio
    while r > 2.0:
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r *= 2.0
    filters.append(f"atempo={r:.6f}")
    return ",".join(filters)


def _pad_or_trim(y: np.ndarray, target_duration: float, sr: int) -> np.ndarray:
    target_frames = int(target_duration * sr)
    if len(y) >= target_frames:
        return y[:target_frames]
    padding = np.zeros(target_frames - len(y), dtype=np.float32)
    return np.concatenate([y, padding])


def _write_silence(out_path: str, duration: float, sr: int) -> None:
    silence = np.zeros(int(duration * sr), dtype=np.float32)
    sf.write(out_path, silence, sr)
