"""Extract and normalise audio from a video file using FFmpeg."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import soundfile as sf

from config import SAMPLE_RATE, CHANNELS

log = logging.getLogger(__name__)


class AudioExtractor:
    """Extracts 16 kHz mono WAV from any video/audio container."""

    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir

    # ------------------------------------------------------------------

    def extract(self, video_path: Path, out_name: str = "full_audio.wav") -> Path:
        """
        Extract audio track from `video_path` as 16 kHz mono WAV.
        Returns path to the output WAV.
        """
        out_path = self.temp_dir / out_name
        if out_path.exists():
            log.info("Audio already extracted: %s", out_path)
            return out_path

        log.info("Extracting audio from '%s' …", video_path.name)
        self._run_ffmpeg(str(video_path), str(out_path))
        self._verify(out_path)
        return out_path

    def extract_segment(
        self,
        audio_path: Path,
        start: float,
        end: float,
        out_path: Path,
    ) -> Path:
        """Slice a portion of an audio file."""
        duration = end - start
        if duration <= 0:
            raise ValueError(f"Invalid segment: start={start}, end={end}")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-t", str(duration),
            "-i", str(audio_path),
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-c:a", "pcm_s16le",
            str(out_path),
        ]
        _run(cmd, "segment extraction")
        return out_path

    # ------------------------------------------------------------------

    def _run_ffmpeg(self, in_path: str, out_path: str) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", in_path,
            "-vn",                       # no video
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            out_path,
        ]
        _run(cmd, "audio extraction")

    @staticmethod
    def _verify(path: Path) -> None:
        info = sf.info(str(path))
        log.info(
            "Audio: %.1f s, %d Hz, %d ch → %s",
            info.duration, info.samplerate, info.channels, path.name,
        )
        if info.samplerate != SAMPLE_RATE:
            raise RuntimeError(
                f"Expected {SAMPLE_RATE} Hz but got {info.samplerate} Hz"
            )


# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> None:
    log.debug("Running %s: %s", label, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg {label} failed (rc={result.returncode}):\n{result.stderr[-2000:]}"
        )
