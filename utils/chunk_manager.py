"""Audio chunking with overlap and boundary-aware splitting."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from config import SAMPLE_RATE, CHUNK_DURATION, OVERLAP_DURATION

log = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    chunk_index: int
    start: float          # seconds in original file
    end: float
    audio_path: Path      # path to WAV for this chunk

    @property
    def duration(self) -> float:
        return self.end - self.start


class ChunkManager:
    """
    Splits a long WAV file into overlapping chunks and reconstructs
    segment offsets back to original timeline.
    """

    def __init__(
        self,
        audio_path: Path,
        temp_dir: Path,
        chunk_duration: float = CHUNK_DURATION,
        overlap: float = OVERLAP_DURATION,
    ):
        self.audio_path = audio_path
        self.temp_dir = temp_dir
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        info = sf.info(str(audio_path))
        self.total_duration = info.duration
        self.sample_rate = info.samplerate

    # ------------------------------------------------------------------

    def iter_chunks(self) -> Iterator[AudioChunk]:
        """Yield AudioChunk objects covering the full audio."""
        step = self.chunk_duration - self.overlap
        chunk_idx = 0
        start = 0.0

        while start < self.total_duration:
            end = min(start + self.chunk_duration, self.total_duration)
            chunk_path = self.temp_dir / f"chunk_{chunk_idx:04d}.wav"

            if not chunk_path.exists():
                self._extract_chunk(start, end, chunk_path)

            yield AudioChunk(
                chunk_index=chunk_idx,
                start=start,
                end=end,
                audio_path=chunk_path,
            )

            if end >= self.total_duration:
                break
            start += step
            chunk_idx += 1

    def total_chunks(self) -> int:
        step = self.chunk_duration - self.overlap
        return max(1, int(np.ceil((self.total_duration - self.overlap) / step)))

    # ------------------------------------------------------------------

    def _extract_chunk(self, start: float, end: float, out_path: Path) -> None:
        start_frame = int(start * self.sample_rate)
        end_frame = int(end * self.sample_rate)

        audio, sr = sf.read(str(self.audio_path), start=start_frame, stop=end_frame, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # force mono

        sf.write(str(out_path), audio, sr)
        log.debug("Chunk %s: %.1f–%.1f s written", out_path.name, start, end)

    # ------------------------------------------------------------------
    @staticmethod
    def segment_to_global_time(segment_start: float, chunk: AudioChunk) -> float:
        """Convert chunk-relative timestamp → global audio timestamp."""
        return chunk.start + segment_start

    @staticmethod
    def is_boundary_segment(seg_end: float, chunk: AudioChunk, tolerance: float = 1.5) -> bool:
        """True if segment falls within `tolerance` seconds of chunk boundary."""
        return abs(seg_end - (chunk.end - chunk.start)) < tolerance

    @staticmethod
    def deduplicate_segments(segments: list, overlap: float = OVERLAP_DURATION) -> list:
        """
        Remove duplicate segments produced by overlapping chunks.
        Keeps the segment whose chunk-local start is earliest (less likely to be cut).
        """
        seen: dict[str, object] = {}
        result = []
        for seg in sorted(segments, key=lambda s: s.start):
            key = f"{seg.speaker_id}_{seg.start:.2f}"
            if key not in seen:
                seen[key] = seg
                result.append(seg)
        return result
