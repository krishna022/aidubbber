"""Speaker Embedding Dictionary – central store for per-speaker voice profiles."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import librosa

log = logging.getLogger(__name__)


@dataclass
class SpeakerProfile:
    speaker_id: str
    reference_audio_path: str          # Best reference clip path
    all_segment_paths: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None   # Serialisable numpy vector
    total_speech_duration: float = 0.0
    estimated_gender: Optional[str] = None
    estimated_pitch_hz: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def embedding_array(self) -> Optional[np.ndarray]:
        if self.embedding is None:
            return None
        return np.array(self.embedding, dtype=np.float32)

    def set_embedding(self, arr: np.ndarray) -> None:
        self.embedding = arr.tolist()


class SpeakerRegistry:
    """
    Thread-safe registry mapping speaker_id → SpeakerProfile.

    Persists to JSON on disk so the registry survives crashes.
    """

    def __init__(self, registry_path: Path):
        self._path = registry_path
        self._profiles: dict[str, SpeakerProfile] = {}
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_or_update(
        self,
        speaker_id: str,
        segment_audio_path: str,
        duration: float,
    ) -> SpeakerProfile:
        with self._lock:
            if speaker_id not in self._profiles:
                self._profiles[speaker_id] = SpeakerProfile(
                    speaker_id=speaker_id,
                    reference_audio_path=segment_audio_path,
                )
                log.info("Registry: new speaker '%s'", speaker_id)

            profile = self._profiles[speaker_id]
            profile.all_segment_paths.append(segment_audio_path)
            profile.total_speech_duration += duration

            # Keep the longest clean clip as reference (up to 20 s)
            if (
                duration > self._ref_duration(profile.reference_audio_path)
                and duration <= 20.0
            ):
                profile.reference_audio_path = segment_audio_path

            return profile

    def get(self, speaker_id: str) -> Optional[SpeakerProfile]:
        with self._lock:
            return self._profiles.get(speaker_id)

    def all_speakers(self) -> list[str]:
        with self._lock:
            return list(self._profiles.keys())

    def update_embedding(self, speaker_id: str, embedding: np.ndarray) -> None:
        with self._lock:
            if speaker_id in self._profiles:
                self._profiles[speaker_id].set_embedding(embedding)

    def analyse_speakers(self) -> None:
        """Estimate pitch and gender for every registered speaker."""
        with self._lock:
            for sid, profile in self._profiles.items():
                try:
                    y, sr = librosa.load(profile.reference_audio_path, sr=None, mono=True)
                    f0, _, _ = librosa.pyin(
                        y, fmin=librosa.note_to_hz("C2"),
                        fmax=librosa.note_to_hz("C7"),
                    )
                    valid = f0[~np.isnan(f0)]
                    if len(valid):
                        pitch = float(np.median(valid))
                        profile.estimated_pitch_hz = pitch
                        profile.estimated_gender = "female" if pitch > 165 else "male"
                    log.info(
                        "Speaker '%s': pitch=%.1f Hz, gender=%s",
                        sid, profile.estimated_pitch_hz or 0, profile.estimated_gender,
                    )
                except Exception as exc:
                    log.warning("Could not analyse speaker '%s': %s", sid, exc)
            self._save()

    def save(self) -> None:
        with self._lock:
            self._save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ref_duration(path: str) -> float:
        try:
            info = sf.info(path)
            return info.duration
        except Exception:
            return 0.0

    def _save(self) -> None:
        data = {sid: asdict(p) for sid, p in self._profiles.items()}
        self._path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for sid, raw in data.items():
                self._profiles[sid] = SpeakerProfile(**raw)
            log.info("Registry: loaded %d speakers from %s", len(self._profiles), self._path)
        except Exception as exc:
            log.warning("Could not load registry: %s", exc)
