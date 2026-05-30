"""Coqui XTTS v2 voice cloner with per-speaker reference audio lookup."""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from config import TTS_MODEL, XTTS_LANG_CODES, TEMP_DIR, SAMPLE_RATE, Segment
from utils.speaker_registry import SpeakerRegistry
from utils.timing import adjust_audio_duration

log = logging.getLogger(__name__)

# XTTS hard limit is 400 GPT-2 tokens.
# For Hindi (Devanagari), ~200 chars is the safe character ceiling.
_XTTS_MAX_CHARS = 190


class VoiceCloner:
    """
    Synthesises dubbed audio using Coqui XTTS v2.

    Handles XTTS's 400-token limit by chunking long translated texts,
    generating audio per chunk, and concatenating before time-stretching.
    """

    def __init__(
        self,
        registry: SpeakerRegistry,
        target_lang: str,
        temp_dir: Path = TEMP_DIR,
        tts_model: str = TTS_MODEL,
    ):
        self.registry = registry
        self.target_lang = target_lang
        self.xtts_lang = XTTS_LANG_CODES.get(target_lang, target_lang)
        self.temp_dir = temp_dir
        self.tts_model = tts_model
        self._tts = None

    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._tts is not None:
            return
        log.info("Loading XTTS v2 model …")
        import os
        from TTS.api import TTS

        os.environ["COQUI_TOS_AGREED"] = "1"
        logging.getLogger("TTS").setLevel(logging.WARNING)
        self._tts = TTS(self.tts_model, progress_bar=False)
        log.info("XTTS v2 loaded (lang=%s)", self.xtts_lang)

    # ------------------------------------------------------------------

    def clone_segment(self, segment: Segment) -> Segment:
        self.load()

        text = (segment.translated_text or "").strip()
        if not text:
            log.warning("Segment %s has no translated text; skipping TTS", segment.segment_id)
            return segment

        profile = self.registry.get(segment.speaker_id)
        if profile is None:
            log.warning("No profile for speaker '%s'; skipping TTS", segment.speaker_id)
            return segment

        reference_path = profile.reference_audio_path
        if not Path(reference_path).exists():
            log.error("Reference audio missing: %s", reference_path)
            return segment

        raw_out = self.temp_dir / f"tts_{segment.segment_id}.wav"
        final_out = self.temp_dir / f"dubbed_{segment.segment_id}.wav"

        try:
            self._run_xtts_chunked(text, reference_path, str(raw_out))
        except Exception as exc:
            log.error("XTTS failed for segment %s: %s", segment.segment_id, exc)
            return segment

        # Normalize raw audio loudness before time-stretching
        _normalize_wav(str(raw_out), str(raw_out))

        # Stretch / compress to match original segment timing
        adjust_audio_duration(str(raw_out), segment.duration, str(final_out))

        segment.dubbed_audio_path = str(final_out)
        log.debug(
            "Cloned %s  src=%.1fs  target=%.1fs",
            segment.segment_id, _wav_duration(str(raw_out)), segment.duration,
        )
        return segment

    def clone_batch(self, segments: list[Segment]) -> list[Segment]:
        for i, seg in enumerate(segments):
            log.info("Voice cloning %d/%d: %s", i + 1, len(segments), seg.segment_id)
            self.clone_segment(seg)
        return segments

    # ------------------------------------------------------------------
    # Text chunking — fixes XTTS 400-token / 250-char hard limit
    # ------------------------------------------------------------------

    def _split_text_for_xtts(self, text: str) -> list[str]:
        """
        Split translated text into chunks that fit within XTTS's token limit.
        Splits on sentence boundaries (।  .  !  ?) first, then on commas.
        """
        # Primary split: sentence endings (includes Devanagari danda ।)
        sentences = re.split(r"(?<=[।.!?])\s+", text.strip())

        chunks: list[str] = []
        buf = ""
        for sent in sentences:
            # If adding this sentence keeps us under limit, accumulate
            if len(buf) + len(sent) + 1 <= _XTTS_MAX_CHARS:
                buf = (buf + " " + sent).strip()
            else:
                if buf:
                    chunks.append(buf)
                # Single sentence still too long → split on commas
                if len(sent) > _XTTS_MAX_CHARS:
                    sub_parts = re.split(r"(?<=[,;،])\s+", sent)
                    sub_buf = ""
                    for part in sub_parts:
                        if len(sub_buf) + len(part) + 1 <= _XTTS_MAX_CHARS:
                            sub_buf = (sub_buf + " " + part).strip()
                        else:
                            if sub_buf:
                                chunks.append(sub_buf)
                            # Force-split by character if still too long
                            while len(part) > _XTTS_MAX_CHARS:
                                chunks.append(part[:_XTTS_MAX_CHARS])
                                part = part[_XTTS_MAX_CHARS:]
                            sub_buf = part
                    if sub_buf:
                        chunks.append(sub_buf)
                    buf = ""
                else:
                    buf = sent

        if buf:
            chunks.append(buf)

        return [c for c in chunks if c.strip()]

    # ------------------------------------------------------------------

    def _run_xtts_chunked(self, text: str, reference_wav: str, out_path: str) -> None:
        """
        Generate TTS audio for text, splitting into XTTS-safe chunks if needed,
        then concatenate all chunks with a short silence between them.
        """
        chunks = self._split_text_for_xtts(text)
        log.debug("XTTS: %d chunk(s) for %d chars", len(chunks), len(text))

        if len(chunks) == 1:
            self._tts.tts_to_file(
                text=chunks[0],
                speaker_wav=reference_wav,
                language=self.xtts_lang,
                file_path=out_path,
            )
            return

        # Multiple chunks → generate each, concatenate
        parts: list[np.ndarray] = []
        sr = SAMPLE_RATE
        silence = np.zeros(int(sr * 0.15), dtype=np.float32)  # 150 ms gap

        with tempfile.TemporaryDirectory() as tmp:
            for i, chunk in enumerate(chunks):
                chunk_path = f"{tmp}/chunk_{i:03d}.wav"
                self._tts.tts_to_file(
                    text=chunk,
                    speaker_wav=reference_wav,
                    language=self.xtts_lang,
                    file_path=chunk_path,
                )
                audio, sr = sf.read(chunk_path, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                parts.append(audio)
                if i < len(chunks) - 1:
                    parts.append(silence)

        combined = np.concatenate(parts).astype(np.float32)
        sf.write(out_path, combined, sr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_wav(in_path: str, out_path: str, target_peak: float = 0.92) -> None:
    """Peak-normalize audio to target_peak to ensure consistent loudness."""
    try:
        audio, sr = sf.read(in_path, dtype="float32")
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio * (target_peak / peak)
        sf.write(out_path, audio, sr)
    except Exception as exc:
        log.debug("Normalization skipped: %s", exc)


def _wav_duration(path: str) -> float:
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        return 0.0
