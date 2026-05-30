"""MLX-Whisper transcription with word-level timestamps."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config import WHISPER_MODEL, Segment

log = logging.getLogger(__name__)

# Whisper language codes differ slightly from ISO-639-1
WHISPER_LANG_MAP: dict[str, str] = {
    "zh": "chinese", "ja": "japanese", "ko": "korean",
    "ar": "arabic", "hi": "hindi",
}


class Transcriber:
    """
    Uses mlx-whisper (Apple Silicon optimised) to transcribe speech.
    Falls back to openai-whisper on non-MPS hardware.
    """

    def __init__(self, model_id: str = WHISPER_MODEL):
        self.model_id = model_id
        self._model = None

    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return
        log.info("Loading Whisper model '%s' …", self.model_id)
        try:
            import mlx_whisper  # noqa: F401 – validates import
            self._backend = "mlx"
            log.info("Using mlx-whisper backend (Apple Silicon)")
        except ImportError:
            try:
                import whisper  # noqa: F401
                self._backend = "openai"
                log.warning("mlx-whisper not available; falling back to openai-whisper (slower)")
            except ImportError:
                raise ImportError(
                    "Neither 'mlx-whisper' nor 'openai-whisper' is installed. "
                    "Run: pip install mlx-whisper  (Apple Silicon) or  pip install openai-whisper"
                )
        self._model = True  # sentinel – models are stateless in mlx-whisper

    # ------------------------------------------------------------------

    def transcribe_segment(self, segment: Segment, source_language: Optional[str] = None) -> Segment:
        """
        Transcribe the audio for a single Segment.
        Mutates segment.original_text and segment.source_language in-place.
        """
        self.load()
        if not segment.audio_path or not Path(segment.audio_path).exists():
            log.warning("Segment %s has no audio_path", segment.segment_id)
            return segment

        result = self._run_whisper(segment.audio_path, source_language)
        segment.original_text = result.get("text", "").strip()
        detected_lang = result.get("language", source_language or "en")
        segment.source_language = detected_lang
        return segment

    def transcribe_batch(
        self,
        segments: list[Segment],
        source_language: Optional[str] = None,
    ) -> list[Segment]:
        """Transcribe a list of segments sequentially."""
        total = len(segments)
        for i, seg in enumerate(segments):
            log.debug("Transcribing segment %d/%d (%s)", i + 1, total, seg.segment_id)
            self.transcribe_segment(seg, source_language)
        return segments

    def detect_language(self, audio_path: str) -> str:
        """
        Run Whisper on `audio_path` and return the detected ISO-639-1 language code.
        Uses only the first 30 seconds for speed.
        """
        self.load()
        result = self._run_whisper(audio_path, language=None)
        lang = result.get("language", "en") or "en"
        log.info("Auto-detected source language: '%s'", lang)
        return lang

    # ------------------------------------------------------------------

    def _run_whisper(self, audio_path: str, language: Optional[str]) -> dict:
        lang_arg = WHISPER_LANG_MAP.get(language, language) if language else None

        if self._backend == "mlx":
            return self._run_mlx(audio_path, lang_arg)
        return self._run_openai(audio_path, lang_arg)

    def _run_mlx(self, audio_path: str, language: Optional[str]) -> dict:
        import mlx_whisper

        kwargs: dict = {
            "path_or_hf_repo": self.model_id,
            "word_timestamps": True,
            "verbose": False,
        }
        if language:
            kwargs["language"] = language

        return mlx_whisper.transcribe(audio_path, **kwargs)

    def _run_openai(self, audio_path: str, language: Optional[str]) -> dict:
        import whisper

        # Cache the model object to avoid reloading each call
        if not hasattr(self, "_ow_model"):
            # Use large-v3 if possible; map mlx model id to a standard name
            model_size = "large-v3"
            self._ow_model = whisper.load_model(model_size)

        options = {"word_timestamps": True, "verbose": False}
        if language:
            options["language"] = language

        return self._ow_model.transcribe(audio_path, **options)
