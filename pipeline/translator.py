"""NLLB-200 translation with sentence-level batching and context retention."""
from __future__ import annotations

import logging
import re
from typing import Optional

from config import NLLB_MODEL, NLLB_LANG_CODES, DEVICE, Segment

log = logging.getLogger(__name__)

MAX_TOKENS = 512          # NLLB token limit per call
CONTEXT_WINDOW = 3        # previous sentences carried as context


class Translator:
    """
    Translates Segment.original_text using NLLB-200 via HuggingFace Transformers.
    Runs on MPS (Apple Silicon) when available.
    """

    def __init__(
        self,
        model_id: str = NLLB_MODEL,
        source_lang: str = "en",
        target_lang: str = "es",
    ):
        self.model_id = model_id
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._src_code = NLLB_LANG_CODES.get(source_lang, "eng_Latn")
        self._tgt_code = NLLB_LANG_CODES.get(target_lang, "spa_Latn")
        self._model = None
        self._tokenizer = None
        self._context: list[str] = []   # recent sentences for context

    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return
        log.info("Loading NLLB model '%s' …", self.model_id)
        import torch
        from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

        self._tokenizer = NllbTokenizer.from_pretrained(
            self.model_id, src_lang=self._src_code
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)

        device_str = DEVICE
        if device_str == "mps":
            self._model = self._model.to(torch.device("mps"))
        elif device_str == "cuda":
            self._model = self._model.cuda()

        self._model.eval()
        log.info("NLLB loaded on device=%s (%s → %s)", device_str, self._src_code, self._tgt_code)

    # ------------------------------------------------------------------

    def translate_segment(self, segment: Segment) -> Segment:
        """Translate segment.original_text → segment.translated_text."""
        self.load()
        segment.target_language = self.target_lang

        text = (segment.original_text or "").strip()
        if not text:
            segment.translated_text = ""
            return segment

        sentences = _split_sentences(text)
        translated_sentences = []
        for sent in sentences:
            t = self._translate_with_context(sent)
            translated_sentences.append(t)
            self._context.append(sent)
            if len(self._context) > CONTEXT_WINDOW:
                self._context.pop(0)

        segment.translated_text = " ".join(translated_sentences)
        return segment

    def translate_batch(self, segments: list[Segment]) -> list[Segment]:
        total = len(segments)
        for i, seg in enumerate(segments):
            log.debug("Translating segment %d/%d", i + 1, total)
            self.translate_segment(seg)
        return segments

    def reset_context(self) -> None:
        """Call between videos or very different topics."""
        self._context.clear()

    def update_source_lang(self, lang_code: str) -> None:
        """
        Re-point the tokenizer source language after Whisper auto-detection.
        Only re-loads if the language actually changed.
        Call this before processing the first translation.
        """
        new_code = NLLB_LANG_CODES.get(lang_code, "eng_Latn")
        if new_code == self._src_code:
            return
        log.info("Updating source language: %s → %s", self._src_code, new_code)
        self._src_code = new_code
        self.source_lang = lang_code
        # Re-init tokenizer with the correct source language
        if self._tokenizer is not None:
            from transformers import NllbTokenizer
            self._tokenizer = NllbTokenizer.from_pretrained(
                self.model_id, src_lang=self._src_code
            )

    # ------------------------------------------------------------------

    def _translate_with_context(self, text: str) -> str:
        import torch

        # Translate segment text directly — no context prepending.
        # Prepending previous sentences caused NLLB to repeat context in output,
        # producing very long strings that overflowed XTTS token limits.
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
        )
        device_str = DEVICE
        if device_str in ("mps", "cuda"):
            inputs = {k: v.to(device_str) for k, v in inputs.items()}

        # Target language token (API differs between transformers versions)
        if hasattr(self._tokenizer, "lang_code_to_id"):
            tgt_lang_id = self._tokenizer.lang_code_to_id[self._tgt_code]
        else:
            tgt_lang_id = self._tokenizer.convert_tokens_to_ids(self._tgt_code)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=MAX_TOKENS,
                num_beams=4,
                early_stopping=True,
            )

        translated = self._tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        return translated


# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter that respects common punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    merged: list[str] = []
    buf = ""
    last_idx = len(parts) - 1
    for i, part in enumerate(parts):
        buf = (buf + " " + part).strip()
        tokens = buf.split()
        if len(tokens) >= 20 or i == last_idx:
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return [s for s in merged if s]
