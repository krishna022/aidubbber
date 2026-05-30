"""
Speaker diarization — fully token-free, no HuggingFace account required.

Stack:
  • Silero VAD     – detects speech segments (torch.hub, cached locally)
  • SpeechBrain ECAPA-TDNN – extracts per-segment speaker embeddings
  • sklearn AgglomerativeClustering – groups embeddings into speakers
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from config import TEMP_DIR, MIN_SEGMENT_DURATION, SAMPLE_RATE, Segment
from utils.speaker_registry import SpeakerRegistry
from pipeline.audio_extractor import AudioExtractor

log = logging.getLogger(__name__)

# Public SpeechBrain model — no token, no account required
_ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
_ECAPA_SAVEDIR = str(Path(__file__).parent.parent / "models" / "spkrec-ecapa")


class Diarizer:
    """
    Token-free speaker diarizer.
    Works on Apple Silicon via MPS for VAD; SpeechBrain inference on CPU/MPS.
    """

    def __init__(self, registry: SpeakerRegistry, temp_dir: Path = TEMP_DIR):
        self.registry = registry
        self.temp_dir = temp_dir
        self._extractor = AudioExtractor(temp_dir)
        self._vad = None
        self._get_ts = None       # silero helper function
        self._encoder = None      # SpeechBrain speaker encoder

    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._vad is not None:
            return
        self._load_vad()
        self._load_encoder()

    # ------------------------------------------------------------------

    def diarize_chunk(
        self,
        audio_path: Path,
        chunk_start: float,
        chunk_index: int,
    ) -> list[Segment]:
        """
        Full diarize pass on one audio chunk.
        Returns Segment list with global timestamps.
        """
        self.load()

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # --- 1. VAD -------------------------------------------------------
        speech_times = self._run_vad(audio, sr)
        if not speech_times:
            log.warning("No speech detected in chunk %d", chunk_index)
            return []

        # --- 2. Speaker embeddings ----------------------------------------
        embeddings, valid_times = [], []
        for start, end in speech_times:
            emb = self._embed_segment(audio, sr, start, end)
            if emb is not None:
                embeddings.append(emb)
                valid_times.append((start, end))

        if not embeddings:
            return []

        # --- 3. Cluster into speakers -------------------------------------
        emb_arr = np.vstack(embeddings)
        n_spk = self._estimate_n_speakers(emb_arr)
        labels = self._cluster(emb_arr, n_spk)

        # --- 4. Build Segment objects -------------------------------------
        segments: list[Segment] = []
        for idx, ((start, end), label) in enumerate(zip(valid_times, labels)):
            speaker_id = f"SPEAKER_{label:02d}"
            global_start = chunk_start + start
            global_end = chunk_start + end

            seg_path = (
                self.temp_dir
                / f"seg_c{chunk_index:04d}_{idx:04d}_{speaker_id}.wav"
            )
            self._extractor.extract_segment(audio_path, start, end, seg_path)
            self.registry.add_or_update(
                speaker_id=speaker_id,
                segment_audio_path=str(seg_path),
                duration=end - start,
            )

            segments.append(
                Segment(
                    segment_id=f"seg_c{chunk_index:04d}_{idx:04d}",
                    speaker_id=speaker_id,
                    start=global_start,
                    end=global_end,
                    chunk_index=chunk_index,
                    audio_path=str(seg_path),
                )
            )

        log.info(
            "Chunk %d: %d speech segments → %d speaker(s)",
            chunk_index, len(segments), n_spk,
        )
        return segments

    # ------------------------------------------------------------------
    # Private: model loaders
    # ------------------------------------------------------------------

    def _load_vad(self) -> None:
        log.info("Loading Silero VAD …")
        vad, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )
        self._vad = vad
        self._get_ts = utils[0]          # get_speech_timestamps
        log.info("Silero VAD loaded")

    def _load_encoder(self) -> None:
        log.info("Loading SpeechBrain ECAPA-TDNN speaker encoder …")
        # Handle both speechbrain < 1.0 and >= 1.0 import paths
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier  # type: ignore

        self._encoder = EncoderClassifier.from_hparams(
            source=_ECAPA_SOURCE,
            savedir=_ECAPA_SAVEDIR,
            run_opts={"device": "cpu"},  # CPU is safer for SpeechBrain on MPS
        )
        log.info("SpeechBrain ECAPA-TDNN loaded")

    # ------------------------------------------------------------------
    # Private: inference helpers
    # ------------------------------------------------------------------

    def _run_vad(self, audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """Return list of (start_sec, end_sec) speech intervals."""
        tensor = torch.from_numpy(audio)
        timestamps = self._get_ts(
            tensor,
            self._vad,
            sampling_rate=sr,
            min_speech_duration_ms=int(MIN_SEGMENT_DURATION * 1000),
            min_silence_duration_ms=250,
            return_seconds=True,
        )
        result = []
        for ts in timestamps:
            start = float(ts["start"])
            end = float(ts["end"])
            if end - start >= MIN_SEGMENT_DURATION:
                result.append((start, end))
        return result

    def _embed_segment(
        self, audio: np.ndarray, sr: int, start: float, end: float
    ) -> Optional[np.ndarray]:
        """Extract ECAPA speaker embedding for one time slice."""
        start_f = int(start * sr)
        end_f = int(end * sr)
        seg = audio[start_f:end_f]

        if len(seg) < sr * 0.3:          # skip clips shorter than 0.3 s
            return None

        seg_tensor = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)

        # ECAPA expects 16 kHz
        if sr != 16000:
            seg_tensor = torchaudio.functional.resample(seg_tensor, sr, 16000)

        try:
            with torch.no_grad():
                emb = self._encoder.encode_batch(seg_tensor)
            return emb.squeeze().cpu().numpy()
        except Exception as exc:
            log.debug("Embedding failed for %.1f–%.1f: %s", start, end, exc)
            return None

    def _estimate_n_speakers(self, emb_arr: np.ndarray, max_speakers: int = 8) -> int:
        """Use silhouette score to find the best number of speakers."""
        n = len(emb_arr)
        if n <= 1:
            return 1
        if n == 2:
            cos = float(np.dot(emb_arr[0], emb_arr[1]) / (
                np.linalg.norm(emb_arr[0]) * np.linalg.norm(emb_arr[1]) + 1e-9
            ))
            return 1 if cos > 0.80 else 2

        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        best_n, best_score = 1, -1.0
        for k in range(2, min(max_speakers, n - 1) + 1):
            try:
                labels = AgglomerativeClustering(
                    n_clusters=k, metric="cosine", linkage="average"
                ).fit_predict(emb_arr)
                score = silhouette_score(emb_arr, labels, metric="cosine")
                if score > best_score:
                    best_score, best_n = score, k
            except Exception:
                break

        log.info("Estimated %d speaker(s) (silhouette=%.3f)", best_n, best_score)
        return best_n

    @staticmethod
    def _cluster(emb_arr: np.ndarray, n: int) -> list[int]:
        if n == 1:
            return [0] * len(emb_arr)
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(
            n_clusters=n, metric="cosine", linkage="average"
        ).fit_predict(emb_arr).tolist()
