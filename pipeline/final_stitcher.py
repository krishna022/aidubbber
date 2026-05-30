"""Final video/audio assembly using FFmpeg."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from config import SAMPLE_RATE, TEMP_DIR, Segment
from utils.timing import merge_audio_segments

log = logging.getLogger(__name__)


class FinalStitcher:
    """
    Composites dubbed segments onto a silent canvas, strips the original
    speech (using VAD) while keeping background music/SFX, then muxes
    everything back into the original video container.

    Audio layers in final output:
      Layer 1 – Dubbed Hindi voice         (100% volume)
      Layer 2 – Original background only   (music/SFX, speech silenced via VAD)
    """

    def __init__(
        self,
        temp_dir: Path = TEMP_DIR,
        keep_background: bool = True,
        background_volume: float = 0.80,   # higher now — speech is already removed
    ):
        self.temp_dir = temp_dir
        self.keep_background = keep_background
        self.background_volume = background_volume

    # ------------------------------------------------------------------

    def stitch(
        self,
        video_path: Path,
        segments: list[Segment],
        output_path: Path,
        total_duration: float,
    ) -> Path:
        log.info("Stitching %d dubbed segments …", len(segments))

        # --- 1. Build dubbed audio canvas --------------------------------
        dubbed_clips = [
            (seg.start, seg.dubbed_audio_path)
            for seg in segments
            if seg.dubbed_audio_path and Path(seg.dubbed_audio_path).exists()
        ]

        if not dubbed_clips:
            raise RuntimeError("No dubbed audio segments found. Check voice cloner output.")

        dubbed_wav = self.temp_dir / "dubbed_full.wav"
        merge_audio_segments(dubbed_clips, total_duration, str(dubbed_wav))
        log.info("Dubbed audio canvas built: %s", dubbed_wav)

        # --- 2. Extract background-only from original (speech removed) ---
        final_audio = dubbed_wav
        if self.keep_background:
            original_wav = self.temp_dir / "full_audio.wav"
            if original_wav.exists():
                bg_wav = self.temp_dir / "background_only.wav"
                mixed_wav = self.temp_dir / "mixed_final.wav"
                self._extract_background(str(original_wav), str(bg_wav))
                self._mix(str(dubbed_wav), str(bg_wav), str(mixed_wav))
                final_audio = mixed_wav

        # --- 3. Mux into final video ------------------------------------
        self._mux(video_path, final_audio, output_path)
        log.info("Final video written: %s", output_path)
        return output_path

    # ------------------------------------------------------------------

    def _extract_background(self, original_wav: str, out_wav: str) -> None:
        """
        Use Silero VAD to detect speech in the original audio, then silence
        those regions (with smooth fade edges to avoid clicks).
        The result keeps only music, SFX, and ambient sound.
        """
        log.info("Extracting background audio (removing original speech) …")
        import torch

        audio, sr = sf.read(original_wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Load Silero VAD (already cached from diarization stage)
        vad, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad",
            force_reload=False, trust_repo=True, verbose=False,
        )
        get_ts = utils[0]

        speech_ts = get_ts(
            torch.from_numpy(audio),
            vad,
            sampling_rate=sr,
            min_speech_duration_ms=200,
            min_silence_duration_ms=100,
            return_seconds=True,
        )

        bg = audio.copy()
        fade_s = 0.03   # 30 ms crossfade to avoid clicks
        fade_len = int(fade_s * sr)

        for ts in speech_ts:
            s = int(ts["start"] * sr)
            e = int(ts["end"] * sr)
            seg_len = e - s
            fl = min(fade_len, seg_len // 4)

            # Fade out at start of speech
            if fl > 0:
                bg[s: s + fl] *= np.linspace(1.0, 0.0, fl)
            # Silence the body
            bg[s + fl: e - fl] = 0.0
            # Fade in at end of speech
            if fl > 0 and e - fl < e:
                bg[e - fl: e] *= np.linspace(0.0, 1.0, fl)

        sf.write(out_wav, bg, sr)
        log.info(
            "Background extracted: %d speech regions silenced", len(speech_ts)
        )

    def _mix(self, dubbed_wav: str, bg_wav: str, out_wav: str) -> None:
        """
        Mix dubbed voice (full volume) with speech-free background.
        Uses FFmpeg amix so both tracks are at natural length.
        """
        vol = self.background_volume
        cmd = [
            "ffmpeg", "-y",
            "-i", dubbed_wav,
            "-i", bg_wav,
            "-filter_complex",
            (
                f"[0:a]volume=1.0[dub];"
                f"[1:a]volume={vol}[bg];"
                f"[dub][bg]amix=inputs=2:duration=first:dropout_transition=0[out]"
            ),
            "-map", "[out]",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            out_wav,
        ]
        _run(cmd, "background mix")
        log.info("Mixed dubbed voice + background: %s", out_wav)

    def _mux(self, video_path: Path, audio_wav: Path, out_path: Path) -> None:
        """Mux original video stream (copy) with new audio. Re-encodes audio to AAC 192k."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_wav),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(out_path),
        ]
        _run(cmd, "final mux")

    # ------------------------------------------------------------------

    @staticmethod
    def extract_subtitles(segments: list[Segment], out_path: Path, lang: str = "en") -> Path:
        lines = []
        counter = 1
        for seg in segments:
            if not seg.translated_text:
                continue
            lines.append(
                f"{counter}\n"
                f"{_srt_ts(seg.start)} --> {_srt_ts(seg.end)}\n"
                f"{seg.translated_text}\n"
            )
            counter += 1
        out_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Subtitles written: %s", out_path)
        return out_path


# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> None:
    log.debug("FFmpeg %s: %s", label, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg {label} failed (rc={result.returncode}):\n{result.stderr[-2000:]}"
        )


def _srt_ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
