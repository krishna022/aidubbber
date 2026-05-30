#!/usr/bin/env python3
"""
Run this before your first video to verify all models and imports are working.

    python validate.py
"""
from __future__ import annotations

import sys
import traceback

CHECKS: list[tuple[str, str]] = []

def check(label: str):
    def decorator(fn):
        CHECKS.append((label, fn))
        return fn
    return decorator


@check("Python ≥ 3.10")
def _():
    assert sys.version_info >= (3, 10), f"Need Python 3.10+, got {sys.version}"


@check("PyTorch + MPS")
def _():
    import torch
    print(f"    torch {torch.__version__}", end="  ")
    if torch.backends.mps.is_available():
        t = torch.ones(1, device="mps")
        print(f"MPS OK (tensor={t.item()})")
    else:
        print("MPS not available – will use CPU")


@check("MLX")
def _():
    import mlx.core as mx
    print(f"    mlx {mx.__version__}  device={mx.default_device()}")


@check("mlx-whisper")
def _():
    import mlx_whisper
    print(f"    mlx-whisper imported OK")


@check("SpeechBrain (speaker encoder)")
def _():
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier  # type: ignore
    print("    speechbrain EncoderClassifier importable")

@check("scikit-learn (clustering)")
def _():
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    print("    scikit-learn clustering importable")


@check("transformers (NLLB)")
def _():
    from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
    print("    transformers imported OK")


@check("Coqui TTS")
def _():
    from TTS.api import TTS
    print("    TTS imported OK")


@check("soundfile / librosa")
def _():
    import soundfile, librosa
    print(f"    soundfile {soundfile.__version__}  librosa {librosa.__version__}")


@check("ffmpeg binary")
def _():
    import subprocess
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    ver = r.stdout.decode().splitlines()[0] if r.returncode == 0 else "NOT FOUND"
    assert r.returncode == 0, "ffmpeg not found in PATH"
    print(f"    {ver[:60]}")


@check("config imports")
def _():
    from config import DEVICE, NLLB_LANG_CODES, Segment
    seg = Segment("id0", "SPEAKER_00", 0.0, 1.0)
    print(f"    device={DEVICE}  segment OK  langs={len(NLLB_LANG_CODES)}")


@check("pipeline imports")
def _():
    from pipeline import (
        AudioExtractor, Diarizer, Transcriber,
        Translator, VoiceCloner, FinalStitcher,
    )
    print("    all pipeline classes importable")


@check("utils imports")
def _():
    from utils import SpeakerRegistry, ChunkManager, PipelineQueue, PipelineStage
    print("    all util classes importable")


# ---------------------------------------------------------------------------

def main() -> None:
    passed = 0
    failed = 0
    print("\n── video_dubd validation ──────────────────────────────────────\n")
    for label, fn in CHECKS:
        try:
            fn()
            print(f"  ✓  {label}")
            passed += 1
        except Exception as exc:
            print(f"  ✗  {label}")
            print(f"     {exc}")
            traceback.print_exc()
            failed += 1
        print()

    print(f"── Result: {passed} passed, {failed} failed ──────────────────────────\n")
    if failed:
        print("Run  setup.sh  to fix missing dependencies.\n")
        sys.exit(1)
    else:
        print("All checks passed. Ready to dub!\n")


if __name__ == "__main__":
    main()
