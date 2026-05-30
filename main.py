#!/usr/bin/env python3
"""
video_dubd – Local-first AI video dubbing engine for Apple Silicon.

Usage:
    python main.py --input video.mp4 --target-lang es [options]

Environment:
    HF_TOKEN  – HuggingFace token required for pyannote diarization.
                Get one at https://huggingface.co/settings/tokens
                and accept the pyannote model licence at:
                https://hf.co/pyannote/speaker-diarization-3.1
"""
from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
from pathlib import Path

from config import (
    BASE_DIR, TEMP_DIR, OUTPUT_DIR, QUEUE_MAXSIZE,
    NLLB_LANG_CODES, Segment,
)
from pipeline import (
    AudioExtractor, Diarizer, Transcriber,
    Translator, VoiceCloner, FinalStitcher,
)
from utils import SpeakerRegistry, PipelineQueue, QueueItem, SENTINEL, PipelineStage
from utils.chunk_manager import ChunkManager
import soundfile as sf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s – %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "dubd.log"),
    ],
)
log = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def run_pipeline(
    video_path: Path,
    target_lang: str,
    source_lang: str,
    output_path: Path,
    keep_background: bool,
    background_volume: float,
) -> None:
    if target_lang not in NLLB_LANG_CODES:
        raise ValueError(
            f"Unsupported target language '{target_lang}'. "
            f"Supported: {sorted(NLLB_LANG_CODES)}"
        )

    job_temp = TEMP_DIR / video_path.stem
    job_temp.mkdir(exist_ok=True)

    registry_path = job_temp / "speaker_registry.json"
    registry = SpeakerRegistry(registry_path)

    # -----------------------------------------------------------------------
    # Stage 0: Extract audio
    # -----------------------------------------------------------------------
    log.info("=== Stage 0: Audio extraction ===")
    extractor = AudioExtractor(job_temp)
    audio_path = extractor.extract(video_path)
    info = sf.info(str(audio_path))
    total_duration = info.duration
    log.info("Total audio duration: %.1f s (%.1f min)", total_duration, total_duration / 60)

    # -----------------------------------------------------------------------
    # Stage 1: Diarization (chunk-based for long videos)
    # -----------------------------------------------------------------------
    log.info("=== Stage 1: Speaker diarization ===")
    diarizer = Diarizer(registry, job_temp)
    chunk_mgr = ChunkManager(audio_path, job_temp)

    all_segments: list[Segment] = []
    total_chunks = chunk_mgr.total_chunks()
    log.info("Processing %d chunk(s) …", total_chunks)

    for chunk in chunk_mgr.iter_chunks():
        log.info(
            "Diarizing chunk %d/%d (%.0f–%.0f s) …",
            chunk.chunk_index + 1, total_chunks, chunk.start, chunk.end,
        )
        segs = diarizer.diarize_chunk(chunk.audio_path, chunk.start, chunk.chunk_index)
        all_segments.extend(segs)

    all_segments = ChunkManager.deduplicate_segments(all_segments)
    all_segments.sort(key=lambda s: s.start)
    log.info("Total segments after dedup: %d", len(all_segments))

    # Analyse speakers now that all chunks have been diarized
    registry.analyse_speakers()
    registry.save()

    # -----------------------------------------------------------------------
    # Stage 2–4: Transcription → Translation → Voice cloning (queue-based)
    # -----------------------------------------------------------------------
    log.info("=== Stages 2–4: Transcription / Translation / Cloning ===")

    transcription_q = PipelineQueue("transcription", QUEUE_MAXSIZE)
    translation_q = PipelineQueue("translation", QUEUE_MAXSIZE)
    cloning_q = PipelineQueue("cloning", QUEUE_MAXSIZE)
    done_q: queue.Queue[Segment] = queue.Queue()

    transcriber = Transcriber()

    # Auto-detect source language from the first available segment audio
    # if the user did not specify --source-lang explicitly.
    detected_source_lang = source_lang
    if not detected_source_lang and all_segments:
        log.info("=== Auto-detecting source language ===")
        transcriber.load()
        probe_seg = next(
            (s for s in all_segments if s.audio_path and Path(s.audio_path).exists()),
            None,
        )
        if probe_seg:
            detected_source_lang = transcriber.detect_language(probe_seg.audio_path)
        else:
            detected_source_lang = "en"
            log.warning("No audio segment found for language detection; defaulting to 'en'")

    translator = Translator(source_lang=detected_source_lang, target_lang=target_lang)
    voice_cloner = VoiceCloner(registry, target_lang=target_lang, temp_dir=job_temp)

    # Pre-load all models before starting workers (avoids thread-unsafe init races)
    log.info("Pre-loading models …")
    transcriber.load()
    translator.load()
    voice_cloner.load()

    # Worker functions -------------------------------------------------------
    _lang_updated = threading.Event()

    def transcribe_worker(data: Segment, meta: dict):
        seg = transcriber.transcribe_segment(data, source_language=detected_source_lang or None)
        # On the first segment, refresh translator source lang with Whisper's detection
        if not _lang_updated.is_set() and seg.source_language:
            translator.update_source_lang(seg.source_language)
            _lang_updated.set()
        return seg

    def translate_worker(data: Segment, meta: dict):
        return translator.translate_segment(data)

    def clone_worker(data: Segment, meta: dict):
        result = voice_cloner.clone_segment(data)
        done_q.put(result)
        return None  # don't forward to another queue

    # Wire stages ------------------------------------------------------------
    stage_transcribe = PipelineStage(
        "transcribe", transcribe_worker,
        transcription_q, translation_q, num_workers=1,
    )
    stage_translate = PipelineStage(
        "translate", translate_worker,
        translation_q, cloning_q, num_workers=1,
    )

    # Cloning stage writes directly to done_q
    cloning_sink = PipelineQueue("cloning_sink", QUEUE_MAXSIZE)
    stage_clone = PipelineStage(
        "clone", clone_worker,
        cloning_q, cloning_sink, num_workers=1,
    )

    for stage in (stage_transcribe, stage_translate, stage_clone):
        stage.start()

    # Feed segments into transcription queue ---------------------------------
    def feed():
        for seg in all_segments:
            transcription_q.put(QueueItem(data=seg))
        transcription_q.put(SENTINEL)

    feed_thread = threading.Thread(target=feed, daemon=True)
    feed_thread.start()

    # Drain results: wait for the clone stage sentinel, then harvest done_q ----
    completed: list[Segment] = []

    # Block until cloning_sink receives its sentinel (signals stage_clone done)
    while True:
        try:
            item = cloning_sink.get(timeout=10)
            if item.is_sentinel:
                break
            if item.is_error:
                log.error("Pipeline error: %s", item.error)
        except queue.Empty:
            # All three stages signalled done via _sentinel_count
            if (
                stage_transcribe._sentinel_count >= 1
                and stage_translate._sentinel_count >= 1
                and stage_clone._sentinel_count >= 1
            ):
                break

    # Harvest all processed segments from the side-channel done_q
    while True:
        try:
            completed.append(done_q.get_nowait())
        except queue.Empty:
            break

    feed_thread.join()
    for s in (stage_transcribe, stage_translate, stage_clone):
        s.join()

    log.info("Pipeline complete: %d segments dubbed", len(completed))
    _log_pipeline_errors(stage_transcribe, stage_translate, stage_clone)

    # -----------------------------------------------------------------------
    # Stage 5: Final assembly
    # -----------------------------------------------------------------------
    log.info("=== Stage 5: Final assembly ===")
    stitcher = FinalStitcher(
        temp_dir=job_temp,
        keep_background=keep_background,
        background_volume=background_volume,
    )

    # Use all_segments for timing info; completed for dubbed paths
    dubbed_map = {s.segment_id: s for s in completed}
    final_segments = []
    for seg in all_segments:
        dubbed = dubbed_map.get(seg.segment_id, seg)
        final_segments.append(dubbed)

    stitcher.stitch(video_path, final_segments, output_path, total_duration)

    # Write subtitles
    srt_path = output_path.with_suffix(".srt")
    stitcher.extract_subtitles(final_segments, srt_path, lang=target_lang)

    log.info("Done! Output: %s", output_path)
    log.info("Subtitles: %s", srt_path)


# ---------------------------------------------------------------------------

def _log_pipeline_errors(*stages) -> None:
    for stage in stages:
        if stage.errors:
            log.warning("Stage '%s' had %d error(s):", stage.name, len(stage.errors))
            for err in stage.errors[:5]:
                log.warning("  %s", err)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_dubd",
        description="Local-first AI video dubbing engine (Apple M-series optimised)",
    )
    p.add_argument("--input", required=True, type=Path, help="Input video file")
    p.add_argument(
        "--target-lang", required=True,
        help=f"Target language code. Supported: {sorted(NLLB_LANG_CODES)}",
    )
    p.add_argument(
        "--source-lang", default="",
        help="Source language code (auto-detect if omitted)",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output video path (default: output/<stem>_<lang>.mp4)",
    )
    p.add_argument(
        "--no-background", action="store_true",
        help="Discard original background audio (music/SFX)",
    )
    p.add_argument(
        "--bg-volume", type=float, default=0.15,
        help="Background audio volume 0.0–1.0 (default 0.15)",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    video_path: Path = args.input.resolve()
    if not video_path.exists():
        parser.error(f"Input file not found: {video_path}")

    target_lang: str = args.target_lang.lower()
    source_lang: str = args.source_lang.lower()

    if args.output:
        output_path = args.output.resolve()
    else:
        output_path = OUTPUT_DIR / f"{video_path.stem}_{target_lang}.mp4"

    log.info("video_dubd starting")
    log.info("  Input      : %s", video_path)
    log.info("  Source lang: %s (auto-detect if empty)", source_lang or "(auto)")
    log.info("  Target lang: %s", target_lang)
    log.info("  Output     : %s", output_path)

    run_pipeline(
        video_path=video_path,
        target_lang=target_lang,
        source_lang=source_lang,
        output_path=output_path,
        keep_background=not args.no_background,
        background_volume=args.bg_volume,
    )


if __name__ == "__main__":
    main()
