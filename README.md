# AIDubber

**Local-first AI video dubbing engine — 100% free, 100% offline, built for Apple Silicon.**

Dub any video into 16 languages while preserving each speaker's original voice characteristics. No paid APIs, no cloud, no accounts. Everything runs on your Mac.

---

## What It Does

AIDubber takes a video in any language and produces a fully dubbed video in your target language. It:

- Detects and separates every speaker in the video automatically
- Transcribes speech using OpenAI Whisper (MLX-accelerated)
- Translates to your target language using Meta's NLLB-200
- Clones each speaker's voice using Coqui XTTS v2 (matches pitch, tone, gender)
- Strips the original voice from the final video while keeping background music and sound effects
- Outputs a ready-to-watch `.mp4` + an `.srt` subtitle file

---

## Requirements

| Item | Minimum |
|---|---|
| Hardware | Apple Silicon Mac (M1 / M2 / M3 / M4 / M5) |
| RAM | 16 GB (48 GB recommended for 1-hour videos) |
| Disk | 15 GB free (models + temp files) |
| macOS | Ventura 13.5+ |
| Python | 3.10 or 3.11 (3.11 recommended) |
| Internet | Only on first run (to download models) |

> **Windows / Linux:** The pipeline works but you must remove `mlx` and `mlx-whisper` from `requirements.txt` and use `openai-whisper` instead. MPS device detection in `config.py` will fall back to CUDA or CPU automatically.

---

## Supported Languages

| Code | Language | Code | Language |
|---|---|---|---|
| `hi` | Hindi | `es` | Spanish |
| `fr` | French | `de` | German |
| `it` | Italian | `pt` | Portuguese |
| `ru` | Russian | `zh` | Chinese |
| `ja` | Japanese | `ko` | Korean |
| `ar` | Arabic | `nl` | Dutch |
| `pl` | Polish | `tr` | Turkish |
| `uk` | Ukrainian | `en` | English |

---

## Installation

### Step 1 — Install system dependencies

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install ffmpeg and Python 3.11
brew install ffmpeg python@3.11
```

### Step 2 — Clone the repository

```bash
git clone https://github.com/krishna022/aidubbber.git
cd aidubbber
```

### Step 3 — Run the setup script

```bash
bash setup.sh
```

The setup script will:
- Create a Python 3.11 virtual environment (`.venv/`)
- Install PyTorch with MPS (Apple Silicon GPU) support
- Install MLX and mlx-whisper
- Install all other dependencies
- Download all required models (**~6 GB total, one-time only**):
  - Whisper large-v3 MLX (~1.5 GB)
  - NLLB-200 translation model (~2.4 GB)
  - XTTS v2 voice cloning model (~1.8 GB)
  - SpeechBrain ECAPA-TDNN speaker encoder (~90 MB)
  - Silero VAD (~5 MB, via torch.hub)

> After first run, all models are cached locally. The pipeline runs **fully offline**.

### Step 4 — Validate your setup

```bash
source .venv/bin/activate
python validate.py
```

You should see `13 passed, 0 failed`.

---

## Usage

### Basic — Dub a video to Hindi

```bash
source .venv/bin/activate
python main.py --input input/myvideo.mp4 --target-lang hi
```

Output appears at `output/myvideo_hi.mp4` with `output/myvideo_hi.srt`.

### Full options

```bash
python main.py \
  --input   input/myvideo.mp4\   # Path to your video (required)
  --target-lang  hi \             # Target language code (required)
  --source-lang  en \             # Source language — auto-detected if omitted
  --output  output/dubbed.mp4 \   # Custom output path (optional)
  --no-background \               # Drop original background audio entirely
  --bg-volume 0.8                 # Background music volume, 0.0–1.0 (default 0.8)
```

### Where to put your video

Drop it anywhere and pass the full path, or use the `input/` folder:

```
aidubbber/
├── input/
│   └── myvideo.mp4     ← put your video here
├── output/
│   ├── myvideo_hi.mp4  ← dubbed video appears here
│   └── myvideo_hi.srt  ← subtitle file appears here
```

---

## Configuration

All model and pipeline settings are in [`config.py`](config.py).

### Switch Whisper model size

```python
# config.py — line 23
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"   # best accuracy (default)
WHISPER_MODEL = "mlx-community/whisper-medium-mlx"     # faster, slightly less accurate
WHISPER_MODEL = "mlx-community/whisper-small-mlx"      # fastest, lower accuracy
```

### Adjust chunk size for very long videos

```python
# config.py
CHUNK_DURATION = 300    # seconds per diarization chunk (default 5 min)
OVERLAP_DURATION = 10   # overlap between chunks to avoid boundary cuts
```

### Tune time-stretching threshold

```python
# config.py
MAX_TIME_STRETCH_RATIO = 1.6  # beyond this ratio, pad/trim instead of stretch
```

---

## How It Works — Pipeline Architecture

```
Video ──► AudioExtractor ──► full_audio.wav (16 kHz mono)
                │
                ▼
          Diarizer (Silero VAD + SpeechBrain ECAPA + sklearn)
          Detects speakers, extracts reference clips
                │
                ▼
    ┌─────────────────────────────────────────────┐
    │            Queue-based Pipeline              │
    │                                              │
    │  Transcriber (mlx-whisper large-v3)          │
    │       ↓                                      │
    │  Translator (NLLB-200 on MPS)               │
    │       ↓                                      │
    │  VoiceCloner (Coqui XTTS v2, per-speaker)   │
    └─────────────────────────────────────────────┘
                │
                ▼
          FinalStitcher
          ├── VAD-gates original audio (removes speech, keeps music/SFX)
          ├── Merges all dubbed clips onto timeline
          ├── Mixes dubbed voice + background music
          └── FFmpeg mux → output .mp4 + .srt
```

### Components

| Stage | Tool | Purpose |
|---|---|---|
| Audio extraction | FFmpeg | Extract 16 kHz mono WAV from any container |
| Voice activity detection | Silero VAD | Detect speech boundaries |
| Speaker diarization | SpeechBrain ECAPA-TDNN + sklearn | Identify & separate individual speakers |
| Transcription | mlx-whisper large-v3 | Speech-to-text with Apple Silicon acceleration |
| Translation | NLLB-200 (Meta, 600M) on MPS | High-quality neural machine translation |
| Voice cloning | Coqui XTTS v2 | Synthesise speech in target language matching original voice |
| Audio processing | librosa + FFmpeg | Time-stretch, normalise, mix |
| Final assembly | FFmpeg | Mux video + dubbed audio + subtitles |

### Speaker Registry

Every speaker found during diarization gets a profile stored in `.temp/<video>/speaker_registry.json`:

```json
{
  "SPEAKER_00": {
    "reference_audio_path": ".temp/video/seg_c0000_0003_SPEAKER_00.wav",
    "total_speech_duration": 142.3,
    "estimated_gender": "female",
    "estimated_pitch_hz": 214.5
  }
}
```

XTTS v2 uses the reference audio to match each speaker's voice in the dubbed output.

---

## Output Files

```
output/
├── video_input1_hi.mp4    ← Final dubbed video
└── video_input1_hi.srt    ← Hindi subtitle file (load in VLC / IINA)

.temp/<video_stem>/
├── speaker_registry.json  ← Speaker profiles (cached, reused on re-runs)
├── chunk_XXXX.wav         ← Audio chunks (diarization input)
├── seg_*.wav              ← Per-speaker segment clips
├── tts_*.wav              ← Raw XTTS output per segment
├── dubbed_*.wav           ← Time-adjusted dubbed clips
├── dubbed_full.wav        ← All clips merged on timeline
├── background_only.wav    ← Original audio with speech removed
└── mixed_final.wav        ← dubbed_full + background_only mixed
```

---

## Troubleshooting

### "No module named 'mlx_whisper'"
```bash
source .venv/bin/activate
pip install mlx mlx-whisper
```

### "XTTS failed … maximum of 400 tokens"
This is handled automatically — long text is split into chunks and concatenated. If you still see it, the text may have no sentence boundaries. The segment will be skipped (silent gap). Reduce `CHUNK_DURATION` in `config.py` to get shorter segments.

### "Stretch ratio X.XX out of range"
Normal warning — appears when the dubbed clip is much shorter or longer than the original. The clip is padded with silence or trimmed instead of being stretched unnaturally. Increasing `MAX_TIME_STRETCH_RATIO` in `config.py` allows more aggressive stretching.

### Both original and dubbed voice audible
Run with `--no-background` to fully drop the original audio track:
```bash
python main.py --input input/video.mp4 --target-lang hi --no-background
```

### MPS not available
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```
If `False`, reinstall PyTorch: `pip install torch torchvision torchaudio`

### Out of memory on long videos
Reduce chunk size in `config.py`:
```python
CHUNK_DURATION = 120   # 2 minutes instead of 5
```

---

## Project Structure

```
aidubbber/
├── main.py                  # Entry point + pipeline orchestration
├── config.py                # All settings, language codes, Segment dataclass
├── validate.py              # Pre-flight dependency check
├── requirements.txt         # Python dependencies
├── setup.sh                 # One-shot environment setup
│
├── pipeline/
│   ├── audio_extractor.py   # FFmpeg audio extraction
│   ├── diarizer.py          # Speaker detection (Silero VAD + ECAPA)
│   ├── transcriber.py       # Whisper MLX transcription
│   ├── translator.py        # NLLB-200 translation
│   ├── voice_cloner.py      # XTTS v2 voice synthesis
│   └── final_stitcher.py    # Audio mixing + FFmpeg mux
│
└── utils/
    ├── speaker_registry.py  # Thread-safe speaker profile store
    ├── chunk_manager.py     # Long-audio chunking with overlap
    ├── queue_manager.py     # Thread-based pipeline queue system
    └── timing.py            # Audio time-stretch + merge utilities
```

---

## Models Used

| Model | Size | Purpose | License |
|---|---|---|---|
| [Whisper large-v3 (MLX)](https://huggingface.co/mlx-community/whisper-large-v3-mlx) | ~1.5 GB | Speech-to-text | MIT |
| [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | ~2.4 GB | Translation | CC-BY-NC |
| [XTTS v2](https://huggingface.co/coqui/XTTS-v2) | ~1.8 GB | Voice cloning | CPML (non-commercial) |
| [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | ~90 MB | Speaker embeddings | Apache 2.0 |
| [Silero VAD](https://github.com/snakers4/silero-vad) | ~5 MB | Voice activity detection | MIT |

> **All models are free for non-commercial use.** XTTS v2 requires agreeing to the [Coqui CPML licence](https://coqui.ai/cpml) for commercial use.

---

## Performance (M5 Pro, 48 GB RAM)

| Video length | Diarization | Transcription | Translation | Voice cloning | Total |
|---|---|---|---|---|---|
| 5 min | ~10 s | ~30 s | ~20 s | ~8 min | ~9 min |
| 15 min | ~30 s | ~90 s | ~60 s | ~25 min | ~28 min |
| 30 min | ~60 s | ~3 min | ~2 min | ~50 min | ~56 min |
| 1 hour | ~2 min | ~6 min | ~4 min | ~100 min | ~112 min |

Voice cloning (XTTS v2) is the bottleneck. It runs at ~0.5× real-time on CPU. Enabling MPS for XTTS would cut this significantly.

---

## License

This project is released under the **MIT License**.

The underlying models have their own licences — see the table above. Ensure you comply with each model's licence for your use case, especially XTTS v2 (non-commercial CPML) and NLLB-200 (CC-BY-NC).
