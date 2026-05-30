#!/usr/bin/env bash
# video_dubd setup script for macOS Apple Silicon (M1/M2/M3/M4/M5)
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── 0. Prerequisites ────────────────────────────────────────────────────────

command -v python3 >/dev/null 2>&1 || error "python3 not found. Install via: brew install python"
command -v ffmpeg  >/dev/null 2>&1 || { warn "ffmpeg missing. Installing via brew …"; brew install ffmpeg; }

PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [[ $PYTHON_MINOR -lt 10 ]]; then
    error "Python 3.10+ required (found 3.${PYTHON_MINOR}). Use pyenv or brew."
fi

# ── 1. Virtual environment ───────────────────────────────────────────────────

VENV_DIR="$(pwd)/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment at .venv …"
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Activated virtualenv: $VENV_DIR"

# ── 2. pip upgrade ──────────────────────────────────────────────────────────

pip install --upgrade pip wheel setuptools -q

# ── 3. PyTorch (MPS – Apple Silicon) ────────────────────────────────────────

info "Installing PyTorch with MPS support …"
pip install torch torchvision torchaudio --quiet

# Quick MPS sanity check
python3 - <<'PYCHECK'
import torch
if torch.backends.mps.is_available():
    print("  MPS backend: AVAILABLE ✓")
else:
    print("  MPS backend: NOT available – will use CPU")
PYCHECK

# ── 4. MLX ──────────────────────────────────────────────────────────────────

info "Installing MLX and mlx-whisper …"
pip install mlx mlx-whisper --quiet

# ── 5. Project requirements ──────────────────────────────────────────────────

info "Installing project requirements …"
# Install without strict version pinning first; let pip resolve
pip install -r requirements.txt --quiet

# ── 6. Coqui TTS ────────────────────────────────────────────────────────────

info "Installing Coqui TTS (XTTS v2) …"
# TTS sometimes needs specific build deps
pip install TTS --quiet 2>/dev/null || {
    warn "TTS pip install had warnings — checking if import works …"
    python3 -c "from TTS.api import TTS; print('  Coqui TTS: OK ✓')" || \
        error "TTS import failed. Try: pip install TTS --no-deps and resolve manually."
}
python3 -c "from TTS.api import TTS; print('  Coqui TTS: OK ✓')"

# ── 7. Download XTTS v2 model ─────────────────────────────────────────────

info "Pre-downloading XTTS v2 model (≈1.8 GB, one-time) …"
python3 - <<'PYXTTS'
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
print("  XTTS v2 model: ready ✓")
PYXTTS

# ── 8. Download Whisper MLX model ────────────────────────────────────────────

info "Pre-fetching Whisper large-v3 MLX weights (≈1.5 GB) …"
python3 - <<'PYWHISPER'
import mlx_whisper
# Trigger model download by transcribing a tiny silent file
import numpy as np, tempfile, soundfile as sf
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    sf.write(f.name, np.zeros(16000, dtype=np.float32), 16000)
    mlx_whisper.transcribe(f.name, path_or_hf_repo="mlx-community/whisper-large-v3-mlx", verbose=False)
print("  Whisper MLX model: ready ✓")
PYWHISPER

# ── 9. NLLB-200 model ────────────────────────────────────────────────────────

info "Pre-fetching NLLB-200-600M translation model (≈2.3 GB) …"
python3 - <<'PYNLLB'
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
print("  NLLB-200 model: ready ✓")
PYNLLB

# ── 10. SpeechBrain speaker model pre-download ────────────────────────────

info "Pre-fetching SpeechBrain ECAPA-TDNN speaker model (no token required) …"
python3 - <<PYSPKR
try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier
EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/spkrec-ecapa",
    run_opts={"device": "cpu"},
)
print("  SpeechBrain ECAPA-TDNN: ready ✓")
PYSPKR

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
info "Setup complete! No tokens or accounts required. To run:"
echo ""
echo "  source .venv/bin/activate"
echo "  python main.py --input input/myvideo.mp4 --target-lang hi"
echo ""
