from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / ".temp"

for _d in [MODELS_DIR, OUTPUT_DIR, TEMP_DIR]:
    _d.mkdir(exist_ok=True)

# Audio
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 300          # seconds per processing chunk
OVERLAP_DURATION = 10         # seconds of overlap between chunks
REFERENCE_MIN_DURATION = 5    # min seconds for reference audio
REFERENCE_MAX_DURATION = 20   # max seconds for reference audio

# Models
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx" # Change Model here
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
DIARIZATION_MODEL = "speechbrain/spkrec-ecapa-voxceleb"  # no token, no account needed
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# Device — torch imported lazily so config can be imported before torch is installed
def _get_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

DEVICE = _get_device()

# NLLB language codes
NLLB_LANG_CODES: dict[str, str] = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn",
    "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
    "ru": "rus_Cyrl", "zh": "zho_Hans", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "ar": "arb_Arab", "hi": "hin_Deva",
    "nl": "nld_Latn", "pl": "pol_Latn", "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
}

# XTTS language codes
XTTS_LANG_CODES: dict[str, str] = {
    "en": "en", "es": "es", "fr": "fr", "de": "de",
    "it": "it", "pt": "pt", "ru": "ru", "zh": "zh-cn",
    "ja": "ja", "ko": "ko", "ar": "ar", "hi": "hi",
    "nl": "nl", "pl": "pl", "tr": "tr", "uk": "uk",
}

# Queue / pipeline
QUEUE_MAXSIZE = 100
WORKER_TIMEOUT = 600           # seconds

# Segment constraints
MIN_SEGMENT_DURATION = 0.3
MAX_SEGMENT_DURATION = 25.0
MAX_TIME_STRETCH_RATIO = 1.6   # beyond this, insert silence instead of stretching


@dataclass
class Segment:
    """Core data unit flowing through the entire pipeline."""
    segment_id: str
    speaker_id: str
    start: float                       # seconds in original audio
    end: float
    chunk_index: int = 0
    audio_path: Optional[str] = None   # extracted segment WAV
    original_text: Optional[str] = None
    translated_text: Optional[str] = None
    dubbed_audio_path: Optional[str] = None
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start
