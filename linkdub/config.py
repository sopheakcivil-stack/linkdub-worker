from __future__ import annotations

import os
from dataclasses import dataclass


LANGUAGES: dict[str, dict[str, str]] = {
    "English": {"translate": "en", "voice": "en-US-AvaMultilingualNeural"},
    "Khmer": {"translate": "km", "voice": "km-KH-SreymomNeural"},
    "Thai": {"translate": "th", "voice": "th-TH-PremwadeeNeural"},
    "Vietnamese": {"translate": "vi", "voice": "vi-VN-HoaiMyNeural"},
    "French": {"translate": "fr", "voice": "fr-FR-DeniseNeural"},
    "Spanish": {"translate": "es", "voice": "es-ES-ElviraNeural"},
}


@dataclass(frozen=True, slots=True)
class Settings:
    worker_api_url: str = os.getenv(
        "LINKDUB_WORKER_API_URL",
        "https://icdcqkqerqrquxilvxgz.supabase.co/functions/v1/worker-api",
    )
    supabase_project_ref: str = os.getenv(
        "LINKDUB_SUPABASE_PROJECT_REF", "icdcqkqerqrquxilvxgz"
    )
    oidc_audience: str = os.getenv("LINKDUB_OIDC_AUDIENCE", "linkdub-worker")
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    whisper_batch_size: int = int(os.getenv("WHISPER_BATCH_SIZE", "8"))
    whisper_cpu_threads: int = int(os.getenv("WHISPER_CPU_THREADS", "4"))
    transcription_chunks: int = int(os.getenv("TRANSCRIPTION_CHUNKS", "4"))
    translation_workers: int = int(os.getenv("TRANSLATION_WORKERS", "8"))
    voice_workers: int = int(os.getenv("VOICE_WORKERS", "6"))
    max_output_bytes: int = int(os.getenv("MAX_OUTPUT_BYTES", "1900000000"))
    original_audio_volume: float = float(os.getenv("ORIGINAL_AUDIO_VOLUME", "0.18"))
    enable_demucs: bool = os.getenv("ENABLE_DEMUCS", "0") == "1"
    burn_subtitles: bool = os.getenv("BURN_SUBTITLES", "0") == "1"
    max_video_seconds: int = int(os.getenv("MAX_VIDEO_SECONDS", "10800"))
    ffmpeg_bin: str = os.getenv("FFMPEG_BIN", "ffmpeg")
    ffprobe_bin: str = os.getenv("FFPROBE_BIN", "ffprobe")

SETTINGS = Settings()
