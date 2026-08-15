# LinkDub Worker

LinkDub's real background video-localization pipeline. It claims queued jobs
from Supabase, imports a public video, transcribes Chinese speech, splits and
translates timestamped segments, generates target-language speech, synchronizes
the voices, mixes audio, renders an MP4 with a selectable subtitle track, uploads
all results, and reports live progress.

## Pipeline

1. `yt-dlp` public-link import with direct-media fallback
2. FFmpeg audio extraction
3. Faster Whisper Chinese transcription with word timestamps and VAD
4. Punctuation/time-aware segment splitting
5. Chinese translation to English, Khmer, Thai, Vietnamese, French, or Spanish
6. Microsoft neural Edge TTS voices
7. Per-segment FFmpeg timing fit and a streaming PCM voice timeline
8. Original-audio ducking, or optional Demucs background isolation
9. MP4 render with translated subtitle track
10. Resumable signed uploads to Supabase Storage

The production GitHub Actions worker authenticates to Supabase using GitHub
OIDC. No database or storage secret is stored in this repository. The workflow
runs every five minutes and can also be dispatched manually. A Dockerfile is
included so the same worker can later move to a continuously running CPU/GPU
container without changing the processing pipeline.

## Local tests

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

Local queue execution intentionally requires GitHub Actions OIDC. Unit tests do
not contact Supabase or external AI services.

## Optional source separation

Install `requirements-demucs.txt` and set `ENABLE_DEMUCS=1`. Without Demucs,
LinkDub preserves background sound by ducking the original audio under the dub.
Demucs is disabled in the default CPU workflow because the PyTorch model makes
startup much slower; it is intended for the eventual persistent worker host.

