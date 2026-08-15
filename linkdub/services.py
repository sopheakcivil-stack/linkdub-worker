from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any, Iterable

import edge_tts
from deep_translator import GoogleTranslator

from .config import LANGUAGES, SETTINGS, Settings
from .models import Segment


class ProcessingServiceError(RuntimeError):
    pass


SPLIT_PUNCTUATION = re.compile(r"[ã€‚ï¼ï¼Ÿ!?ï¼›;â€¦]$")


def transcribe_chinese(
    audio_path: Path,
    settings: Settings = SETTINGS,
) -> list[Segment]:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        settings.whisper_model,
        device="cpu",
        compute_type=settings.whisper_compute_type,
        cpu_threads=0,
        num_workers=1,
    )
    raw_segments, _ = model.transcribe(
        str(audio_path),
        language="zh",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 450},
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    result: list[Segment] = []
    for raw in raw_segments:
        words = list(raw.words or [])
        if not words:
            text = raw.text.strip()
            if text:
                result.append(Segment(float(raw.start), float(raw.end), text))
            continue
        result.extend(_split_words(words, float(raw.start), float(raw.end)))
    if not result:
        raise ProcessingServiceError("No Chinese speech was detected in the video")
    return result


def _split_words(words: list[Any], fallback_start: float, fallback_end: float) -> list[Segment]:
    output: list[Segment] = []
    text_parts: list[str] = []
    start = float(words[0].start if words[0].start is not None else fallback_start)
    end = start
    for word in words:
        word_start = float(word.start if word.start is not None else end)
        word_end = float(word.end if word.end is not None else word_start + 0.1)
        token = str(word.word)
        if not text_parts:
            start = word_start
        text_parts.append(token)
        end = word_end
        text = "".join(text_parts).strip()
        should_split = (
            SPLIT_PUNCTUATION.search(text) is not None
            or end - start >= 8.0
            or len(text) >= 72
        )
        if should_split and text:
            output.append(Segment(start, max(start + 0.05, end), text))
            text_parts = []
    remaining = "".join(text_parts).strip()
    if remaining:
        output.append(Segment(start, max(start + 0.05, end or fallback_end), remaining))
    return output


def translate_segments(segments: list[Segment], target_language: str) -> list[Segment]:
    language = LANGUAGES.get(target_language)
    if not language:
        raise ProcessingServiceError(f"Unsupported target language: {target_language}")
    translator = GoogleTranslator(source="zh-CN", target=language["translate"])
    for index, segment in enumerate(segments):
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                translated = translator.translate(segment.source_text)
                if not translated or not translated.strip():
                    raise ProcessingServiceError("Translation service returned empty text")
                segment.translated_text = translated.strip()
                break
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
        else:
            raise ProcessingServiceError(
                f"Translation failed for segment {index + 1}: {last_error}"
            )
    return segments


async def _save_voice(text: str, voice: str, path: Path) -> None:
    communicator = edge_tts.Communicate(text=text, voice=voice)
    await communicator.save(str(path))


def generate_voice(text: str, target_language: str, path: Path) -> None:
    language = LANGUAGES.get(target_language)
    if not language:
        raise ProcessingServiceError(f"Unsupported target language: {target_language}")
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            asyncio.run(_save_voice(text, language["voice"], path))
            if path.exists() and path.stat().st_size > 0:
                return
            raise ProcessingServiceError("Text-to-speech service returned an empty audio file")
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
    raise ProcessingServiceError(f"Voice generation failed: {last_error}")


def _srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: Iterable[Segment], destination: Path, translated: bool) -> None:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = segment.translated_text if translated else segment.source_text
        blocks.append(
            f"{index}\n{_srt_time(segment.start)} --> {_srt_time(segment.end)}\n{text}\n"
        )
    destination.write_text("\n".join(blocks), encoding="utf-8")

