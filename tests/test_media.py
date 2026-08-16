from __future__ import annotations

import wave
from pathlib import Path

import pytest

from linkdub.media import (
    MediaError,
    _atempo_chain,
    build_timed_voice_track,
    render_final_video,
    split_speech_audio,
    validate_public_url,
)
from linkdub.models import Segment


def make_voice(path: Path, seconds: float, rate: int = 24000) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\x01\x00" * round(seconds * rate))


def test_rejects_private_and_non_http_urls() -> None:
    for url in ("file:///etc/passwd", "http://127.0.0.1/video.mp4", "ftp://example.com/a"):
        with pytest.raises(MediaError):
            validate_public_url(url)


def test_atempo_chain_supports_large_speedup() -> None:
    chain = _atempo_chain(5.0)
    assert chain.count("atempo=") == 3


def test_timeline_inserts_silence_and_has_exact_duration(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    output = tmp_path / "timeline.wav"
    make_voice(first, 0.5)
    make_voice(second, 0.5)
    segments = [
        Segment(0.5, 1.0, "一", "one"),
        Segment(1.5, 2.0, "二", "two"),
    ]
    build_timed_voice_track(segments, [first, second], 2.5, output)
    with wave.open(str(output), "rb") as audio:
        assert audio.getframerate() == 24000
        assert audio.getnframes() == 60000


def test_audio_is_split_into_equal_parallel_chunks(tmp_path: Path, monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr("linkdub.media.run", lambda command: commands.append(command))

    chunks = split_speech_audio(
        tmp_path / "speech.wav",
        tmp_path / "chunks",
        duration_seconds=400.0,
        chunk_count=4,
    )

    assert [offset for _, offset in chunks] == [0.0, 100.0, 200.0, 300.0]
    assert [path.name for path, _ in chunks] == [
        "speech-0.flac",
        "speech-1.flac",
        "speech-2.flac",
        "speech-3.flac",
    ]
    assert len(commands) == 4


def test_final_mix_uses_size_safe_audio_bitrate(tmp_path: Path, monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr("linkdub.media.has_audio", lambda *_args: True)
    monkeypatch.setattr("linkdub.media.run", lambda command: commands.append(command))

    render_final_video(
        tmp_path / "source.mp4",
        tmp_path / "voices.wav",
        tmp_path / "translated.srt",
        tmp_path / "result.mp4",
        duration_seconds=10_545.946,
        target_language="English",
    )

    bitrate_index = commands[0].index("-b:a")
    assert commands[0][bitrate_index + 1] == "96k"
