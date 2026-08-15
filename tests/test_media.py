from __future__ import annotations

import wave
from pathlib import Path

import pytest

from linkdub.media import MediaError, _atempo_chain, build_timed_voice_track, validate_public_url
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
        Segment(0.5, 1.0, "ä¸€", "one"),
        Segment(1.5, 2.0, "äºŒ", "two"),
    ]
    build_timed_voice_track(segments, [first, second], 2.5, output)
    with wave.open(str(output), "rb") as audio:
        assert audio.getframerate() == 24000
        assert audio.getnframes() == 60000

