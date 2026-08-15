from __future__ import annotations

from types import SimpleNamespace

from linkdub.config import Settings
from linkdub.models import Segment
from linkdub.services import _split_words, _srt_time, translate_segments


def word(text: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(word=text, start=start, end=end)


def test_split_words_uses_chinese_sentence_punctuation() -> None:
    segments = _split_words(
        [
            word("你好", 0.0, 0.5),
            word("。", 0.5, 0.7),
            word("世界", 1.0, 1.5),
            word("！", 1.5, 1.7),
        ],
        0.0,
        1.7,
    )
    assert [(segment.source_text, segment.start, segment.end) for segment in segments] == [
        ("你好。", 0.0, 0.7),
        ("世界！", 1.0, 1.7),
    ]


def test_srt_time_rounds_to_milliseconds() -> None:
    assert _srt_time(3661.2346) == "01:01:01,235"


def test_translation_preserves_segment_order_with_parallel_workers(monkeypatch) -> None:
    class Translator:
        def __init__(self, **_kwargs):
            pass

        def translate(self, text: str) -> str:
            return f"translated-{text}"

    monkeypatch.setattr("linkdub.services.GoogleTranslator", Translator)
    segments = [Segment(0, 1, "一"), Segment(1, 2, "二"), Segment(2, 3, "三")]

    translate_segments(segments, "English", Settings(translation_workers=3))

    assert [segment.translated_text for segment in segments] == [
        "translated-一",
        "translated-二",
        "translated-三",
    ]
