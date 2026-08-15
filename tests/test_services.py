from __future__ import annotations

from types import SimpleNamespace

from linkdub.services import _split_words, _srt_time


def word(text: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(word=text, start=start, end=end)


def test_split_words_uses_chinese_sentence_punctuation() -> None:
    segments = _split_words(
        [
            word("ä½ å¥½", 0.0, 0.5),
            word("ã€‚", 0.5, 0.7),
            word("ä¸–ç•Œ", 1.0, 1.5),
            word("ï¼", 1.5, 1.7),
        ],
        0.0,
        1.7,
    )
    assert [(segment.source_text, segment.start, segment.end) for segment in segments] == [
        ("ä½ å¥½ã€‚", 0.0, 0.7),
        ("ä¸–ç•Œï¼", 1.0, 1.7),
    ]


def test_srt_time_rounds_to_milliseconds() -> None:
    assert _srt_time(3661.2346) == "01:01:01,235"

