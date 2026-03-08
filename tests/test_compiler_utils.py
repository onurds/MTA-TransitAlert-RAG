from __future__ import annotations

from pipeline.compiler.utils import extract_llm_text_content


class _Resp:
    def __init__(self, content):
        self.content = content


def test_extract_llm_text_content_returns_plain_string_unchanged():
    response = _Resp('{"ok": true}')
    assert extract_llm_text_content(response) == '{"ok": true}'


def test_extract_llm_text_content_extracts_text_from_gemini_style_blocks():
    response = _Resp(
        [
            {
                "type": "text",
                "text": '{"ok": true}',
                "extras": {"signature": "abc"},
            }
        ]
    )
    assert extract_llm_text_content(response) == '{"ok": true}'


def test_extract_llm_text_content_joins_multiple_text_blocks():
    response = _Resp(
        [
            {"type": "thinking", "text": ""},
            {"type": "text", "text": '{"a": 1,'},
            {"type": "text", "text": '"b": 2}'},
        ]
    )
    assert extract_llm_text_content(response) == '{"a": 1,\n"b": 2}'
