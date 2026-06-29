from __future__ import annotations

from pipeline.compiler.intent_parser import IntentParser


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, prompt: str):
        return _Resp(self._content)


class _FakeRetriever:
    def _route_tokens_from_text(self, text: str):
        return []


def test_intent_parser_accepts_gemini_style_block_content():
    llm = _FakeLLM(
        [
            {
                "type": "text",
                "text": (
                    '{'
                    '"alert_text":"Southbound Q52-SBS will not stop at stop id 553345.",'
                    '"temporal_text":"tomorrow 8pm to 11pm",'
                    '"explicit_route_ids":["Q52-SBS"],'
                    '"explicit_stop_ids":["553345"],'
                    '"location_phrases":["stop id 553345"],'
                    '"effect_hint":"NO_SERVICE",'
                    '"cause_hint":null,'
                    '"style_intent":"moderate",'
                    '"parse_confidence":0.95'
                    '}'
                ),
            }
        ]
    )
    parser = IntentParser(
        retriever=_FakeRetriever(),
        ensure_llm=lambda: True,
        llm_getter=lambda: llm,
        bump_telemetry=lambda key, amount: None,
    )

    parsed = parser.parse(
        "Southbound Q52-SBS will not stop at stop id 553345. timeframe is tomorrow 8pm to 11pm."
    )

    assert parsed.alert_text == "Southbound Q52-SBS will not stop at stop id 553345."
    assert parsed.temporal_text == "tomorrow 8pm to 11pm"
    assert parsed.explicit_route_ids == ("Q52-SBS",)
    assert parsed.explicit_stop_ids == ("553345",)
    assert parsed.location_phrases == ("stop id 553345",)
    assert parsed.effect_hint == "NO_SERVICE"


def test_replacement_route_policy_treats_replacement_as_affected():
    llm = _FakeLLM(
        (
            '{'
            '"alert_text":"Late night 3 replaces 4 service to/from New Lots Av",'
            '"temporal_text":"late night",'
            '"explicit_route_ids":["3","4"],'
            '"explicit_stop_ids":[],'
            '"location_phrases":["New Lots Av"],'
            '"effect_hint":"REPLACEMENT",'
            '"cause_hint":"PLANNED_WORK",'
            '"style_intent":"moderate",'
            '"parse_confidence":0.85,'
            '"alternative_service_text":null'
            '}'
        )
    )
    parser = IntentParser(
        retriever=_FakeRetriever(),
        ensure_llm=lambda: True,
        llm_getter=lambda: llm,
        bump_telemetry=lambda key, amount: None,
    )

    parsed = parser.parse("Late night 3 replaces 4 service to/from New Lots Av.")

    assert parsed.explicit_route_ids == ("3",)
