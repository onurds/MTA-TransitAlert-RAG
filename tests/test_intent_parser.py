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

    def link_route_mentions(self, mentions):
        mapping = {
            "q52-sbs": "Q52-SBS",
            "montauk branch": "5",
            "f": "F",
            "3": "3",
            "4": "4",
        }
        return [mapping[m.lower()] for m in mentions if m.lower() in mapping]


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
                    '"affected_route_mentions":[{"text":"Q52-SBS","source_span":"Southbound Q52-SBS","role":"affected_route"}],'
                    '"affected_stop_mentions":[{"text":"stop id 553345","source_span":"stop at stop id 553345","role":"affected_stop"}],'
                    '"alternative_route_mentions":[],'
                    '"alternative_stop_mentions":[],'
                    '"corridor_endpoints":[],'
                    '"effect_hint":"NO_SERVICE",'
                    '"cause_hint":null,'
                    '"style_intent":"moderate",'
                    '"parse_confidence":0.95,'
                    '"alternative_service_text":null'
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
            '"affected_route_mentions":[{"text":"4","source_span":"4","role":"affected_route"}],'
            '"affected_stop_mentions":[{"text":"New Lots Av","source_span":"New Lots Av","role":"affected_stop"}],'
            '"alternative_route_mentions":[{"text":"3","source_span":"3","role":"alternative_route"}],'
            '"alternative_stop_mentions":[],'
            '"corridor_endpoints":[],'
            '"effect_hint":"REPLACEMENT",'
            '"cause_hint":"PLANNED WORK",'
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


def test_intent_parser_discards_mentions_without_source_evidence():
    instruction = "Buses replace trains between Babylon and Montauk."
    llm = _FakeLLM(
        (
            '{'
            '"alert_text":"Buses replace trains between Babylon and Montauk.",'
            '"temporal_text":null,'
            '"explicit_route_ids":[],'
            '"explicit_stop_ids":[],'
            '"affected_route_mentions":[{"text":"Montauk Branch","source_span":"Montauk Branch","role":"affected_route"}],'
            '"affected_stop_mentions":['
            '{"text":"Babylon","source_span":"between Babylon and Montauk","role":"affected_stop"},'
            '{"text":"Montauk","source_span":"between Babylon and Montauk","role":"affected_stop"}'
            '],'
            '"alternative_route_mentions":[],'
            '"alternative_stop_mentions":[],'
            '"corridor_endpoints":['
            '{"text":"Babylon","source_span":"between Babylon and Montauk","role":"corridor_endpoint"},'
            '{"text":"Montauk","source_span":"between Babylon and Montauk","role":"corridor_endpoint"}'
            '],'
            '"effect_hint":"MODIFIED_SERVICE",'
            '"cause_hint":null,'
            '"style_intent":"moderate",'
            '"parse_confidence":0.95,'
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

    parsed = parser.parse(instruction)

    assert parsed.affected_route_mentions == ()
    assert [mention.text for mention in parsed.affected_stop_mentions] == ["Babylon", "Montauk"]
    assert [mention.text for mention in parsed.corridor_endpoints] == ["Babylon", "Montauk"]
    assert parsed.location_phrases == ("Babylon", "Montauk")


def test_intent_parser_does_not_promote_alternative_explicit_route_ids():
    instruction = "No F between Church Av and Coney Island-Stillwell Av. Take the D instead."
    llm = _FakeLLM(
        (
            '{'
            '"alert_text":"No F between Church Av and Coney Island-Stillwell Av.",'
            '"temporal_text":null,'
            '"explicit_route_ids":["F","D"],'
            '"explicit_stop_ids":[],'
            '"affected_route_mentions":[{"text":"F","source_span":"No F between","role":"affected_route"}],'
            '"affected_stop_mentions":['
            '{"text":"Church Av","source_span":"Church Av","role":"affected_stop"},'
            '{"text":"Coney Island Stillwell Av","source_span":"Coney Island Stillwell Av","role":"affected_stop"}'
            '],'
            '"alternative_route_mentions":[{"text":"D","source_span":"Take the D instead","role":"alternative_route"}],'
            '"alternative_stop_mentions":[],'
            '"corridor_endpoints":['
            '{"text":"Church Av","source_span":"Church Av","role":"corridor_endpoint"},'
            '{"text":"Coney Island Stillwell Av","source_span":"Coney Island Stillwell Av","role":"corridor_endpoint"}'
            '],'
            '"effect_hint":"NO_SERVICE",'
            '"cause_hint":null,'
            '"style_intent":"moderate",'
            '"parse_confidence":0.95,'
            '"alternative_service_text":"Take the D instead."'
            '}'
        )
    )
    parser = IntentParser(
        retriever=_FakeRetriever(),
        ensure_llm=lambda: True,
        llm_getter=lambda: llm,
        bump_telemetry=lambda key, amount: None,
    )

    parsed = parser.parse(instruction)

    assert parsed.explicit_route_ids == ("F",)
    assert [mention.text for mention in parsed.alternative_route_mentions] == ["D"]
    assert parsed.location_phrases == ("Church Av", "Coney Island Stillwell Av")
