from __future__ import annotations

from pipeline.compiler.entity_selector import EntitySelector


class _Resp:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return _Resp(self._content)


def test_single_location_tie_uses_llm_tiebreaker():
    llm = _FakeLLM('{"selected_stop_id":"S2","confidence":0.91}')
    selector = EntitySelector(
        ensure_llm=lambda: True,
        llm_getter=lambda: llm,
    )
    stop_entities = [
        {"agency_id": "MTA NYCT", "stop_id": "S1"},
        {"agency_id": "MTA NYCT", "stop_id": "S2"},
    ]
    stop_candidates = [
        {"stop_id": "S1", "route_id": "B20", "stop_name": "Decatur St / Wilson Ave", "score": 0.83},
        {"stop_id": "S2", "route_id": "B20", "stop_name": "Hospital Main Entrance", "score": 0.80},
    ]

    chosen = selector.choose_stops_for_single_location(
        stop_entities=stop_entities,
        source_text="B20 stop near the hospital is closed.",
        stop_candidates=stop_candidates,
        location_hints=["hospital"],
        tie_delta=0.05,
    )

    assert len(chosen) == 1
    assert chosen[0]["stop_id"] == "S2"
    assert len(llm.prompts) == 1


def test_single_location_large_gap_skips_llm_and_keeps_top_scored_stop():
    llm = _FakeLLM('{"selected_stop_id":"S2","confidence":0.91}')
    selector = EntitySelector(
        ensure_llm=lambda: True,
        llm_getter=lambda: llm,
    )
    stop_entities = [
        {"agency_id": "MTA NYCT", "stop_id": "S1"},
        {"agency_id": "MTA NYCT", "stop_id": "S2"},
    ]
    stop_candidates = [
        {"stop_id": "S1", "route_id": "B20", "stop_name": "Decatur St / Wilson Ave", "score": 0.91},
        {"stop_id": "S2", "route_id": "B20", "stop_name": "Hospital Main Entrance", "score": 0.72},
    ]

    chosen = selector.choose_stops_for_single_location(
        stop_entities=stop_entities,
        source_text="B20 stop near the hospital is closed.",
        stop_candidates=stop_candidates,
        location_hints=["hospital"],
        tie_delta=0.05,
    )

    assert len(chosen) == 1
    assert chosen[0]["stop_id"] == "S1"
    assert llm.prompts == []


def test_single_location_near_tie_different_routes_skips_llm():
    llm = _FakeLLM('{"selected_stop_id":"S2","confidence":0.91}')
    selector = EntitySelector(
        ensure_llm=lambda: True,
        llm_getter=lambda: llm,
    )
    stop_entities = [
        {"agency_id": "MTA NYCT", "stop_id": "S1"},
        {"agency_id": "MTA NYCT", "stop_id": "S2"},
    ]
    stop_candidates = [
        {"stop_id": "S1", "route_id": "B20", "stop_name": "Decatur St / Wilson Ave", "score": 0.83},
        {"stop_id": "S2", "route_id": "Q58", "stop_name": "Hospital Main Entrance", "score": 0.80},
    ]

    chosen = selector.choose_stops_for_single_location(
        stop_entities=stop_entities,
        source_text="Stop near the hospital is closed.",
        stop_candidates=stop_candidates,
        location_hints=["hospital"],
        tie_delta=0.05,
    )

    assert len(chosen) == 1
    assert chosen[0]["stop_id"] == "S1"
    assert llm.prompts == []
