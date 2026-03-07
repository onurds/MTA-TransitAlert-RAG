from __future__ import annotations

from pipeline.compiler.mercury_resolver import (
    FALLBACK_PRIORITY_KEY,
    FALLBACK_PRIORITY_NUMBER,
    MERCURY_PRIORITY_BY_KEY,
    MercuryResolver,
)


class _Resp:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, prompt: str):
        return _Resp(self._content)


def _resolver(content: str = '{"priority_key":"PLANNED_REROUTE","confidence":0.91}') -> MercuryResolver:
    llm = _FakeLLM(content)
    return MercuryResolver(ensure_llm=lambda: True, llm_getter=lambda: llm, min_confidence=0.6)


def test_mercury_catalog_matches_proto_priorities():
    assert len(MERCURY_PRIORITY_BY_KEY) == 35
    assert MERCURY_PRIORITY_BY_KEY["PLANNED_REROUTE"] == 18
    assert MERCURY_PRIORITY_BY_KEY["SERVICE_CHANGE"] == 22
    assert MERCURY_PRIORITY_BY_KEY["SUSPENDED"] == 35


def test_alert_type_humanization_is_canonical():
    assert MercuryResolver.alert_type_for_key("PLANNED_REROUTE") == "Planned - Reroute"
    assert MercuryResolver.alert_type_for_key("DELAYS_AND_CANCELLATIONS") == "Delays And Cancellations"
    assert MercuryResolver.alert_type_for_key("SERVICE_CHANGE") == "Service Change"


def test_invalid_or_low_confidence_resolution_falls_back_to_service_change():
    resolver = _resolver('{"priority_key":"NOT_A_REAL_VALUE","confidence":0.95}')
    out = resolver.resolve(
        instruction="A trains are running with changes",
        header_text="A trains are running with changes",
        description_text=None,
        cause="UNKNOWN_CAUSE",
        effect="UNKNOWN_EFFECT",
        active_periods=[{"start": "2026-03-07T09:05:00", "end": "2026-03-07T13:05:00"}],
        informed_entities=[{"agency_id": "MTASBWY", "route_id": "A"}],
    )
    assert out.priority_key == FALLBACK_PRIORITY_KEY
    assert out.priority_number == FALLBACK_PRIORITY_NUMBER
    assert out.alert_type == "Service Change"

    low_confidence = _resolver('{"priority_key":"PLANNED_REROUTE","confidence":0.2}')
    out2 = low_confidence.resolve(
        instruction="A trains are running with changes",
        header_text="A trains are running with changes",
        description_text=None,
        cause="UNKNOWN_CAUSE",
        effect="UNKNOWN_EFFECT",
        active_periods=[{"start": "2026-03-07T09:05:00", "end": "2026-03-07T13:05:00"}],
        informed_entities=[{"agency_id": "MTASBWY", "route_id": "A"}],
    )
    assert out2.priority_number == FALLBACK_PRIORITY_NUMBER


def test_mercury_resolution_accepts_priority_number():
    resolver = _resolver('{"priority_number":18,"confidence":0.93}')
    out = resolver.resolve(
        instruction="A trains run via F",
        header_text="A trains run via F",
        description_text=None,
        cause="MAINTENANCE",
        effect="MODIFIED_SERVICE",
        active_periods=[{"start": "2026-03-07T09:05:00", "end": "2026-03-07T13:05:00"}],
        informed_entities=[{"agency_id": "MTASBWY", "route_id": "A"}],
    )
    assert out.priority_key == "PLANNED_REROUTE"
    assert out.priority_number == 18
    assert out.alert_type == "Planned - Reroute"


def test_annotate_entities_adds_sort_order_only_to_route_entities():
    resolver = _resolver()
    entities = [
        {"agency_id": "MTASBWY", "route_id": "A"},
        {"agency_id": "MTASBWY", "stop_id": "A33"},
    ]
    annotated = resolver.annotate_entities(entities, priority_number=18)

    assert annotated[0]["mercury_entity_selector"]["sort_order"] == "MTASBWY:A:18"
    assert "mercury_entity_selector" not in annotated[1]


def test_human_readable_active_period_formats_same_day_single_window():
    resolver = _resolver()
    text = resolver.human_readable_active_period(
        [{"start": "2026-03-07T09:05:00", "end": "2026-03-07T13:05:00"}]
    )
    assert text == "Mar 7, Sat 9:05 AM to 1:05 PM"


def test_human_readable_active_period_formats_multi_day_window():
    resolver = _resolver()
    text = resolver.human_readable_active_period(
        [{"start": "2026-03-06T23:45:00", "end": "2026-03-09T05:00:00"}]
    )
    assert text == "Mar 6 - 9, Fri 11:45 PM to Mon 5:00 AM"


def test_human_readable_active_period_formats_weekly_recurrence():
    resolver = _resolver()
    text = resolver.human_readable_active_period(
        [
            {"start": "2026-05-03T22:45:00", "end": "2026-05-04T00:00:00"},
            {"start": "2026-05-10T22:45:00", "end": "2026-05-11T00:00:00"},
            {"start": "2026-05-17T22:45:00", "end": "2026-05-18T00:00:00"},
        ]
    )
    assert text == "Sundays in May from 10:45 PM to midnight"


def test_human_readable_active_period_joins_heterogeneous_periods():
    resolver = _resolver()
    text = resolver.human_readable_active_period(
        [
            {"start": "2026-03-07T09:05:00", "end": "2026-03-07T13:05:00"},
            {"start": "2026-03-08T10:00:00", "end": "2026-03-08T11:30:00"},
        ]
    )
    assert text == "Mar 7, Sat 9:05 AM to 1:05 PM; Mar 8, Sun 10:00 AM to 11:30 AM"


def test_human_readable_active_period_uses_until_further_notice_when_no_timeframe():
    resolver = _resolver()
    text = resolver.human_readable_active_period(
        [{"start": "2026-03-07T09:05:00"}],
        no_timeframe_mentioned=True,
    )
    assert text == "Until further notice"
