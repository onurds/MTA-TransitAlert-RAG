from __future__ import annotations

from pipeline.compiler import AlertCompiler, CompileRequest
from pipeline.temporal_resolver import TemporalResolver


def _build_compiler() -> AlertCompiler:
    return AlertCompiler(
        graph_path="data/mta_knowledge_graph.gpickle",
        calendar_path="data/2026_english_calendar.csv",
        timezone="America/New_York",
        confidence_threshold=0.85,
    )


def _build_compiler_no_llm() -> AlertCompiler:
    compiler = _build_compiler()
    compiler._ensure_llm = lambda: False  # type: ignore[method-assign]
    compiler.retriever.geocode_fallback_entities = lambda **kwargs: {  # type: ignore[method-assign]
        "status": "skipped",
        "entities": [],
        "confidence": 0.0,
    }
    return compiler


def test_instruction_only_compile():
    compiler = _build_compiler()
    request = CompileRequest(
        instruction=(
            "header: Northbound B20 buses are detoured due to utility work on Decatur St at Wilson Ave, "
            "make description use: buses will not make stops on Decatur St from Bushwick Ave to Irving Ave. "
            "dates are from 09:00 PM to 10:00 PM today."
        )
    )

    compiled = compiler.compile(request)

    assert compiled["id"].startswith("lmm:")
    assert compiled["header"]
    assert compiled["description"]
    assert "severity" in compiled and compiled["severity"] is None
    assert compiled["cause"] in {"MAINTENANCE", "UNKNOWN_CAUSE"}
    assert compiled["effect"] in {"DETOUR", "MODIFIED_SERVICE", "UNKNOWN_EFFECT"}
    assert isinstance(compiled["active_periods"], list) and compiled["active_periods"]
    assert any(e.get("route_id") == "B20" for e in compiled["informed_entities"])


def test_alert_plus_instruction_preserves_id_and_applies_override():
    compiler = _build_compiler()
    request = CompileRequest(
        alert={
            "id": "lmm:alert:508367",
            "header": "[Q] trains are running with delays.",
            "description": "Original description",
            "effect": "UNKNOWN_EFFECT",
            "cause": "UNKNOWN_CAUSE",
            "severity": None,
            "active_periods": [{"start": "2026-02-11T09:47:03", "end": "2026-02-11T10:25:54"}],
            "informed_entities": [{"agency_id": "MTASBWY", "route_id": "Q"}],
        },
        instruction="make description use: Updated description for riders.",
    )

    compiled = compiler.compile(request)
    assert compiled["id"] == "lmm:alert:508367"
    assert compiled["description"].startswith("Updated description")
    assert any(e.get("route_id") == "Q" for e in compiled["informed_entities"])


def test_ambiguous_text_keeps_unknown_enums():
    compiler = _build_compiler()
    request = CompileRequest(instruction="header: Q trains are operating. description: Regular service update.")
    compiled = compiler.compile(request)

    assert compiled["cause"] == "UNKNOWN_CAUSE"
    assert compiled["effect"] == "UNKNOWN_EFFECT"


def test_temporal_today_range():
    resolver = TemporalResolver(calendar_path="data/2026_english_calendar.csv", timezone="America/New_York")
    period = resolver.resolve("dates are from 09:00 PM to 10:00 PM today")
    assert period is not None
    assert period.start.endswith("21:00:00")
    assert period.end.endswith("22:00:00")


def test_no_false_a_route_and_no_a_brackets():
    compiler = _build_compiler()
    request = CompileRequest(
        instruction=(
            "Downtown [1] trains are running with delays after we addressed a door problem "
            "on a train at Cathedral Pkwy-110 St. Dates are tomorrow from 7 PM to 11 PM."
        )
    )
    compiled = compiler.compile(request)

    assert "[a]" not in compiled["header"].lower()
    routes = {e.get("route_id") for e in compiled["informed_entities"] if e.get("route_id")}
    assert "1" in routes
    assert "A" not in routes


def test_subway_directional_stop_collapses_without_explicit_direction():
    compiler = _build_compiler()
    request = CompileRequest(
        instruction=(
            "Downtown [1] trains are running with delays after we addressed a door problem "
            "on a train at Cathedral Pkwy-110 St."
        )
    )
    compiled = compiler.compile(request)

    stop_ids = {e.get("stop_id") for e in compiled["informed_entities"] if e.get("stop_id")}
    assert "118" in stop_ids
    assert "118N" not in stop_ids
    assert "118S" not in stop_ids


def test_freeform_dates_clause_does_not_leak_into_header_and_preserves_stop():
    compiler = _build_compiler_no_llm()
    request = CompileRequest(
        instruction=(
            "Southbound Q52-SBS and Q53-SBS stop on Cross Bay Blvd at Liberty Ave has been temporarily relocated "
            "down the block before 107th Ave. The dates should be from tomorrow 8PM and repeat every monday "
            "for 4 weeks at the same time frames, end hour is 10PM."
        )
    )

    compiled = compiler.compile(request)
    assert "dates should be" not in compiled["header"].lower()
    assert "what's happening" not in compiled["header"].lower()
    assert len(compiled["active_periods"]) == 4
    assert any(e.get("route_id") == "Q52+" for e in compiled["informed_entities"])
    assert any(e.get("route_id") == "Q53+" for e in compiled["informed_entities"])
    assert any(e.get("stop_id") == "553345" for e in compiled["informed_entities"])


def test_whats_happening_section_is_not_leaked_into_header():
    compiler = _build_compiler_no_llm()
    request = CompileRequest(
        instruction=(
            "Southbound Q52-SBS and Q53-SBS stop on Cross Bay Blvd at Liberty Ave has been temporarily relocated "
            "down the block before 107th Ave. What's happening? Rockaway Blvd Subway Station Accessibility Upgrades. "
            "Plan your trip at mta.info or use the MTA app. Dates: from right now to 2 hours after."
        )
    )

    compiled = compiler.compile(request)
    assert "what's happening" not in compiled["header"].lower()
    assert "plan your trip" not in compiled["header"].lower()
    assert any(e.get("stop_id") == "553345" for e in compiled["informed_entities"])


def test_natural_text_with_dates_token_does_not_require_directive_parsing():
    compiler = _build_compiler_no_llm()
    request = CompileRequest(
        instruction=(
            "Eastbound M66 stop on W 65th St at Columbus Ave is being bypassed; "
            "use the stop on W 65th at Broadway instead. "
            "The word dates appears naturally here, timeframe is tomorrow 8pm to 11pm."
        )
    )
    compiled = compiler.compile(request)

    assert compiled["header"]
    assert any(e.get("route_id") == "M66" for e in compiled["informed_entities"])
    assert compiled["active_periods"] and compiled["active_periods"][0]["start"].endswith("20:00:00")


def test_explicit_stop_id_is_hard_locked_when_valid():
    compiler = _build_compiler_no_llm()
    request = CompileRequest(
        instruction=(
            "Southbound Q52-SBS and Q53-SBS will not stop at stop id 553345. "
            "timeframe is tomorrow 8pm to 11pm."
        )
    )

    compiled = compiler.compile(request)
    stop_ids = {e.get("stop_id") for e in compiled["informed_entities"] if e.get("stop_id")}
    assert "553345" in stop_ids
    assert "982075" not in stop_ids
    assert "stop id 553345" not in compiled["header"].lower()
    assert "cross bay" in compiled["header"].lower()
    assert any(e.get("route_id") == "Q52+" for e in compiled["informed_entities"])
    assert any(e.get("route_id") == "Q53+" for e in compiled["informed_entities"])


def test_invalid_explicit_stop_id_is_dropped():
    compiler = _build_compiler_no_llm()
    request = CompileRequest(
        instruction=(
            "Southbound Q52-SBS and Q53-SBS will not stop at stop id 9999999. "
            "timeframe is tomorrow 8pm to 11pm."
        )
    )
    compiled = compiler.compile(request)

    stop_ids = {e.get("stop_id") for e in compiled["informed_entities"] if e.get("stop_id")}
    assert "9999999" not in stop_ids


def test_freeform_timeframe_is_not_leaked_to_header():
    compiler = _build_compiler_no_llm()
    request = CompileRequest(
        instruction="Southbound B20 buses are detoured on Decatur St at Wilson Ave, timeframe is tomorrow 8pm to 11pm."
    )
    compiled = compiler.compile(request)
    assert "timeframe is" not in compiled["header"].lower()


def test_m66_bypass_uses_single_affected_stop_and_until_friday_window():
    compiler = _build_compiler_no_llm()
    request = CompileRequest(
        instruction=(
            "Eastbound M66 stop on W 65th St at Columbus Ave is being bypassed; "
            "use the stop on W 65th at Broadway instead. "
            "description: See a map of this stop change. What's happening?Building Construction Note: "
            "Bus arrival information may not be available/accurate while buses are detoured. "
            "Timeframe is from today timestamp until friday 10PM."
        )
    )
    compiled = compiler.compile(request)

    assert compiled["header"].startswith("Eastbound [M66] stop on W 65th St at Columbus Ave is being bypassed")
    assert "description:" not in compiled["header"].lower()
    assert "timeframe is" not in (compiled.get("description") or "").lower()
    stop_ids = [e.get("stop_id") for e in compiled["informed_entities"] if e.get("stop_id")]
    assert stop_ids == ["403573"]
    assert compiled["active_periods"] and compiled["active_periods"][0]["end"].endswith("22:00:00")


def test_m66_alternative_stop_phrase_does_not_override_affected_stop():
    compiler = _build_compiler_no_llm()
    request = CompileRequest(
        instruction=(
            "Eastbound M66 stop on W 65th St at Columbus Ave is being bypassed; "
            "use the stop on W 65th at Broadway instead. "
            "See a map of this stop change. Timeframe is from today timestamp until friday 10PM."
        )
    )
    compiled = compiler.compile(request)
    stop_ids = [e.get("stop_id") for e in compiled["informed_entities"] if e.get("stop_id")]
    assert stop_ids == ["403573"]
    assert "see a map" not in compiled["header"].lower()


def test_m66_mixed_single_clause_still_selects_columbus_affected_stop():
    compiler = _build_compiler_no_llm()
    request = CompileRequest(
        instruction=(
            "Eastbound M66 stop on W 65th St at Columbus Ave is being bypassed, "
            "use the stop on W 65th at Broadway instead. "
            "Timeframe is from today timestamp until friday 10PM."
        )
    )
    compiled = compiler.compile(request)
    stop_ids = [e.get("stop_id") for e in compiled["informed_entities"] if e.get("stop_id")]
    assert stop_ids == ["403573"]
