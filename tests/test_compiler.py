from __future__ import annotations

from pipeline.compiler import AlertCompiler, CompileRequest
from pipeline.temporal_resolver import TemporalResolver


def _header_text(compiled: dict) -> str:
    t = (compiled.get("header_text") or {}).get("translation") or []
    if isinstance(t, list) and t:
        return str((t[0] or {}).get("text") or "")
    return ""


def _description_text(compiled: dict) -> str:
    d = compiled.get("description_text") or {}
    t = d.get("translation") if isinstance(d, dict) else []
    if isinstance(t, list) and t:
        return str((t[0] or {}).get("text") or "")
    return ""


def _translation_langs(node: dict) -> set[str]:
    t = (node or {}).get("translation") or []
    langs = set()
    if isinstance(t, list):
        for row in t:
            if isinstance(row, dict) and row.get("language"):
                langs.add(str(row.get("language")))
    return langs


def _entities(compiled: dict) -> list:
    return compiled.get("informed_entity", []) if isinstance(compiled, dict) else []


def _periods(compiled: dict) -> list:
    return compiled.get("active_period", []) if isinstance(compiled, dict) else []


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
    compiler = _build_compiler_no_llm()
    request = CompileRequest(
        instruction=(
            "Northbound B20 buses are detoured due to utility work on Decatur St at Wilson Ave. "
            "Buses will not make stops on Decatur St from Bushwick Ave to Irving Ave. "
            "From 09:00 PM to 10:00 PM today."
        )
    )

    compiled = compiler.compile(request)

    assert compiled["id"].startswith("lmm:")
    assert _header_text(compiled)
    assert "description_text" in compiled
    assert _translation_langs(compiled.get("header_text", {})) >= {"en", "en-html", "zh", "es"}
    if compiled.get("description_text"):
        assert _translation_langs(compiled.get("description_text", {})) >= {"en", "en-html", "zh", "es"}
    assert "tts_header_text" in compiled
    assert "tts_description_text" in compiled
    assert compiled["cause"] in {"MAINTENANCE", "UNKNOWN_CAUSE"}
    assert compiled["effect"] in {"DETOUR", "MODIFIED_SERVICE", "UNKNOWN_EFFECT"}
    assert isinstance(_periods(compiled), list) and _periods(compiled)
    assert any(e.get("route_id") == "B20" for e in _entities(compiled))


def test_ambiguous_text_keeps_unknown_enums():
    compiler = _build_compiler_no_llm()
    request = CompileRequest(instruction="Q trains are operating. Regular service update.")
    compiled = compiler.compile(request)

    assert compiled["cause"] == "UNKNOWN_CAUSE"
    assert compiled["effect"] == "UNKNOWN_EFFECT"


def test_temporal_today_range():
    resolver = TemporalResolver(calendar_path="data/2026_english_calendar.csv", timezone="America/New_York")
    period = resolver.resolve("from 09:00 PM to 10:00 PM today")
    assert period is not None
    assert period.start.endswith("21:00:00")
    assert period.end.endswith("22:00:00")


def test_no_false_a_route_and_no_a_brackets():
    compiler = _build_compiler_no_llm()
    request = CompileRequest(
        instruction=(
            "Downtown [1] trains are running with delays after we addressed a door problem "
            "on a train at Cathedral Pkwy-110 St. Dates are tomorrow from 7 PM to 11 PM."
        )
    )
    compiled = compiler.compile(request)

    assert "[a]" not in _header_text(compiled).lower()
    routes = {e.get("route_id") for e in _entities(compiled) if e.get("route_id")}
    assert "1" in routes
    assert "A" not in routes


def test_subway_directional_stop_collapses_without_explicit_direction():
    compiler = _build_compiler_no_llm()
    request = CompileRequest(
        instruction=(
            "Downtown [1] trains are running with delays after we addressed a door problem "
            "on a train at Cathedral Pkwy-110 St."
        )
    )
    compiled = compiler.compile(request)

    stop_ids = {e.get("stop_id") for e in _entities(compiled) if e.get("stop_id")}
    assert "118" in stop_ids
    assert "118N" not in stop_ids
    assert "118S" not in stop_ids


def test_plain_prose_with_header_dates_description_words_not_treated_as_directives():
    compiler = _build_compiler_no_llm()
    request = CompileRequest(
        instruction=(
            "Operator note: the words header, dates, and description appear in this sentence naturally. "
            "Southbound Q52-SBS and Q53-SBS will not stop at stop id 553345 tomorrow 8pm to 11pm."
        )
    )
    compiled = compiler.compile(request)

    assert _header_text(compiled)
    stop_ids = {e.get("stop_id") for e in _entities(compiled) if e.get("stop_id")}
    assert "553345" in stop_ids


def test_explicit_stop_id_is_hard_locked_when_valid():
    compiler = _build_compiler_no_llm()
    request = CompileRequest(
        instruction=(
            "Southbound Q52-SBS and Q53-SBS will not stop at stop id 553345. "
            "timeframe is tomorrow 8pm to 11pm."
        )
    )

    compiled = compiler.compile(request)
    stop_ids = {e.get("stop_id") for e in _entities(compiled) if e.get("stop_id")}
    assert "553345" in stop_ids
    assert "982075" not in stop_ids
    assert "stop id 553345" not in _header_text(compiled).lower()
    assert any(e.get("route_id") == "Q52+" for e in _entities(compiled))
    assert any(e.get("route_id") == "Q53+" for e in _entities(compiled))


def test_invalid_explicit_stop_id_is_dropped():
    compiler = _build_compiler_no_llm()
    request = CompileRequest(
        instruction=(
            "Southbound Q52-SBS and Q53-SBS will not stop at stop id 9999999. "
            "timeframe is tomorrow 8pm to 11pm."
        )
    )
    compiled = compiler.compile(request)

    stop_ids = {e.get("stop_id") for e in _entities(compiled) if e.get("stop_id")}
    assert "9999999" not in stop_ids


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
    stop_ids = [e.get("stop_id") for e in _entities(compiled) if e.get("stop_id")]
    assert stop_ids == ["403573"]
    assert "see a map" not in _header_text(compiled).lower()


def test_output_key_order_and_nullable_fields_present():
    compiler = _build_compiler_no_llm()
    compiled = compiler.compile(
        CompileRequest(instruction="Southbound B20 buses are detoured at Decatur St at Wilson Ave from now to 2 hours after")
    )

    assert list(compiled.keys()) == [
        "id",
        "active_period",
        "informed_entity",
        "cause",
        "effect",
        "header_text",
        "description_text",
        "tts_header_text",
        "tts_description_text",
    ]
    assert "description_text" in compiled
    assert "tts_description_text" in compiled
    assert _description_text(compiled) == "" or isinstance(compiled["description_text"], dict) or compiled["description_text"] is None
