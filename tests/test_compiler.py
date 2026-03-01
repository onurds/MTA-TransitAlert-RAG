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
