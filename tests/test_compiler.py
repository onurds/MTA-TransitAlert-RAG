from __future__ import annotations
import re

from pipeline.compiler import AlertCompiler, CompileRequest
from pipeline.compiler.utils import normalize_entities_for_output
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


def _mercury_alert(compiled: dict) -> dict:
    return compiled.get("mercury_alert", {}) if isinstance(compiled, dict) else {}


def _build_compiler() -> AlertCompiler:
    return AlertCompiler(
        graph_path="data/mta_knowledge_graph.gpickle",
        calendar_path="data/2026_english_calendar.csv",
        timezone="America/New_York",
        confidence_threshold=0.85,
    )


def test_temporal_today_range():
    resolver = TemporalResolver(calendar_path="data/2026_english_calendar.csv", timezone="America/New_York")
    period = resolver.resolve("from 09:00 PM to 10:00 PM today")
    assert period is not None
    assert period.start.endswith("21:00:00")
    assert period.end.endswith("22:00:00")


def test_gs_token_does_not_block_directional_stop_collapse():
    entities = [
        {"agency_id": "MTASBWY", "route_id": "GS"},
        {"agency_id": "MTASBWY", "stop_id": "902N"},
    ]
    normalized = normalize_entities_for_output(entities, "GS service runs overnight")
    stop_ids = {e.get("stop_id") for e in normalized if e.get("stop_id")}
    assert "902" in stop_ids
    assert "902N" not in stop_ids




def test_output_key_order_and_nullable_fields_present():
    """This test requires a working LLM. Skip if LLM is unavailable."""
    compiler = _build_compiler()
    try:
        compiled = compiler.compile(
            CompileRequest(instruction="Southbound B20 buses are detoured at Decatur St at Wilson Ave from now to 2 hours after")
        )
    except RuntimeError:
        import pytest
        pytest.skip("LLM unavailable, skipping integration test")

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
        "mercury_alert",
    ]
    assert "description_text" in compiled
    assert "tts_description_text" in compiled
    assert "mercury_alert" in compiled
    mercury_alert = _mercury_alert(compiled)
    assert mercury_alert.get("display_before_active") == "3600"
    assert re.fullmatch(r"\d+", str(mercury_alert.get("created_at", "")))
    assert re.fullmatch(r"\d+", str(mercury_alert.get("updated_at", "")))
    assert isinstance(mercury_alert.get("human_readable_active_period"), dict)
    for entity in _entities(compiled):
        if not isinstance(entity, dict):
            continue
        selector = entity.get("mercury_entity_selector")
        if entity.get("route_id"):
            assert isinstance(selector, dict)
            assert re.fullmatch(r"[^:]+:[^:]+:\d+", str(selector.get("sort_order", "")))
        else:
            assert selector is None
    assert _description_text(compiled) == "" or isinstance(compiled["description_text"], dict) or compiled["description_text"] is None


def test_compile_fails_without_llm():
    """Verify that the compiler raises RuntimeError when LLM is not available."""
    compiler = _build_compiler()
    compiler.llm = None

    def _fail_ensure():
        raise RuntimeError("LLM unavailable (test)")

    # Patch ensure_llm on all sub-components that captured it at init time
    compiler._ensure_llm = _fail_ensure  # type: ignore[method-assign]
    compiler.intent_parser.ensure_llm = _fail_ensure
    compiler.entity_selector.ensure_llm = _fail_ensure
    compiler.enum_resolver.ensure_llm = _fail_ensure
    compiler.mercury_resolver.ensure_llm = _fail_ensure

    import pytest
    with pytest.raises(RuntimeError, match="LLM unavailable"):
        compiler.compile(
            CompileRequest(instruction="Southbound Q52-SBS will not stop at stop id 553345.")
        )
