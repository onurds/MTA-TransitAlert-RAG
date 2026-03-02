from __future__ import annotations

from pipeline.graph_retriever import GraphRetriever


def _build_retriever() -> GraphRetriever:
    return GraphRetriever(graph_path="data/mta_knowledge_graph.gpickle")


def test_route_scope_excludes_alternative_route_mentions():
    retriever = _build_retriever()
    text = "[Q] trains are bypassing 86 St. Take the B instead."
    result = retriever.retrieve_affected_entities(text)

    assert result["status"] == "success"
    routes = {e.get("route_id") for e in result["informed_entities"] if e.get("route_id")}
    assert "Q" in routes
    assert "B" not in routes


def test_b20_detour_is_grounded_to_route():
    retriever = _build_retriever()
    text = (
        "Northbound B20 buses are detoured due to utility work on Decatur St at Wilson Ave. "
        "Buses will not make stops on Decatur St from Bushwick Ave to Irving Ave."
    )
    result = retriever.retrieve_affected_entities(text)

    assert result["status"] == "success"
    assert any(e.get("route_id") == "B20" for e in result["informed_entities"])


def test_q52_q53_relocation_extracts_intersection_stop_candidates():
    retriever = _build_retriever()
    text = (
        "Southbound Q52-SBS and Q53-SBS stop on Cross Bay Blvd at Liberty Ave has been temporarily relocated "
        "down the block before 107th Ave."
    )
    result = retriever.retrieve_affected_entities(text)

    assert result["status"] == "success"
    assert result["location_hints"] == ["Cross Bay Blvd", "Liberty Ave"]
    stop_ids = {c.get("stop_id") for c in result.get("stop_candidates", [])}
    assert "553345" in stop_ids


def test_bxm11_detour_location_hints_are_sanitized_for_fallback():
    retriever = _build_retriever()
    text = (
        "Southbound BxM11 buses are detouring from E 177th St at Bronx River Parkway "
        "to Bruckner Blvd at Bruckner Expy - no stops will be missed"
    )
    result = retriever.retrieve_affected_entities(text)

    assert result["status"] == "success"
    hints = result["location_hints"]
    assert "E 177th St at Bronx River Parkway" in hints
    assert "Bruckner Blvd at Bruckner Expy" in hints
    assert all(" to " not in hint.lower() for hint in result["location_hints"])


def test_from_to_location_hint_is_split_not_dropped():
    retriever = _build_retriever()
    merged = retriever._merge_location_hints([], ["Bronx River Parkway to Bruckner Blvd at Bruckner Expy"])
    assert "Bronx River Parkway" in merged
    assert "Bruckner Blvd at Bruckner Expy" in merged


def test_split_affected_and_alternative_for_mixed_single_clause():
    retriever = _build_retriever()
    text = "Eastbound M66 stop on W 65th St at Columbus Ave is being bypassed, use the stop on W 65th at Broadway instead"
    affected, alternative = retriever._split_affected_and_alternative_segments(text)
    assert any("Columbus Ave" in seg for seg in affected)
    assert any("Broadway" in seg for seg in alternative)


def test_explicit_id_validators():
    retriever = _build_retriever()
    assert retriever.is_valid_route_id("Q52-SBS")
    assert retriever.is_valid_stop_id("553345")
    assert not retriever.is_valid_stop_id("9999999")
