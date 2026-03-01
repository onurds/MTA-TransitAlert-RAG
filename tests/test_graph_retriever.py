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
