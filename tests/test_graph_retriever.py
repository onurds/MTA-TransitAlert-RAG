from __future__ import annotations

import re

import pytest

from pipeline.graph_retriever import GraphRetriever


def _build_retriever() -> GraphRetriever:
    return GraphRetriever(graph_path="data/mta_knowledge_graph.gpickle")


def _station_stop_ids(result: dict) -> set[str]:
    out = set()
    for row in result.get("stop_candidates", []):
        stop_id = str(row.get("stop_id", "")).upper()
        if not stop_id:
            continue
        out.add(re.sub(r"[NSEW]$", "", stop_id))
    return out


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


def test_route_aliases_normalize_to_gtfs_codes():
    retriever = _build_retriever()
    assert retriever.normalize_route_id("SIR") == "SI"
    assert retriever.normalize_route_id("42 Street Shuttle") == "GS"
    assert retriever.normalize_route_id("Franklin Avenue Shuttle") == "FS"
    assert retriever.normalize_route_id("Rockaway Park Shuttle") == "H"
    assert retriever.normalize_route_id("B41-SBS") == "B41"
    assert retriever.normalize_route_id("Q52-SBS") == "Q52+"
    assert retriever.normalize_route_id("Q53-SBS") == "Q53+"

    result = retriever.retrieve_affected_entities("42 Street Shuttle trains are running with delays overnight.")
    assert result["status"] == "success"
    routes = {e.get("route_id") for e in result["informed_entities"] if e.get("route_id")}
    assert "GS" in routes


def test_data_driven_long_name_alias_for_f1_when_present():
    retriever = _build_retriever()
    if not retriever.is_valid_route_id("F1"):
        pytest.skip("F1 not present in this GTFS snapshot")

    hits = retriever.route_alias_matches("CULVER LINK F Shuttle Bus service change")
    assert "F1" in hits


def test_single_letter_route_alias_is_not_matched_from_plain_text():
    retriever = _build_retriever()
    hits = retriever.route_alias_matches("a service update near 69 st")
    assert hits == []


def test_route_long_name_schedule_text_does_not_infer_stops_without_stop_intent():
    retriever = _build_retriever()
    text = "Kings Hwy Station - Kings Plaza B2 buses run on a modified schedule for electrical improvements."
    result = retriever.retrieve_affected_entities(text)

    assert result["status"] == "success"
    routes = {e.get("route_id") for e in result["informed_entities"] if e.get("route_id")}
    stops = [e.get("stop_id") for e in result["informed_entities"] if e.get("stop_id")]
    assert "B2" in routes
    assert stops == []


def test_numeric_street_phrase_does_not_trigger_numeric_route_alias():
    retriever = _build_retriever()
    text = "In the Bronx, Westchester Sq-bound 6X express runs local from 3 Av-138 St to Parkchester."
    result = retriever.retrieve_affected_entities(text)

    assert result["status"] == "success"
    routes = {e.get("route_id") for e in result["informed_entities"] if e.get("route_id")}
    assert "3" not in routes


def test_prefixed_bus_route_list_expands_family_prefix_before_numeric_subway_routes():
    retriever = _build_retriever()
    text = (
        "These buses are delayed: B: 4, 4A, 5, 6, 8, 11, 12, 13, 17, 19, 21, 22, 24, 25, 26, 27, "
        "28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 41-SBS, 42. "
        "We're running as much service as we can with the buses and operators we have available."
    )
    route_ids = retriever._extract_route_ids(text)

    assert "B4" in route_ids
    assert "B6" in route_ids
    assert "B41" in route_ids
    assert "B42" in route_ids
    assert "4" not in route_ids
    assert "5" not in route_ids
    assert "6" not in route_ids


def test_primary_corridor_recovery_prefers_affected_4_route_over_shuttle_text():
    retriever = _build_retriever()
    text = (
        "In Brooklyn, no 4 between Crown Hts-Utica Av and New Lots Av. "
        "Free shuttle buses run between Crown Hts-Utica Av and New Lots Av."
    )
    result = retriever.retrieve_affected_entities(text)

    assert result["status"] == "success"
    assert result["primary_route_ids"] == ["4"]
    assert result["primary_location_hints"] == ["Crown Hts-Utica Av", "New Lots Av"]
    assert result["primary_corridor_expanded"] is True
    assert result["route_pattern_selected"].get("4")
    assert {"250", "251", "252", "253", "254", "255", "256", "257"}.issubset(
        _station_stop_ids(result)
    )


def test_primary_corridor_recovery_prefers_affected_g_route_over_alt_bus_text():
    retriever = _build_retriever()
    text = "No G between Bedford-Nostrand Avs and Court Sq. Take the B98 shuttle bus instead."
    result = retriever.retrieve_affected_entities(text)

    assert result["status"] == "success"
    assert result["route_ids"] == ["G"]
    assert result["primary_route_ids"] == ["G"]
    assert result["primary_corridor_expanded"] is True
    assert result["route_pattern_selected"].get("G")
    assert {"G22", "G24", "G26", "G28", "G29", "G30", "G31", "G32", "G33"}.issubset(
        _station_stop_ids(result)
    )


def test_seven_corridor_uses_pattern_selection_with_borough_qualifiers():
    retriever = _build_retriever()
    text = (
        "No 7 between 74 St-Broadway, Queens and 34 St-Hudson Yards, Manhattan. "
        "Use the E, F, R, N, 4, 5 or 6 instead."
    )
    result = retriever.retrieve_affected_entities(text)

    assert result["status"] == "success"
    assert result["route_ids"] == ["7"]
    assert result["primary_location_hints"] == ["74 St-Broadway", "34 St-Hudson Yards"]
    assert result["primary_corridor_expanded"] is True
    assert result["route_pattern_selected"].get("7")
    station_ids = _station_stop_ids(result)
    assert "711" in station_ids
    assert "713" in station_ids


def test_graph_routes_expose_stop_patterns_for_runtime_selection():
    retriever = _build_retriever()
    patterns = retriever.route_patterns_for_node("gtfs_subway_7")
    assert patterns
    assert any(pattern.pattern_id for pattern in patterns)
    assert any("711" in pattern.public_stop_ids for pattern in patterns)


def test_lirr_montauk_branch_is_inferred_from_station_pair():
    retriever = _build_retriever()
    text = "Buses replace trains between Babylon and Montauk."
    result = retriever.retrieve_affected_entities(text)

    assert result["status"] == "success"
    route_entities = [
        e for e in result["informed_entities"]
        if e.get("route_id") == "5" and not e.get("stop_id")
    ]
    assert route_entities
    assert route_entities[0]["agency_id"] == "LI"
    assert result["route_ids"] == ["5"]


def test_lirr_montauk_branch_is_inferred_from_llm_mentions_without_text_detection():
    retriever = _build_retriever()
    route_ids = retriever.infer_commuter_route_ids_from_mentions(["Babylon", "Montauk"])
    result = retriever.retrieve_affected_entities(
        "Buses replace trains between Babylon and Montauk.",
        route_ids_override=route_ids,
        location_hints_override=["Babylon", "Montauk"],
        primary_location_hints_override=["Babylon", "Montauk"],
        allow_text_route_detection=False,
        allow_text_location_detection=False,
    )

    assert route_ids == ["5"]
    assert result["status"] == "success"
    assert result["route_ids"] == ["5"]


def test_generic_commuter_words_do_not_create_commuter_context():
    retriever = _build_retriever()

    assert not retriever._has_commuter_rail_context("Check TrainTime for branch or line information.")


def test_lirr_greenport_service_is_inferred_from_station_pair():
    retriever = _build_retriever()
    text = "Buses replace trains between Ronkonkoma and Greenport."
    result = retriever.retrieve_affected_entities(text)

    assert result["status"] == "success"
    assert {"agency_id": "LI", "route_id": "13"} in [
        {k: v for k, v in e.items() if k in {"agency_id", "route_id"}}
        for e in result["informed_entities"]
        if not e.get("stop_id")
    ]
    assert result["route_ids"] == ["13"]


def test_lirr_city_terminal_zone_is_inferred_from_terminal_stations():
    retriever = _build_retriever()
    text = "No service at Forest Hills and Kew Gardens. Trains bypass Woodside."
    result = retriever.retrieve_affected_entities(text)

    assert result["status"] == "success"
    assert {"agency_id": "LI", "route_id": "12"} in [
        {k: v for k, v in e.items() if k in {"agency_id", "route_id"}}
        for e in result["informed_entities"]
        if not e.get("stop_id")
    ]
    assert result["route_ids"] == ["12"]


def test_lirr_montauk_pair_takes_precedence_over_babylon_branch_detail():
    retriever = _build_retriever()
    text = (
        "Buses replace trains between Babylon and Montauk. "
        "In most cases, buses connect in Babylon with regular Babylon Branch trains, "
        "and express diesel trains to/from Jamaica will not run."
    )
    result = retriever.retrieve_affected_entities(text)

    assert result["status"] == "success"
    assert result["route_ids"][0] == "5"
    assert {"agency_id": "LI", "route_id": "5"} in [
        {k: v for k, v in e.items() if k in {"agency_id", "route_id"}}
        for e in result["informed_entities"]
        if not e.get("stop_id")
    ]


def test_lirr_city_terminal_pair_takes_precedence_over_port_washington_detail():
    retriever = _build_retriever()
    text = (
        "No eastbound service at Forest Hills and Kew Gardens; eastbound trains towards Jamaica bypass Woodside. "
        "Port Washington Branch trains continue to stop at Woodside in both directions."
    )
    result = retriever.retrieve_affected_entities(text)

    assert result["status"] == "success"
    assert result["route_ids"][0] == "12"
    assert {"agency_id": "LI", "route_id": "12"} in [
        {k: v for k, v in e.items() if k in {"agency_id", "route_id"}}
        for e in result["informed_entities"]
        if not e.get("stop_id")
    ]


def test_mnr_line_aliases_resolve_to_commuter_routes():
    retriever = _build_retriever()

    harlem = retriever.retrieve_affected_entities("Harlem Line trains are delayed.")
    new_haven = retriever.retrieve_affected_entities("New Haven Line trains are delayed.")

    assert {"agency_id": "MNR", "route_id": "2"} in [
        {k: v for k, v in e.items() if k in {"agency_id", "route_id"}}
        for e in harlem["informed_entities"]
        if not e.get("stop_id")
    ]
    assert {"agency_id": "MNR", "route_id": "3"} in [
        {k: v for k, v in e.items() if k in {"agency_id", "route_id"}}
        for e in new_haven["informed_entities"]
        if not e.get("stop_id")
    ]


def test_llm_route_mention_linking_preserves_commuter_agency():
    retriever = _build_retriever()

    assert retriever.link_route_mention_entities(["Harlem Line"]) == [
        {"agency_id": "MNR", "route_id": "2"}
    ]
    assert retriever.link_route_mention_entities(["Montauk Branch"]) == [
        {"agency_id": "LI", "route_id": "5"}
    ]


def test_llm_route_mention_linking_normalizes_descriptive_subway_spans():
    retriever = _build_retriever()

    assert retriever.link_route_mentions(["downtown 4 local"]) == ["4"]
    assert retriever.link_route_mentions(["Coney Island-bound F", "Church Av-bound G"]) == ["F", "G"]
    assert retriever.link_route_mention_entities(["downtown 4 local"]) == [
        {"agency_id": "MTASBWY", "route_id": "4"}
    ]
    assert retriever.link_route_mentions(["take a train"]) == []
    assert retriever.link_route_mentions(["8:57am train from Poughkeepsie to Grand Central"]) == []


def test_explicit_subway_numeric_context_still_prefers_subway_route():
    retriever = _build_retriever()
    text = "No 5 trains between Bowling Green and Eastchester-Dyre Av."
    result = retriever.retrieve_affected_entities(text)

    assert result["status"] == "success"
    route_entities = [
        e for e in result["informed_entities"]
        if e.get("route_id") == "5" and not e.get("stop_id")
    ]
    assert route_entities
    assert route_entities[0]["agency_id"] == "MTASBWY"
