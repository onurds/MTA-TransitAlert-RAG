from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .constants import AGENCY_ID_BY_GTFS_NAMESPACE, RouteChoice
from .geocode_fallback import GeocodeFallbackMixin
from .indexes import GraphIndexMixin
from .location_hints import LocationHintMixin
from .route_resolver import RouteResolverMixin
from .stop_matcher import StopMatcherMixin


class GraphRetriever(
    GraphIndexMixin,
    LocationHintMixin,
    RouteResolverMixin,
    StopMatcherMixin,
    GeocodeFallbackMixin,
):
    """
    Deterministic GTFS graph grounding service.

    Responsibilities:
    - Resolve route IDs from text and/or seed entities.
    - Restrict stop matching to route neighborhoods (Route -> Stop Serves edges).
    - Exclude likely alternative/detour recommendation stops.
    - Provide confidence signals for downstream fallback decisions.
    """

    def __init__(self, graph_path: str = "data/mta_knowledge_graph.gpickle"):
        self.graph_path = graph_path
        self.route_nodes_by_id: Dict[str, List[str]] = {}
        self.route_agency_by_node: Dict[str, str] = {}
        self.route_long_name_by_id: Dict[str, str] = {}
        self.route_short_name_by_id: Dict[str, str] = {}
        self.route_phrase_to_id: Dict[str, str] = {}
        self.route_code_aliases: Dict[str, str] = {}
        self.route_display_name_by_id: Dict[str, str] = {}
        self.stop_nodes_by_id: Dict[str, List[str]] = {}
        self.stop_agency_by_id: Dict[str, str] = {}
        self._stop_coords = None
        self._stop_tree_nodes = []

        self._load_graph()
        self._build_route_indexes()
        self._build_stop_indexes()
        self._build_stop_kdtree()

    def retrieve_affected_entities(
        self,
        alert_text: str,
        seed_entities: Optional[List[Dict[str, Any]]] = None,
        route_ids_override: Optional[Sequence[str]] = None,
        location_hints_override: Optional[Sequence[str]] = None,
        alternative_hints_override: Optional[str] = None,
        max_stop_candidates: int = 20,
        effect_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        text = (alert_text or "").strip()
        if not text:
            return {
                "status": "error",
                "error": "Empty alert text",
                "informed_entities": [],
                "route_confidence": 0.0,
                "stop_confidence": 0.0,
                "location_hints": [],
                "fallback_needed": True,
                "route_ids": [],
                "has_stop_intent": False,
                "corridor_detected": False,
                "high_level_context": self.retrieve_high_level_context(
                    alert_text="",
                    route_ids=[],
                    location_hints=[],
                    stop_candidates=[],
                ),
            }

        affected_segments, alternative_segments = self._split_affected_and_alternative_segments(text)
        # If the LLM identified alternative service text, use it to override the
        # deterministic alternative_segments — the LLM has better recall on
        # non-standard phrasings (e.g., "D/F/N/R trains serve the same stops").
        if alternative_hints_override:
            alt_override_segments = [s.strip() for s in alternative_hints_override.split(".") if s.strip()]
            if alt_override_segments:
                alternative_segments = alt_override_segments
        has_strong_stop_intent = self._has_strong_stop_intent(text)
        if route_ids_override:
            route_ids = self.validate_route_ids(route_ids_override)
            if seed_entities:
                route_ids = self._dedupe_route_ids(route_ids + self._routes_from_seed(seed_entities))
        else:
            route_ids = self._extract_route_ids(
                text,
                seed_entities=seed_entities,
                affected_segments=affected_segments,
                alternative_segments=alternative_segments,
            )

        if not route_ids:
            location_hints = self._merge_location_hints(
                [],
                location_hints_override,
            ) if location_hints_override else self._merge_location_hints(
                self._extract_location_hints(text),
                None,
            )
            return {
                "status": "error",
                "error": "No route IDs could be resolved",
                "informed_entities": self._seed_route_entities(seed_entities),
                "route_confidence": 0.0,
                "stop_confidence": 0.0,
                "location_hints": location_hints,
                "fallback_needed": True,
                "route_ids": [],
                "has_stop_intent": has_strong_stop_intent or bool(location_hints),
                "corridor_detected": False,
                "high_level_context": self.retrieve_high_level_context(
                    alert_text=text,
                    route_ids=[],
                    location_hints=location_hints,
                    stop_candidates=[],
                ),
            }

        route_choices: List[RouteChoice] = []
        location_hints = self._merge_location_hints(
            [],
            location_hints_override,
        ) if location_hints_override else self._merge_location_hints(
            self._extract_location_hints(text),
            None,
        )

        for route_id in route_ids:
            route_choice = self._choose_route_node(
                route_id=route_id,
                affected_segments=affected_segments,
                seed_entities=seed_entities,
            )
            if route_choice:
                route_choices.append(route_choice)

        if not route_choices:
            return {
                "status": "error",
                "error": "Resolved route IDs were not found in graph",
                "informed_entities": self._seed_route_entities(seed_entities),
                "route_confidence": 0.0,
                "stop_confidence": 0.0,
                "location_hints": location_hints,
                "fallback_needed": True,
                "route_ids": route_ids,
                "has_stop_intent": has_strong_stop_intent or bool(location_hints),
                "corridor_detected": False,
                "high_level_context": self.retrieve_high_level_context(
                    alert_text=text,
                    route_ids=route_ids,
                    location_hints=location_hints,
                    stop_candidates=[],
                ),
            }

        informed_entities: List[Dict[str, Any]] = []
        matched_stop_entities: List[Dict[str, Any]] = []
        stop_candidates: List[Dict[str, Any]] = []
        stop_scores: List[float] = []
        corridor_detected = False

        for choice in route_choices:
            informed_entities.append({"agency_id": choice.agency_id, "route_id": choice.route_id})

            route_stops = self._route_neighborhood_stops(choice.route_node_id)
            route_stop_matches = self._match_affected_stops(
                route_stops=route_stops,
                affected_segments=affected_segments,
                alternative_segments=alternative_segments,
                location_hints=location_hints,
                allow_segment_scoring=has_strong_stop_intent,
            )

            # Detect corridor expansion: 2 endpoint hints produced >2 matched stops,
            # meaning the BFS walk found intermediate corridor stops.
            if len(location_hints or []) == 2 and len(route_stop_matches) > 2:
                corridor_detected = True

            for stop_id, stop_name, score in route_stop_matches:
                stop_candidates.append(
                    {
                        "agency_id": choice.agency_id,
                        "route_id": choice.route_id,
                        "stop_id": self.to_public_stop_id(stop_id),
                        "stop_name": stop_name,
                        "score": round(float(score), 3),
                    }
                )
                matched_stop_entities.append(
                    {
                        "agency_id": choice.agency_id,
                        "route_id": choice.route_id,
                        "stop_id": self.to_public_stop_id(stop_id),
                    }
                )
                stop_scores.append(score)

        stop_candidates.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
        # Allow a wider candidate window when corridor expansion fired so all
        # corridor stops survive the cap and reach the entity selector.
        effective_max = 40 if corridor_detected else max(1, int(max_stop_candidates))
        stop_candidates = stop_candidates[:effective_max]

        # ── Fix A: namespace isolation ────────────────────────────────────────
        # When every selected route is a non-bus service (subway / LIRR / MNR),
        # strip out any bus-namespace stop candidates.  Bus stops enter the
        # candidate pool because subway suspension alerts describe shuttle buses
        # in their body text ("Free shuttle buses make stops at…"), but the
        # affected entities are exclusively subway/commuter-rail stops.
        # Bus agencies: both NYC Transit local buses and MTA Bus Company express buses.
        _BUS_AGENCIES = {"MTA NYCT", "MTABC"}
        _NON_BUS_AGENCIES = {"MTASBWY", "LI", "MNR"}
        if stop_candidates and route_choices:
            route_agencies = {c.agency_id for c in route_choices}
            if route_agencies and route_agencies.issubset(_NON_BUS_AGENCIES):
                # ── Fix A: namespace isolation ────────────────────────────────
                # All routes are subway / LIRR / MNR.  Strip bus-namespace stop
                # candidates that leak in via shuttle bus sentences in descriptions.
                stop_candidates = [
                    c for c in stop_candidates if c.get("agency_id") not in _BUS_AGENCIES
                ]

        # ── Fix B: DETOUR suppression for bus routes ──────────────────────────
        # Gold annotations for bus detour alerts follow MTA GTFS-RT convention:
        # only the route entity is emitted, never individual stop entities.
        # Generating stops for these alerts produces systematic false positives.
        if stop_candidates and route_choices and effect_hint:
            route_agencies = {c.agency_id for c in route_choices}
            if route_agencies and route_agencies.issubset(_BUS_AGENCIES):
                if effect_hint.strip().upper() in {"DETOUR", "MODIFIED_SERVICE"}:
                    stop_candidates = []

        selected_stop_ids = {str(c.get("stop_id", "")).upper() for c in stop_candidates if c.get("stop_id")}
        matched_stop_entities = [
            e
            for e in matched_stop_entities
            if str(e.get("stop_id", "")).upper() in selected_stop_ids
        ]
        informed_entities = self._dedupe_entities(informed_entities + matched_stop_entities)

        route_confidence = min(1.0, 0.7 + 0.3 * (len(route_choices) / max(1, len(route_ids))))
        has_stop_indication = bool(location_hints) or has_strong_stop_intent
        if not has_stop_indication:
            stop_confidence = 0.95
        elif stop_scores:
            stop_confidence = max(0.55, min(1.0, float(np.mean(stop_scores))))
        else:
            stop_confidence = 0.2

        fallback_needed = has_stop_indication and not stop_scores
        route_ids_out = [c.route_id for c in route_choices]

        return {
            "status": "success",
            "informed_entities": informed_entities,
            "stop_candidates": stop_candidates,
            "route_confidence": round(route_confidence, 3),
            "stop_confidence": round(stop_confidence, 3),
            "location_hints": location_hints,
            "fallback_needed": fallback_needed,
            "route_ids": route_ids_out,
            "route_node_ids": [c.route_node_id for c in route_choices],
            "matched_stop_count": len(stop_scores),
            "has_stop_intent": has_stop_indication,
            "corridor_detected": corridor_detected,
            "high_level_context": self.retrieve_high_level_context(
                alert_text=text,
                route_ids=route_ids_out,
                location_hints=location_hints,
                stop_candidates=stop_candidates,
                corridor_detected=corridor_detected,
            ),
        }

    def retrieve_high_level_context(
        self,
        alert_text: str,
        route_ids: Sequence[str],
        location_hints: Sequence[str],
        stop_candidates: Sequence[Dict[str, Any]],
        corridor_detected: bool = False,
    ) -> Dict[str, Any]:
        normalized_routes = self.validate_route_ids(route_ids)
        corridor_stop_names: List[str] = []
        seen_stop_names = set()
        for row in stop_candidates[:5]:
            stop_name = str(row.get("stop_name", "") or "").strip()
            if not stop_name:
                continue
            key = stop_name.lower()
            if key in seen_stop_names:
                continue
            seen_stop_names.add(key)
            corridor_stop_names.append(stop_name)

        co_occurring_routes = [
            rid
            for rid in self._extract_route_ids(alert_text or "")
            if rid not in normalized_routes
        ][:5]

        alert_patterns: List[str] = []
        lower = (alert_text or "").lower()
        for marker, label in (
            ("detour", "detour"),
            ("reroute", "reroute"),
            ("bypass", "stops_skipped"),
            ("delay", "delay"),
            ("suspend", "suspension"),
            ("relocat", "relocation"),
        ):
            if marker in lower and label not in alert_patterns:
                alert_patterns.append(label)

        return {
            "route_display_names": [
                self.route_display_name_for_id(route_id) or route_id
                for route_id in normalized_routes
            ],
            "route_families": self._route_families(normalized_routes),
            "corridor_stop_names": corridor_stop_names,
            "location_hints": [str(h).strip() for h in location_hints if str(h).strip()][:5],
            "co_occurring_routes": co_occurring_routes,
            "agency_context": sorted(
                {
                    self.agency_for_route_id(route_id)
                    for route_id in normalized_routes
                }
            ),
            "alert_patterns": alert_patterns,
            "corridor_detected": corridor_detected,
        }

    @staticmethod
    def _route_families(route_ids: Sequence[str]) -> List[str]:
        families: List[str] = []
        seen = set()
        for route_id in route_ids:
            token = str(route_id or "").strip().upper()
            if not token:
                continue
            match = re.match(r"[A-Z]+", token)
            family = match.group(0) if match else "NUMERIC"
            if family in seen:
                continue
            seen.add(family)
            families.append(family)
        return families

    def geocode_fallback_entities(
        self,
        location_hints: Sequence[str],
        route_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        hints = [h for h in location_hints if h]
        if not hints:
            return {
                "status": "skipped",
                "reason": "No location hints provided",
                "entities": [],
                "confidence": 0.0,
            }

        route_ids_upper = [r.upper() for r in (route_ids or []) if r]
        route_node_ids: List[str] = []
        for rid in route_ids_upper:
            route_node_ids.extend(self.route_nodes_by_id.get(rid, []))

        entities: List[Dict[str, Any]] = []
        confidences: List[float] = []
        for hint in hints:
            result = self._geocode_hint_to_stop(hint, route_node_ids)
            if not result:
                continue
            stop_node_id, dist_m = result
            agency_id = self._agency_id_from_stop_node(stop_node_id)
            entities.append({"agency_id": agency_id, "stop_id": self.to_public_stop_id(stop_node_id)})
            confidences.append(self._distance_confidence(dist_m))

        if not entities:
            return {
                "status": "error",
                "reason": "No geocoded stops",
                "entities": [],
                "confidence": 0.0,
            }

        return {
            "status": "success",
            "entities": self._dedupe_entities(entities),
            "confidence": round(float(np.mean(confidences)), 3) if confidences else 0.0,
        }

    def _agency_id_from_stop_node(self, stop_node_id: str) -> str:
        namespace = self._namespace_from_node(stop_node_id)
        return AGENCY_ID_BY_GTFS_NAMESPACE.get(namespace, "MTA NYCT")
