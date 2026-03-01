from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


GMAPS_API_KEY_FILE = ".vscode/.gmaps_api"

# Public agency IDs observed in MTA alert feeds.
AGENCY_ID_BY_GTFS_NAMESPACE = {
    "gtfs_subway": "MTASBWY",
    "gtfs_busco": "MTABC",
    "gtfs_b": "MTA NYCT",
    "gtfs_bx": "MTA NYCT",
    "gtfs_m": "MTA NYCT",
    "gtfs_q": "MTA NYCT",
    "gtfs_si": "MTA NYCT",
}

ALT_LINE_MARKERS = (
    "instead",
    "alternative",
    "board or exit",
    "board/exit",
    "take the",
    "use the stop",
    "use stops",
    "customers may",
    "as requested",
)

AFFECTED_LINE_MARKERS = (
    "detour",
    "detoured",
    "bypass",
    "bypassing",
    "not make stop",
    "will be closed",
    "skip",
    "suspended",
    "delays",
    "running with delays",
)

_LOCATION_PATTERNS = [
    re.compile(r"\bat\s+([A-Za-z0-9\-\./&' ]+?)(?=\.|,|;|\n|$)", re.IGNORECASE),
    re.compile(
        r"\bbetween\s+([A-Za-z0-9\-\./&' ]+?)\s+and\s+([A-Za-z0-9\-\./&' ]+?)(?=\.|,|;|\n|$)",
        re.IGNORECASE,
    ),
    re.compile(r"\bfrom\s+([A-Za-z0-9\-\./&' ]+?)\s+to\s+([A-Za-z0-9\-\./&' ]+?)(?=\.|,|;|\n|$)", re.IGNORECASE),
]


@dataclass(frozen=True)
class RouteChoice:
    route_id: str
    route_node_id: str
    agency_id: str
    neighborhood_size: int
    stop_match_score: float


class GraphRetriever:
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
        print(f"Loading Knowledge Graph from {graph_path}...")
        try:
            with open(graph_path, "rb") as f:
                self.G: nx.DiGraph = pickle.load(f)
            print(f"Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error loading graph: {e}")
            self.G = nx.DiGraph()

        self.route_nodes_by_id: Dict[str, List[str]] = {}
        self.route_agency_by_node: Dict[str, str] = {}
        self._build_route_indexes()

        self._stop_coords: Optional[np.ndarray] = None
        self._stop_tree_nodes: List[str] = []
        self._build_stop_kdtree()

    # ---------------------------------------------------------------------
    # Public APIs
    # ---------------------------------------------------------------------
    def retrieve_affected_entities(
        self,
        alert_text: str,
        seed_entities: Optional[List[Dict[str, Any]]] = None,
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
            }

        affected_segments, alternative_segments = self._split_affected_and_alternative_segments(text)
        route_ids = self._extract_route_ids(
            text,
            seed_entities=seed_entities,
            affected_segments=affected_segments,
            alternative_segments=alternative_segments,
        )
        if not route_ids:
            return {
                "status": "error",
                "error": "No route IDs could be resolved",
                "informed_entities": self._seed_route_entities(seed_entities),
                "route_confidence": 0.0,
                "stop_confidence": 0.0,
                "location_hints": self._extract_location_hints(text),
                "fallback_needed": True,
                "route_ids": [],
            }

        route_choices: List[RouteChoice] = []
        location_hints = self._extract_location_hints(text)

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
            }

        informed_entities: List[Dict[str, Any]] = []
        matched_stop_entities: List[Dict[str, Any]] = []
        stop_candidates: List[Dict[str, Any]] = []
        stop_scores: List[float] = []

        for choice in route_choices:
            informed_entities.append({"agency_id": choice.agency_id, "route_id": choice.route_id})

            route_stops = self._route_neighborhood_stops(choice.route_node_id)
            route_stop_matches = self._match_affected_stops(
                route_stops=route_stops,
                affected_segments=affected_segments,
                alternative_segments=alternative_segments,
                location_hints=location_hints,
            )

            for stop_id, _stop_name, score in route_stop_matches:
                stop_candidates.append(
                    {
                        "agency_id": choice.agency_id,
                        "route_id": choice.route_id,
                        "stop_id": self.to_public_stop_id(stop_id),
                        "stop_name": _stop_name,
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

        informed_entities = self._dedupe_entities(informed_entities + matched_stop_entities)

        route_confidence = min(1.0, 0.7 + 0.3 * (len(route_choices) / max(1, len(route_ids))))

        has_stop_indication = self._has_stop_indication(text, location_hints)
        if not has_stop_indication:
            stop_confidence = 0.95
        elif stop_scores:
            stop_confidence = max(0.55, min(1.0, float(np.mean(stop_scores))))
        else:
            stop_confidence = 0.2

        fallback_needed = has_stop_indication and not stop_scores

        return {
            "status": "success",
            "informed_entities": informed_entities,
            "stop_candidates": stop_candidates,
            "route_confidence": round(route_confidence, 3),
            "stop_confidence": round(stop_confidence, 3),
            "location_hints": location_hints,
            "fallback_needed": fallback_needed,
            "route_ids": [c.route_id for c in route_choices],
            "route_node_ids": [c.route_node_id for c in route_choices],
            "matched_stop_count": len(stop_scores),
        }

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

            entity: Dict[str, Any] = {
                "agency_id": agency_id,
                "stop_id": self.to_public_stop_id(stop_node_id),
            }

            entities.append(entity)
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

    # ---------------------------------------------------------------------
    # Route extraction + matching
    # ---------------------------------------------------------------------
    def _build_route_indexes(self) -> None:
        self.route_nodes_by_id.clear()
        self.route_agency_by_node.clear()

        for node, attrs in self.G.nodes(data=True):
            if attrs.get("type") != "Route":
                continue

            route_id = self.to_public_route_id(node)
            self.route_nodes_by_id.setdefault(route_id, []).append(node)

            namespace = self._namespace_from_node(node)
            agency_id = AGENCY_ID_BY_GTFS_NAMESPACE.get(namespace, "MTA NYCT")
            self.route_agency_by_node[node] = agency_id

        for rid in self.route_nodes_by_id:
            self.route_nodes_by_id[rid].sort()

    @staticmethod
    def _namespace_from_node(node_id: str) -> str:
        if not node_id or "_" not in node_id:
            return ""
        return "_".join(node_id.split("_")[:-1])

    @staticmethod
    def to_public_route_id(route_node_or_id: str) -> str:
        value = str(route_node_or_id or "").strip().upper()
        return value.split("_")[-1] if "_" in value else value

    @staticmethod
    def to_public_stop_id(stop_node_or_id: str) -> str:
        value = str(stop_node_or_id or "").strip().upper()
        return value.split("_")[-1] if "_" in value else value

    @staticmethod
    def _normalize_route_token(token: str) -> str:
        t = token.strip().upper()
        t = t.strip("[](){}.,;:")
        t = t.replace("-SBS", "+")
        t = t.replace(" SBS", "+")
        if t.endswith("SBS") and not t.endswith("+SBS"):
            t = t[:-3] + "+"
        t = t.replace("-", "")
        return t

    def _extract_route_ids(
        self,
        text: str,
        seed_entities: Optional[List[Dict[str, Any]]] = None,
        affected_segments: Optional[Sequence[str]] = None,
        alternative_segments: Optional[Sequence[str]] = None,
    ) -> List[str]:
        known_route_ids = set(self.route_nodes_by_id.keys())
        found: List[str] = []

        # Token-based discovery from free text.
        token_candidates = self._route_tokens_from_text(text)
        alt_tokens = set()
        affected_tokens = set()

        for segment in alternative_segments or []:
            for raw in self._route_tokens_from_text(segment):
                token = self._normalize_route_token(raw)
                if token in known_route_ids:
                    alt_tokens.add(token)

        for segment in affected_segments or []:
            for raw in self._route_tokens_from_text(segment):
                token = self._normalize_route_token(raw)
                if token in known_route_ids:
                    affected_tokens.add(token)

        # Numeric subway routes should be paired with train context to avoid
        # false positives from times (e.g., "10:00 PM" -> route "1").
        numeric_matches = re.findall(r"\b([1-7])\s+trains?\b|\[([1-7])\]", text, flags=re.IGNORECASE)
        for pair in numeric_matches:
            raw_token = next((p for p in pair if p), "")
            token = self._normalize_route_token(raw_token)
            if token in known_route_ids:
                token_candidates.append(token)

        for raw in token_candidates:
            token = self._normalize_route_token(raw)
            if token in known_route_ids:
                if token in alt_tokens and token not in affected_tokens:
                    continue
                found.append(token)

        # Include seed routes when provided by legacy input.
        for entity in seed_entities or []:
            route_id = self._normalize_route_token(str(entity.get("route_id", "")))
            if route_id in known_route_ids:
                found.append(route_id)

        # Preserve order and dedupe.
        deduped: List[str] = []
        seen = set()
        for route_id in found:
            if route_id in seen:
                continue
            seen.add(route_id)
            deduped.append(route_id)

        return deduped

    @staticmethod
    def _route_tokens_from_text(text: str) -> List[str]:
        tokens: List[str] = []
        src = text or ""

        # Explicit bracket notation, e.g. [Q], [F], [2].
        tokens.extend(re.findall(r"\[([A-Za-z0-9\-\+]+)\]", src))
        # Bus/express IDs that include digits, e.g. B20, BxM2, M14D-SBS.
        tokens.extend(re.findall(r"\b([A-Za-z]{1,4}\d{1,3}[A-Za-z+\-]*)\b", src))
        # Letter subway routes only when attached to explicit plural train context.
        # This avoids false positives like "a train" -> route "A".
        tokens.extend(re.findall(r"\b([A-Za-z]{1,2})\s+trains\b", src, flags=re.IGNORECASE))
        return tokens

    def _seed_route_entities(self, seed_entities: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []
        for e in seed_entities or []:
            if not isinstance(e, dict):
                continue
            route_id = self._normalize_route_token(str(e.get("route_id", "")))
            if not route_id:
                continue
            agency_id = str(e.get("agency_id", "")).strip() or "MTA NYCT"
            entities.append({"agency_id": agency_id, "route_id": route_id})
        return self._dedupe_entities(entities)

    def _choose_route_node(
        self,
        route_id: str,
        affected_segments: Sequence[str],
        seed_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[RouteChoice]:
        candidates = self.route_nodes_by_id.get(route_id, [])
        if not candidates:
            return None

        seed_agencies = {
            str(e.get("agency_id", "")).strip()
            for e in (seed_entities or [])
            if isinstance(e, dict) and self._normalize_route_token(str(e.get("route_id", ""))) == route_id
        }

        best: Optional[RouteChoice] = None
        for route_node_id in candidates:
            agency_id = self.route_agency_by_node.get(route_node_id, "MTA NYCT")
            route_stops = self._route_neighborhood_stops(route_node_id)

            best_stop_score = 0.0
            for _stop_id, stop_name in route_stops:
                score = self._best_segment_match_score(stop_name, affected_segments)
                if score > best_stop_score:
                    best_stop_score = score

            # Seed agency preference for legacy-input compatibility.
            if seed_agencies and agency_id in seed_agencies:
                best_stop_score += 0.08

            choice = RouteChoice(
                route_id=route_id,
                route_node_id=route_node_id,
                agency_id=agency_id,
                neighborhood_size=len(route_stops),
                stop_match_score=min(1.0, best_stop_score),
            )

            if not best or choice.stop_match_score > best.stop_match_score:
                best = choice

        return best

    # ---------------------------------------------------------------------
    # Stop matching
    # ---------------------------------------------------------------------
    def _route_neighborhood_stops(self, route_node_id: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for _, stop_node, edge_data in self.G.out_edges(route_node_id, data=True):
            if edge_data.get("type") != "Serves":
                continue
            attrs = self.G.nodes[stop_node]
            out.append((stop_node, str(attrs.get("name", "")).strip()))
        return out

    @staticmethod
    def _norm_text_tokens(value: str) -> List[str]:
        txt = re.sub(r"[^a-z0-9 ]+", " ", (value or "").lower())
        txt = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", txt)
        txt = (
            txt.replace(" avenue ", " ave ")
            .replace(" street ", " st ")
            .replace(" road ", " rd ")
            .replace(" boulevard ", " blvd ")
            .replace(" parkway ", " pkwy ")
            .replace(" av ", " ave ")
            .replace(" and ", " ")
            .replace(" at ", " ")
        )
        return [tok for tok in txt.split() if tok]

    def _best_segment_match_score(self, phrase: str, segments: Sequence[str]) -> float:
        phrase_tokens = set(self._norm_text_tokens(phrase))
        if not phrase_tokens:
            return 0.0

        best = 0.0
        for segment in segments:
            segment_tokens = set(self._norm_text_tokens(segment))
            if not segment_tokens:
                continue
            overlap = len(phrase_tokens & segment_tokens)
            if overlap == 0:
                continue
            precision = overlap / len(phrase_tokens)
            recall = overlap / len(segment_tokens)
            score = (0.75 * precision) + (0.25 * recall)
            if score > best:
                best = score
        return min(1.0, best)

    def _match_affected_stops(
        self,
        route_stops: Sequence[Tuple[str, str]],
        affected_segments: Sequence[str],
        alternative_segments: Sequence[str],
        location_hints: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, str, float]]:
        if location_hints:
            hint_constraints = [self._parse_hint_constraint(h) for h in location_hints if h]
            combined_constraint = self._merge_hint_constraints(hint_constraints)
            hinted: List[Tuple[str, str, float]] = []
            for stop_id, stop_name in route_stops:
                # Enforce strict intersection-like matching when hint carries
                # strong constraints (street number + named road token).
                if combined_constraint:
                    if not self._stop_matches_hint_constraint(stop_name, combined_constraint):
                        continue
                elif hint_constraints and not any(
                    self._stop_matches_hint_constraint(stop_name, c) for c in hint_constraints
                ):
                    continue
                hint_score = self._best_segment_match_score(stop_name, location_hints)
                alt_score = self._best_segment_match_score(stop_name, alternative_segments)
                if hint_score < 0.62:
                    continue
                if alt_score > 0 and (hint_score - alt_score) < 0.10:
                    continue
                hinted.append((stop_id, stop_name, hint_score))
            hinted.sort(key=lambda x: x[2], reverse=True)
            if hinted:
                return hinted[:6]
            # Strong intersection-like hints should never degrade to loose
            # matching; returning empty is safer than wrong stop_ids.
            if combined_constraint and combined_constraint.get("numbers") and combined_constraint.get("road_tokens"):
                return []

        matches: List[Tuple[str, str, float]] = []
        for stop_id, stop_name in route_stops:
            affected_score = self._best_segment_match_score(stop_name, affected_segments)
            alt_score = self._best_segment_match_score(stop_name, alternative_segments)
            if affected_score < 0.60:
                continue
            if alt_score > 0 and (affected_score - alt_score) < 0.12:
                continue
            matches.append((stop_id, stop_name, affected_score))

        # If no matches from full segments, attempt targeted hint matching.
        if not matches and location_hints:
            for stop_id, stop_name in route_stops:
                if combined_constraint:
                    if not self._stop_matches_hint_constraint(stop_name, combined_constraint):
                        continue
                elif hint_constraints and not any(
                    self._stop_matches_hint_constraint(stop_name, c) for c in hint_constraints
                ):
                    continue
                hint_score = self._best_segment_match_score(stop_name, location_hints)
                if hint_score >= 0.52:
                    matches.append((stop_id, stop_name, hint_score))

        matches.sort(key=lambda x: x[2], reverse=True)
        # Keep top matches only to avoid noise.
        return matches[:10]

    @staticmethod
    def _parse_hint_constraint(hint: str) -> Dict[str, set]:
        tokens = GraphRetriever._norm_text_tokens(hint)
        numbers = {t for t in tokens if t.isdigit()}
        generic = {
            "st",
            "street",
            "ave",
            "av",
            "avenue",
            "rd",
            "road",
            "blvd",
            "pkwy",
            "e",
            "w",
            "n",
            "s",
            "east",
            "west",
            "north",
            "south",
        }
        road_tokens = {t for t in tokens if not t.isdigit() and t not in generic}
        return {"numbers": numbers, "road_tokens": road_tokens}

    @staticmethod
    def _stop_matches_hint_constraint(stop_name: str, constraint: Dict[str, set]) -> bool:
        stop_tokens = set(GraphRetriever._norm_text_tokens(stop_name))
        numbers = constraint.get("numbers", set())
        road_tokens = constraint.get("road_tokens", set())

        # If hint contains street numbers, require at least one number match.
        if numbers and not (numbers & stop_tokens):
            return False
        # If hint contains meaningful road token (e.g., madison), require it.
        if road_tokens and not (road_tokens & stop_tokens):
            return False
        return True

    @staticmethod
    def _merge_hint_constraints(constraints: Sequence[Dict[str, set]]) -> Optional[Dict[str, set]]:
        if not constraints:
            return None
        numbers = set()
        road_tokens = set()
        for c in constraints:
            numbers |= set(c.get("numbers", set()))
            road_tokens |= set(c.get("road_tokens", set()))
        if not numbers and not road_tokens:
            return None
        return {"numbers": numbers, "road_tokens": road_tokens}

    @staticmethod
    def _split_affected_and_alternative_segments(text: str) -> Tuple[List[str], List[str]]:
        parts = [p.strip() for p in re.split(r"[\n\r\.]+", text or "") if p.strip()]
        affected: List[str] = []
        alternative: List[str] = []

        for part in parts:
            lower = part.lower()
            if any(marker in lower for marker in ALT_LINE_MARKERS):
                alternative.append(part)
                continue
            if any(marker in lower for marker in AFFECTED_LINE_MARKERS):
                affected.append(part)
                continue
            # Neutral lines are useful context for affected scope.
            affected.append(part)

        return affected, alternative

    @staticmethod
    def _extract_location_hints(text: str) -> List[str]:
        hints: List[str] = []
        for pattern in _LOCATION_PATTERNS:
            for match in pattern.findall(text or ""):
                if isinstance(match, tuple):
                    pieces = [m.strip() for m in match if m and m.strip()]
                    hints.extend(pieces)
                else:
                    candidate = match.strip()
                    if candidate:
                        hints.append(candidate)

        # Preserve order and dedupe.
        deduped: List[str] = []
        seen = set()
        for hint in hints:
            key = hint.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hint)
        return deduped

    @staticmethod
    def _has_stop_indication(text: str, location_hints: Sequence[str]) -> bool:
        if location_hints:
            return True
        lower = (text or "").lower()
        stop_markers = ("at ", "between ", "from ", "stop on", "station", "st ", "ave")
        return any(marker in lower for marker in stop_markers)

    # ---------------------------------------------------------------------
    # Google Maps fallback
    # ---------------------------------------------------------------------
    def _build_stop_kdtree(self) -> None:
        coords: List[List[float]] = []
        node_ids: List[str] = []

        for node, attrs in self.G.nodes(data=True):
            if attrs.get("type") != "Stop":
                continue

            lat = attrs.get("lat", attrs.get("stop_lat"))
            lon = attrs.get("lon", attrs.get("stop_lon"))
            if lat in (None, "") or lon in (None, ""):
                continue

            try:
                coords.append([float(lat), float(lon)])
                node_ids.append(node)
            except (TypeError, ValueError):
                continue

        if not coords:
            self._stop_coords = None
            self._stop_tree_nodes = []
            return

        self._stop_coords = np.array(coords, dtype=float)
        self._stop_tree_nodes = node_ids

    def _load_gmaps_api_key(self) -> Optional[str]:
        candidate_paths = []
        configured = os.path.expanduser(GMAPS_API_KEY_FILE)
        if os.path.isabs(configured):
            candidate_paths.append(configured)
        else:
            candidate_paths.append(os.path.join(os.getcwd(), configured))
            candidate_paths.append(os.path.join(os.path.expanduser("~"), configured))
        candidate_paths.append(os.path.join(os.getcwd(), ".vscode", ".gmaps_api"))
        candidate_paths.append(os.path.join(os.path.expanduser("~"), ".vscode", ".gmaps_api"))

        seen = set()
        for path in candidate_paths:
            resolved = os.path.realpath(path)
            if resolved in seen:
                continue
            seen.add(resolved)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    key = f.read().strip()
                if key:
                    return key
        return os.environ.get("GOOGLE_MAPS_API_KEY")

    def _geocode_hint_to_stop(
        self,
        location_hint: str,
        preferred_route_node_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Tuple[str, float]]:
        if self._stop_coords is None or not self._stop_tree_nodes:
            return None

        try:
            import googlemaps  # noqa: PLC0415
        except ImportError:
            return None

        api_key = self._load_gmaps_api_key()
        if not api_key:
            return None

        try:
            gmaps = googlemaps.Client(key=api_key)
            results = gmaps.geocode(f"{location_hint}, New York City")
        except Exception:
            return None

        if not results:
            return None

        location = results[0].get("geometry", {}).get("location", {})
        lat = location.get("lat")
        lon = location.get("lng")
        if lat is None or lon is None:
            return None

        distance_deg, idx = self._nearest_stop_index(lat, lon)
        nearest_node = self._stop_tree_nodes[idx]

        if preferred_route_node_ids:
            route_stop_ids = self._route_stop_id_set(preferred_route_node_ids)
            if nearest_node not in route_stop_ids:
                constrained = self._nearest_stop_within_set(lat, lon, route_stop_ids)
                if constrained:
                    nearest_node, distance_deg = constrained

        # Rough conversion from degree to meters.
        distance_m = float(distance_deg) * 111_000.0
        return nearest_node, distance_m

    def _nearest_stop_index(self, lat: float, lon: float) -> Tuple[float, int]:
        assert self._stop_coords is not None
        deltas = self._stop_coords - np.array([lat, lon], dtype=float)
        dists = np.sqrt((deltas ** 2).sum(axis=1))
        idx = int(np.argmin(dists))
        return float(dists[idx]), idx

    def _route_stop_id_set(self, route_node_ids: Sequence[str]) -> set:
        out = set()
        for route_node in route_node_ids:
            for _, stop_node, edge_data in self.G.out_edges(route_node, data=True):
                if edge_data.get("type") == "Serves":
                    out.add(stop_node)
        return out

    def _nearest_stop_within_set(
        self,
        lat: float,
        lon: float,
        allowed_stop_nodes: Iterable[str],
    ) -> Optional[Tuple[str, float]]:
        allowed = set(allowed_stop_nodes)
        if not allowed:
            return None

        best_node = None
        best_distance = None
        for node in allowed:
            attrs = self.G.nodes.get(node, {})
            node_lat = attrs.get("lat", attrs.get("stop_lat"))
            node_lon = attrs.get("lon", attrs.get("stop_lon"))
            if node_lat in (None, "") or node_lon in (None, ""):
                continue
            try:
                d = np.linalg.norm([float(node_lat) - lat, float(node_lon) - lon])
            except (TypeError, ValueError):
                continue

            if best_distance is None or d < best_distance:
                best_distance = d
                best_node = node

        if best_node is None or best_distance is None:
            return None
        return best_node, float(best_distance)

    @staticmethod
    def _distance_confidence(distance_m: float) -> float:
        if distance_m <= 150:
            return 0.95
        if distance_m <= 300:
            return 0.85
        if distance_m <= 600:
            return 0.7
        return 0.5

    def _agency_id_from_stop_node(self, stop_node_id: str) -> str:
        namespace = self._namespace_from_node(stop_node_id)
        return AGENCY_ID_BY_GTFS_NAMESPACE.get(namespace, "MTA NYCT")

    # ---------------------------------------------------------------------
    # Utils
    # ---------------------------------------------------------------------
    @staticmethod
    def _dedupe_entities(entities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            agency_id = str(entity.get("agency_id", "")).strip()
            route_id = str(entity.get("route_id", "")).strip().upper() if entity.get("route_id") else ""
            stop_id = str(entity.get("stop_id", "")).strip().upper() if entity.get("stop_id") else ""
            if not agency_id:
                continue
            if not route_id and not stop_id:
                continue
            key = (agency_id, route_id, stop_id)
            if key in seen:
                continue
            seen.add(key)
            out = {"agency_id": agency_id}
            if route_id:
                out["route_id"] = route_id
            if stop_id:
                out["stop_id"] = stop_id
            deduped.append(out)
        return deduped


def run_offline_smoke_test() -> None:
    """Simple deterministic smoke tests."""
    retriever = GraphRetriever(graph_path="data/mta_knowledge_graph.gpickle")

    alert = (
        "Northbound B20 buses are detoured due to utility work on Decatur St at Wilson Ave. "
        "Buses will not make stops on Decatur St from Bushwick Ave to Irving Ave."
    )
    result = retriever.retrieve_affected_entities(alert)
    assert result["status"] == "success"
    assert any(e.get("route_id") == "B20" for e in result["informed_entities"])

    q_alert = "[Q] trains are bypassing 86 St. Take the B instead."
    q_result = retriever.retrieve_affected_entities(q_alert)
    assert q_result["status"] == "success"
    assert any(e.get("route_id") == "Q" for e in q_result["informed_entities"])

    print("Offline graph retriever smoke tests passed.")


if __name__ == "__main__":
    run_offline_smoke_test()
