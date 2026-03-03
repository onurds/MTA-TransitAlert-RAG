from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .constants import RouteChoice


class RouteResolverMixin:
    route_nodes_by_id: Dict[str, List[str]]
    route_agency_by_node: Dict[str, str]

    def _extract_route_ids(
        self,
        text: str,
        seed_entities: Optional[List[Dict[str, Any]]] = None,
        affected_segments: Optional[Sequence[str]] = None,
        alternative_segments: Optional[Sequence[str]] = None,
    ) -> List[str]:
        known_route_ids = set(self.route_nodes_by_id.keys())
        found: List[str] = []

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

        for entity in seed_entities or []:
            route_id = self._normalize_route_token(str(entity.get("route_id", "")))
            if route_id in known_route_ids:
                found.append(route_id)

        deduped: List[str] = []
        seen = set()
        for route_id in found:
            if route_id in seen:
                continue
            seen.add(route_id)
            deduped.append(route_id)

        return deduped

    @staticmethod
    def _dedupe_route_ids(route_ids: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for rid in route_ids:
            token = str(rid or "").strip().upper()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    def _routes_from_seed(self, seed_entities: Sequence[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for e in seed_entities or []:
            if not isinstance(e, dict):
                continue
            rid = self.normalize_route_id(str(e.get("route_id", "")))
            if rid in self.route_nodes_by_id:
                out.append(rid)
        return self._dedupe_route_ids(out)

    def _route_tokens_from_text(self, text: str) -> List[str]:
        tokens: List[str] = []
        src = text or ""
        tokens.extend(re.findall(r"\[([A-Za-z0-9\-\+]+)\]", src))
        tokens.extend(re.findall(r"\b([A-Za-z]{1,4}\d{1,3}[A-Za-z+\-]*)\b", src))
        tokens.extend(re.findall(r"\b([A-Za-z]{1,2})\s+trains\b", src, flags=re.IGNORECASE))
        tokens.extend(self.route_alias_matches(src))
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
