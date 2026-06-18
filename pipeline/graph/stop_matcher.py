from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .constants import RoutePattern


class StopMatcherMixin:
    def _route_neighborhood_stops(self, route_node_id: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for _, stop_node, edge_data in self.G.out_edges(route_node_id, data=True):
            if edge_data.get("type") != "Serves":
                continue
            attrs = self.G.nodes[stop_node]
            out.append((stop_node, str(attrs.get("name", "")).strip()))
        return out

    def _walk_corridor_between_stops(
        self,
        start_node: str,
        end_node: str,
        route_stop_set: Set[str],
        max_depth: int = 30,
    ) -> List[str]:
        """BFS along Next_Stop edges from start_node to end_node, constrained to route_stop_set."""
        for src, dst in ((start_node, end_node), (end_node, start_node)):
            visited: Set[str] = {src}
            queue: deque = deque([(src, [src])])
            while queue:
                current, path = queue.popleft()
                if len(path) > max_depth:
                    continue
                for _, neighbor, edge_data in self.G.out_edges(current, data=True):
                    if edge_data.get("type") != "Next_Stop":
                        continue
                    if neighbor not in route_stop_set:
                        continue
                    if neighbor in visited:
                        continue
                    new_path = path + [neighbor]
                    if neighbor == dst:
                        return new_path
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        return []

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

    def _pattern_corridor_path(
        self,
        pattern: RoutePattern,
        start_public_stop_id: str,
        end_public_stop_id: str,
    ) -> List[str]:
        if start_public_stop_id == end_public_stop_id:
            return []
        stop_ids = list(pattern.public_stop_ids)
        if start_public_stop_id not in stop_ids or end_public_stop_id not in stop_ids:
            return []
        start_idx = stop_ids.index(start_public_stop_id)
        end_idx = stop_ids.index(end_public_stop_id)
        if start_idx == end_idx:
            return []
        if start_idx < end_idx:
            return list(pattern.stop_nodes[start_idx : end_idx + 1])
        slice_nodes = list(pattern.stop_nodes[end_idx : start_idx + 1])
        slice_nodes.reverse()
        return slice_nodes

    def _score_hint_candidates(
        self,
        route_stops: Sequence[Tuple[str, str]],
        hint: str,
        alternative_segments: Sequence[str],
        min_score: float = 0.56,
        limit: int = 4,
    ) -> List[Tuple[str, str, float]]:
        constraint = self._parse_hint_constraint(hint)
        candidates: List[Tuple[str, str, float]] = []
        for stop_id, stop_name in route_stops:
            if constraint and not self._stop_matches_hint_constraint(stop_name, constraint):
                continue
            hint_score = self._best_segment_match_score(stop_name, [hint])
            alt_score = self._best_segment_match_score(stop_name, alternative_segments)
            if hint_score < min_score:
                continue
            if alt_score > 0 and (hint_score - alt_score) < 0.10:
                continue
            candidates.append((stop_id, stop_name, hint_score))
        candidates.sort(key=lambda row: row[2], reverse=True)
        return candidates[:limit]

    def _dedupe_matches(self, matches: Sequence[Tuple[str, str, float]], limit: int) -> List[Tuple[str, str, float]]:
        out: List[Tuple[str, str, float]] = []
        seen = set()
        for stop_id, stop_name, score in matches:
            public_stop_id = self.to_station_stop_id(stop_id)
            if public_stop_id in seen:
                continue
            seen.add(public_stop_id)
            out.append((stop_id, stop_name, score))
            if len(out) >= limit:
                break
        return out

    def _best_primary_corridor_match(
        self,
        route_stops: Sequence[Tuple[str, str]],
        primary_location_hints: Sequence[str],
        alternative_segments: Sequence[str],
        route_patterns: Sequence[RoutePattern],
    ) -> Dict[str, Any]:
        stop_name_map = {stop_id: stop_name for stop_id, stop_name in route_stops}
        route_stop_set = {stop_id for stop_id, _ in route_stops}
        endpoint_hints = [str(h).strip() for h in primary_location_hints if str(h).strip()][:2]
        if len(endpoint_hints) < 2:
            return {
                "matches": [],
                "primary_corridor_detected": False,
                "primary_corridor_expanded": False,
                "route_pattern_selected": None,
            }

        candidates_by_hint = [
            self._score_hint_candidates(route_stops, hint, alternative_segments)
            for hint in endpoint_hints
        ]
        if any(not rows for rows in candidates_by_hint):
            return {
                "matches": [],
                "primary_corridor_detected": False,
                "primary_corridor_expanded": False,
                "route_pattern_selected": None,
            }

        best_path: List[str] = []
        best_pattern_id: Optional[str] = None
        best_pair_score = -1.0
        best_endpoints: List[Tuple[str, str, float]] = []

        for first in candidates_by_hint[0]:
            first_public = self.to_station_stop_id(first[0])
            for second in candidates_by_hint[1]:
                second_public = self.to_station_stop_id(second[0])
                if first_public == second_public:
                    continue
                path: List[str] = []
                pattern_id: Optional[str] = None
                for pattern in route_patterns:
                    candidate_path = self._pattern_corridor_path(
                        pattern=pattern,
                        start_public_stop_id=first_public,
                        end_public_stop_id=second_public,
                    )
                    if len(candidate_path) > len(path):
                        path = candidate_path
                        pattern_id = pattern.pattern_id
                if not path:
                    path = self._walk_corridor_between_stops(
                        start_node=first[0],
                        end_node=second[0],
                        route_stop_set=route_stop_set,
                    )
                pair_score = first[2] + second[2]
                if len(path) > len(best_path) or (
                    len(path) == len(best_path) and pair_score > best_pair_score
                ):
                    best_path = path
                    best_pattern_id = pattern_id
                    best_pair_score = pair_score
                    best_endpoints = [first, second]

        if len(best_path) > 2:
            endpoint_scores = {
                self.to_station_stop_id(stop_id): score for stop_id, _stop_name, score in best_endpoints
            }
            expanded: List[Tuple[str, str, float]] = []
            for node in best_path:
                public_stop_id = self.to_station_stop_id(node)
                score = endpoint_scores.get(public_stop_id, 0.75)
                expanded.append((node, stop_name_map.get(node, node), score))
            return {
                "matches": self._dedupe_matches(expanded, limit=40),
                "primary_corridor_detected": True,
                "primary_corridor_expanded": True,
                "route_pattern_selected": best_pattern_id,
            }

        fallback_endpoints = self._dedupe_matches(
            [row for rows in candidates_by_hint for row in rows],
            limit=2,
        )
        return {
            "matches": fallback_endpoints,
            "primary_corridor_detected": bool(fallback_endpoints),
            "primary_corridor_expanded": False,
            "route_pattern_selected": best_pattern_id,
        }

    def _match_affected_stops(
        self,
        route_stops: Sequence[Tuple[str, str]],
        affected_segments: Sequence[str],
        alternative_segments: Sequence[str],
        location_hints: Optional[Sequence[str]] = None,
        allow_segment_scoring: bool = True,
        primary_location_hints: Optional[Sequence[str]] = None,
        route_patterns: Optional[Sequence[RoutePattern]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "matches": [],
            "primary_corridor_detected": False,
            "primary_corridor_expanded": False,
            "route_pattern_selected": None,
        }

        if primary_location_hints:
            corridor_result = self._best_primary_corridor_match(
                route_stops=route_stops,
                primary_location_hints=primary_location_hints,
                alternative_segments=alternative_segments,
                route_patterns=route_patterns or [],
            )
            if corridor_result.get("matches"):
                return corridor_result
            result.update(corridor_result)

        if location_hints:
            hint_constraints = [self._parse_hint_constraint(h) for h in location_hints if h]
            combined_constraint = self._merge_hint_constraints(hint_constraints)
            hinted: List[Tuple[str, str, float]] = []
            for stop_id, stop_name in route_stops:
                if hint_constraints and not any(
                    self._stop_matches_hint_constraint(stop_name, constraint)
                    for constraint in hint_constraints
                ):
                    continue
                hint_score = self._hint_match_score(stop_name, location_hints)
                alt_score = self._best_segment_match_score(stop_name, alternative_segments)
                if hint_score < 0.62:
                    continue
                if alt_score > 0 and (hint_score - alt_score) < 0.10:
                    continue
                hinted.append((stop_id, stop_name, hint_score))
            hinted.sort(key=lambda row: row[2], reverse=True)
            if hinted:
                if len(hint_constraints) == 2 and len(hinted) >= 2:
                    route_stop_set = {stop_id for stop_id, _ in route_stops}
                    first = hinted[0][0]
                    first_name = hinted[0][1]
                    second = None
                    for cand_id, cand_name, _ in hinted[1:]:
                        if cand_name != first_name:
                            second = cand_id
                            break
                    if second:
                        corridor = self._walk_corridor_between_stops(
                            start_node=first,
                            end_node=second,
                            route_stop_set=route_stop_set,
                        )
                        if len(corridor) > 2:
                            stop_name_map = {stop_id: stop_name for stop_id, stop_name in route_stops}
                            endpoint_scores = {stop_id: score for stop_id, _, score in hinted}
                            expanded: List[Tuple[str, str, float]] = []
                            for node in corridor:
                                name = stop_name_map.get(node, node)
                                score = endpoint_scores.get(node, 0.75)
                                expanded.append((node, name, score))
                            result["matches"] = self._dedupe_matches(expanded, limit=40)
                            return result
                result["matches"] = self._dedupe_matches(hinted, limit=6)
                return result

            if combined_constraint and combined_constraint.get("numbers") and combined_constraint.get("road_tokens"):
                return result

        matches: List[Tuple[str, str, float]] = []
        if allow_segment_scoring:
            for stop_id, stop_name in route_stops:
                affected_score = self._best_segment_match_score(stop_name, affected_segments)
                alt_score = self._best_segment_match_score(stop_name, alternative_segments)
                if affected_score < 0.60:
                    continue
                if alt_score > 0 and (affected_score - alt_score) < 0.12:
                    continue
                matches.append((stop_id, stop_name, affected_score))

        if not matches and location_hints:
            hint_constraints = [self._parse_hint_constraint(h) for h in location_hints if h]
            for stop_id, stop_name in route_stops:
                if hint_constraints and not any(
                    self._stop_matches_hint_constraint(stop_name, constraint)
                    for constraint in hint_constraints
                ):
                    continue
                hint_score = self._hint_match_score(stop_name, location_hints)
                if hint_score >= 0.52:
                    matches.append((stop_id, stop_name, hint_score))

        matches.sort(key=lambda row: row[2], reverse=True)
        result["matches"] = self._dedupe_matches(matches, limit=10)
        return result

    def _hint_match_score(self, stop_name: str, location_hints: Sequence[str]) -> float:
        if not location_hints:
            return 0.0

        per_hint_scores = [self._best_segment_match_score(stop_name, [hint]) for hint in location_hints]
        if not per_hint_scores:
            return 0.0
        ranked = sorted(per_hint_scores, reverse=True)
        best = ranked[0]
        if len(ranked) > 1:
            best += 0.25 * ranked[1]
        return min(1.0, best)
