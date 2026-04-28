from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple


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
        """BFS along Next_Stop edges from start_node to end_node, constrained to route_stop_set.

        Tries both directions (start→end, end→start) to handle bidirectional routes.
        Returns the ordered path including both endpoints, or [] if unreachable.
        """
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

    def _match_affected_stops(
        self,
        route_stops: Sequence[Tuple[str, str]],
        affected_segments: Sequence[str],
        alternative_segments: Sequence[str],
        location_hints: Optional[Sequence[str]] = None,
        allow_segment_scoring: bool = True,
    ) -> List[Tuple[str, str, float]]:
        if location_hints:
            hint_constraints = [self._parse_hint_constraint(h) for h in location_hints if h]
            combined_constraint = self._merge_hint_constraints(hint_constraints)
            hinted: List[Tuple[str, str, float]] = []
            for stop_id, stop_name in route_stops:
                # Use OR logic across individual constraints so each endpoint hint is
                # evaluated independently.  The merged combined_constraint uses a union
                # of all hints' numbers, which incorrectly rejects stops like
                # "Queensboro Plaza" when the other hint contributes a street number.
                if hint_constraints and not any(
                    self._stop_matches_hint_constraint(stop_name, c) for c in hint_constraints
                ):
                    continue
                hint_score = self._hint_match_score(stop_name, location_hints)
                alt_score = self._best_segment_match_score(stop_name, alternative_segments)
                if hint_score < 0.62:
                    continue
                if alt_score > 0 and (hint_score - alt_score) < 0.10:
                    continue
                hinted.append((stop_id, stop_name, hint_score))
            hinted.sort(key=lambda x: x[2], reverse=True)
            if hinted:
                # Corridor expansion: when exactly 2 distinct endpoints were matched (a
                # "between A and B" pattern), walk Next_Stop edges to include all
                # intermediate stops rather than returning only the named endpoints.
                if len(hint_constraints) == 2 and len(hinted) >= 2:
                    route_stop_set = {stop_id for stop_id, _ in route_stops}
                    stop_name_map = {stop_id: stop_name for stop_id, stop_name in route_stops}
                    endpoint_scores = {stop_id: score for stop_id, _, score in hinted}
                    # Find two stops from different physical stations: stops with N/S or
                    # other directional suffixes can share a root ID — pick the best
                    # representative from each group so the BFS walks a real corridor.
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
                            expanded: List[Tuple[str, str, float]] = []
                            for node in corridor:
                                name = stop_name_map.get(node, node)
                                score = endpoint_scores.get(node, 0.75)
                                expanded.append((node, name, score))
                            return expanded[:40]
                return hinted[:6]

            if combined_constraint and combined_constraint.get("numbers") and combined_constraint.get("road_tokens"):
                return []

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
            for stop_id, stop_name in route_stops:
                if hint_constraints and not any(
                    self._stop_matches_hint_constraint(stop_name, c) for c in hint_constraints
                ):
                    continue
                hint_score = self._hint_match_score(stop_name, location_hints)
                if hint_score >= 0.52:
                    matches.append((stop_id, stop_name, hint_score))

        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:10]

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
