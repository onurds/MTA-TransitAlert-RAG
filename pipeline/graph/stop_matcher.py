from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


class StopMatcherMixin:
    def _route_neighborhood_stops(self, route_node_id: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for _, stop_node, edge_data in self.G.out_edges(route_node_id, data=True):
            if edge_data.get("type") != "Serves":
                continue
            attrs = self.G.nodes[stop_node]
            out.append((stop_node, str(attrs.get("name", "")).strip()))
        return out

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
                if combined_constraint:
                    if not self._stop_matches_hint_constraint(stop_name, combined_constraint):
                        continue
                elif hint_constraints and not any(
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
                return hinted[:6]

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

        if not matches and location_hints:
            for stop_id, stop_name in route_stops:
                if combined_constraint:
                    if not self._stop_matches_hint_constraint(stop_name, combined_constraint):
                        continue
                elif hint_constraints and not any(
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
