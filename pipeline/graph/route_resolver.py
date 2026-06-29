from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .constants import RouteChoice


class RouteResolverMixin:
    route_nodes_by_id: Dict[str, List[str]]
    route_agency_by_node: Dict[str, str]

    COMMUTER_AGENCIES = {"LI", "MNR"}
    CITY_TERMINAL_STOP_NAMES = {
        "atlantic terminal",
        "east new york",
        "forest hills",
        "grand central",
        "jamaica",
        "kew gardens",
        "nostrand avenue",
        "penn station",
        "woodside",
    }

    def _extract_route_ids(
        self,
        text: str,
        seed_entities: Optional[List[Dict[str, Any]]] = None,
        affected_segments: Optional[Sequence[str]] = None,
        alternative_segments: Optional[Sequence[str]] = None,
    ) -> List[str]:
        known_route_ids = set(self.route_nodes_by_id.keys())
        found: List[str] = []

        alias_route_ids = self.route_alias_matches(text)
        token_candidates = self._route_tokens_from_text(text)
        inferred_commuter_routes = self._infer_commuter_route_ids_from_station_context(
            text=text,
            affected_segments=affected_segments,
        )
        commuter_context = self._has_commuter_rail_context(text) or bool(inferred_commuter_routes)
        explicit_subway_numeric = self._has_explicit_subway_numeric_route_context(text)
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

        numeric_matches = re.findall(
            r"\b([1-7])\s+trains?\b|\[([1-7])\]", text, flags=re.IGNORECASE
        )
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
                if (
                    commuter_context
                    and token.isdigit()
                    and token in {"1", "2", "3", "4", "5", "6", "7"}
                    and token not in inferred_commuter_routes
                    and token not in alias_route_ids
                    and not explicit_subway_numeric
                ):
                    continue
                found.append(token)

        found = list(inferred_commuter_routes) + found

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

    def _has_commuter_rail_context(self, text: str) -> bool:
        lower = f" {(text or '').lower()} "
        context_markers = (
            " lirr ",
            " long island rail road ",
            " long island railroad ",
            " metro-north ",
            " metro north ",
            " mnr ",
            " penn station ",
            " grand central ",
            " jamaica ",
            " babylon ",
            " montauk ",
            " ronkonkoma ",
            " greenport ",
            " forest hills ",
            " kew gardens ",
            " woodside ",
            " eastbound trains ",
            " westbound trains ",
        )
        return any(marker in lower for marker in context_markers)

    @staticmethod
    def _has_explicit_subway_numeric_route_context(text: str) -> bool:
        return bool(
            re.search(
                r"(?:\[[1-7]\])|(?:\b[1-7]\s+(?:train|trains|line|subway|express|local)\b)",
                text or "",
                flags=re.IGNORECASE,
            )
        )

    def _infer_commuter_route_ids_from_station_context(
        self,
        text: str,
        affected_segments: Optional[Sequence[str]] = None,
    ) -> List[str]:
        source = " ".join(
            part.strip()
            for part in [text or "", *(affected_segments or [])]
            if str(part or "").strip()
        )
        norm_source = self._normalize_commuter_match_text(source)
        if not norm_source:
            return []

        matched_city_terminal_names = self._matched_commuter_stop_names_for_route(
            route_id="12",
            route_node_id="gtfslirr_12",
            norm_source=norm_source,
        )
        if (
            len(matched_city_terminal_names) >= 2
            and matched_city_terminal_names.issubset(self.CITY_TERMINAL_STOP_NAMES)
        ):
            # Alerts mentioning only terminal-zone LIRR stations often otherwise
            # tie with through-running branches that also serve those stations.
            return ["12"]

        scored: List[Tuple[int, int, int, str]] = []
        for route_id, route_nodes in self.route_nodes_by_id.items():
            for route_node_id in route_nodes:
                agency_id = self.route_agency_by_node.get(route_node_id, "")
                if agency_id not in self.COMMUTER_AGENCIES:
                    continue
                matched_names = self._matched_commuter_stop_names_for_route(
                    route_id=route_id,
                    route_node_id=route_node_id,
                    norm_source=norm_source,
                )
                if len(matched_names) < 2:
                    continue
                long_name = str(self.G.nodes[route_node_id].get("long_name", "") or "")
                long_name_hit = int(
                    self._normalized_phrase_in_text(
                        self._normalize_commuter_match_text(long_name),
                        norm_source,
                    )
                )
                terminal_only = matched_names.issubset(self.CITY_TERMINAL_STOP_NAMES)
                if terminal_only and route_id != "12":
                    continue
                # Prefer more matched stations, then an explicit branch/line name,
                # then the smaller route neighborhood to break through-route ties.
                route_stop_count = len(self._route_neighborhood_stops(route_node_id))
                scored.append((len(matched_names), long_name_hit, -route_stop_count, route_id))

        scored.sort(reverse=True)
        out: List[str] = []
        seen = set()
        for _matches, _name_hit, _size_score, route_id in scored:
            if route_id in seen:
                continue
            seen.add(route_id)
            out.append(route_id)
            if len(out) >= 3:
                break
        return out

    def _matched_commuter_stop_names_for_route(
        self,
        route_id: str,
        route_node_id: str,
        norm_source: str,
    ) -> Set[str]:
        if route_node_id not in self.route_nodes_by_id.get(route_id, []):
            return set()
        matched: Set[str] = set()
        for _stop_node, stop_name in self._route_neighborhood_stops(route_node_id):
            norm_stop = self._normalize_commuter_match_text(stop_name)
            if not norm_stop:
                continue
            if self._normalized_phrase_in_text(norm_stop, norm_source):
                matched.add(norm_stop)
        return matched

    @classmethod
    def _normalize_commuter_match_text(cls, text: str) -> str:
        norm = cls._normalize_route_phrase(text)
        if not norm:
            return ""
        replacements = (
            ("new york penn station", "penn station"),
            ("ny penn station", "penn station"),
            ("nyc penn station", "penn station"),
            ("grand central madison", "grand central"),
            ("hunterspoint av", "hunterspoint avenue"),
        )
        padded = f" {norm} "
        for src, dst in replacements:
            padded = padded.replace(f" {src} ", f" {dst} ")
        return re.sub(r"\s{2,}", " ", padded).strip()

    @staticmethod
    def _normalized_phrase_in_text(phrase: str, norm_text: str) -> bool:
        needle = re.escape((phrase or "").strip())
        if not needle:
            return False
        return bool(re.search(rf"(?<![a-z0-9]){needle}(?![a-z0-9])", norm_text or ""))

    def _extract_primary_affected_route_ids(
        self,
        text: str,
        seed_entities: Optional[List[Dict[str, Any]]] = None,
        affected_segments: Optional[Sequence[str]] = None,
    ) -> List[str]:
        candidate_segments: List[str] = []
        if text:
            first_sentence = re.split(r"[\n\r\.;]+", text, maxsplit=1)[0].strip()
            if first_sentence:
                candidate_segments.append(first_sentence)
        candidate_segments.extend(
            str(segment).strip()
            for segment in (affected_segments or [])
            if str(segment).strip()
        )

        seen = set()
        for segment in candidate_segments[:4]:
            lower = segment.lower()
            if not any(
                marker in lower
                for marker in (" no ", "suspend", "bypass", "skip", "detour", "between", " at ")
            ):
                continue
            route_ids = self._extract_route_ids(
                segment,
                seed_entities=seed_entities,
                affected_segments=[segment],
                alternative_segments=[],
            )
            if route_ids:
                out: List[str] = []
                for route_id in route_ids:
                    if route_id in seen:
                        continue
                    seen.add(route_id)
                    out.append(route_id)
                if out:
                    return out

        return []

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
        tokens.extend(self._prefixed_route_list_tokens(src))
        # Bracket format: [7], [J], [SIR]
        tokens.extend(re.findall(r"\[([A-Za-z0-9\-\+]+)\]", src))
        # Alphanumeric bus/route codes: B46, M15-SBS, Q58, BX39
        tokens.extend(re.findall(r"\b([A-Za-z]{1,4}\d{1,3}[A-Za-z+\-]*)\b", src))
        # "X trains" / "X train" format: "J trains", "F train"
        tokens.extend(
            re.findall(r"\b([A-Za-z0-9]{1,3})\s+trains?\b", src, flags=re.IGNORECASE)
        )
        # "direction-bound X": "Flushing-bound 7", "Woodlawn-bound 4", "Manhattan-bound J"
        tokens.extend(re.findall(r"\b[A-Za-z]+-bound\s+([A-Z0-9]{1,3})\b", src))
        # "no X between/service/trains": "no F between", "no 7 between", "no B service"
        tokens.extend(
            re.findall(r"\bno\s+([A-Z0-9]{1,3})\b", src, flags=re.IGNORECASE)
        )
        # "X skips/runs/bypasses": "7 skips", "J skips", "Q runs", "D runs local"
        tokens.extend(
            re.findall(r"\b([A-Z0-9]{1,3})\s+(?:skips?|runs?\b|bypasses?)", src)
        )
        tokens.extend(self.route_alias_matches(src))
        return tokens

    def _prefixed_route_list_tokens(self, text: str) -> List[str]:
        out: List[str] = []
        src = text or ""
        if not src:
            return out

        for prefix, raw_list in re.findall(r"\b([A-Za-z]{1,3})\s*:\s*([^\n.]+)", src):
            prefix_token = prefix.strip().upper()
            if not prefix_token:
                continue

            expanded: List[str] = []
            for raw_item in re.split(
                r"\s*,\s*|\s+and\s+|\s*/\s*", raw_list.strip(), flags=re.IGNORECASE
            ):
                item = raw_item.strip()
                if not item:
                    continue
                if not re.fullmatch(
                    r"\d{1,3}[A-Za-z]?(?:-SBS|SBS|\+)?", item, flags=re.IGNORECASE
                ):
                    continue
                route_id = self.normalize_route_id(f"{prefix_token}{item}")
                if route_id in self.route_nodes_by_id:
                    expanded.append(route_id)

            if len(expanded) >= 2:
                out.extend(expanded)
        return out

    def _seed_route_entities(
        self, seed_entities: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
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
            if isinstance(e, dict)
            and self._normalize_route_token(str(e.get("route_id", ""))) == route_id
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
