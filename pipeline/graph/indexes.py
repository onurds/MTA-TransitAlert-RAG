from __future__ import annotations

import pickle
import re
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx
import numpy as np

from .constants import AGENCY_ID_BY_GTFS_NAMESPACE, RoutePattern


class GraphIndexMixin:
    graph_path: str
    G: nx.DiGraph
    route_nodes_by_id: Dict[str, List[str]]
    route_agency_by_node: Dict[str, str]
    stop_nodes_by_id: Dict[str, List[str]]
    stop_agency_by_id: Dict[str, str]
    _stop_coords: Optional[np.ndarray]
    _stop_tree_nodes: List[str]
    route_long_name_by_id: Dict[str, str]
    route_short_name_by_id: Dict[str, str]
    route_phrase_to_id: Dict[str, str]
    route_phrase_to_agency: Dict[str, str]
    route_code_aliases: Dict[str, str]
    route_display_name_by_id: Dict[str, str]
    route_patterns_by_node: Dict[str, List[RoutePattern]]

    def _load_graph(self) -> None:
        print(f"Loading Knowledge Graph from {self.graph_path}...")
        try:
            with open(self.graph_path, "rb") as f:
                self.G = pickle.load(f)
            print(f"Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error loading graph: {e}")
            self.G = nx.DiGraph()

    def _build_route_indexes(self) -> None:
        self.route_nodes_by_id.clear()
        self.route_agency_by_node.clear()
        self.route_long_name_by_id.clear()
        self.route_short_name_by_id.clear()
        self.route_phrase_to_id.clear()
        self.route_phrase_to_agency.clear()
        self.route_display_name_by_id.clear()
        self.route_patterns_by_node = {}
        self.route_code_aliases = {"SIR": "SI"}
        stop_name_phrases = self._stop_name_phrase_index()

        for node, attrs in self.G.nodes(data=True):
            if attrs.get("type") != "Route":
                continue

            route_id = self.to_public_route_id(node)
            self.route_nodes_by_id.setdefault(route_id, []).append(node)
            namespace = self._namespace_from_node(node)
            agency_id = AGENCY_ID_BY_GTFS_NAMESPACE.get(namespace, "MTA NYCT")
            self.route_agency_by_node[node] = agency_id
            short_name = str(attrs.get("short_name", "") or "").strip()
            long_name = str(attrs.get("long_name", "") or "").strip()
            if short_name:
                self.route_short_name_by_id.setdefault(route_id, short_name)
            if long_name:
                self.route_long_name_by_id.setdefault(route_id, long_name)
            for phrase in self._route_alias_phrases(
                route_id=route_id,
                short_name=short_name,
                long_name=long_name,
            ):
                self._register_route_phrase_alias(
                    phrase=phrase,
                    route_id=route_id,
                    agency_id=agency_id,
                    stop_name_phrases=stop_name_phrases,
                )

            self.route_patterns_by_node[node] = self._coerce_route_patterns(
                attrs.get("stop_patterns")
            )

            # For commuter-rail routes (LIRR / MNR) there is no short_name, so the
            # only textual handle is the long_name (e.g. "Harlem", "New Haven",
            # "Babylon Branch").  Riders commonly say "Harlem Line" or
            # "New Haven Line", which the plain long_name alias can't cover because
            # the guardrail rejects single-token phrases and the stop-name guardrail
            # rejects multi-word phrases that clash with station names.
            # Generate explicit "X Line" and "X Branch" variants here, where we know
            # the namespace, to guarantee these route-specific phrases are indexed.
            if namespace in ("gtfslirr", "gtfsmnr") and long_name:
                base = long_name.strip()
                # Strip any existing "Branch" / "Service" suffix before adding our own
                base_core = re.sub(r"\s+(Branch|Service|Line)$", "", base, flags=re.IGNORECASE).strip()
                for suffix in (" Line", " Branch"):
                    alias = base_core + suffix
                    if alias.lower() != base.lower():  # avoid dup if already ends with that word
                        self._register_route_phrase_alias(
                            phrase=alias,
                            route_id=route_id,
                            agency_id=agency_id,
                            stop_name_phrases=stop_name_phrases,
                        )

        for rid in self.route_nodes_by_id:
            self.route_nodes_by_id[rid].sort()

        # Rider-facing display substitutions for special normalized shuttle codes.
        for rid in ("GS", "FS", "H"):
            long_name = self.route_long_name_by_id.get(rid)
            if long_name:
                self.route_display_name_by_id[rid] = long_name

    @staticmethod
    def _coerce_route_patterns(raw_patterns: Any) -> List[RoutePattern]:
        out: List[RoutePattern] = []
        if not isinstance(raw_patterns, list):
            return out
        for raw in raw_patterns:
            if not isinstance(raw, dict):
                continue
            stop_nodes = tuple(
                str(node).strip()
                for node in (raw.get("stop_nodes") or [])
                if str(node).strip()
            )
            public_stop_ids = tuple(
                str(stop_id).strip().upper()
                for stop_id in (raw.get("public_stop_ids") or [])
                if str(stop_id).strip()
            )
            if len(stop_nodes) < 2 or len(stop_nodes) != len(public_stop_ids):
                continue
            out.append(
                RoutePattern(
                    pattern_id=str(raw.get("pattern_id") or "").strip() or "pattern",
                    direction_id=str(raw.get("direction_id") or "").strip(),
                    headsign=str(raw.get("headsign") or "").strip(),
                    count=int(raw.get("count") or 0),
                    stop_nodes=stop_nodes,
                    public_stop_ids=public_stop_ids,
                )
            )
        out.sort(key=lambda p: (p.count, len(p.stop_nodes)), reverse=True)
        return out

    def _route_alias_phrases(
        self,
        route_id: str,
        short_name: str,
        long_name: str,
    ) -> List[str]:
        out: List[str] = []
        if short_name:
            out.append(short_name)
        if long_name:
            out.append(long_name)
            # Route long names often include endpoint pairs after separators.
            # Keep a route-specific "primary" segment only when clearly route-like.
            primary = re.split(r"\s*-\s*|,\s*|\bvia\b", long_name, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            primary_norm = self._normalize_route_phrase(primary)
            if (
                primary
                and primary.lower() != long_name.lower()
                and primary_norm
                and self._is_route_specific_phrase(primary_norm, route_id)
            ):
                out.append(primary)

        # De-dupe raw forms before normalization.
        deduped: List[str] = []
        seen = set()
        for value in out:
            key = str(value or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(str(value).strip())
        return deduped

    def _register_route_phrase_alias(
        self,
        phrase: str,
        route_id: str,
        agency_id: str,
        stop_name_phrases: set[str],
    ) -> None:
        norm = self._normalize_route_phrase(phrase)
        if not norm:
            return
        tokens = norm.split()
        if len(tokens) == 1:
            token = tokens[0]
            # Reject ambiguous one-letter aliases ("a", "f", etc.) that can
            # collide with ordinary language. Keep richer codes like "sir".
            if len(token) < 2 and not re.search(r"\d", token):
                return
        is_route_specific = self._is_route_specific_phrase(norm, route_id)
        # Avoid broad aliases like single endpoint names.
        if len(tokens) < 2 and not is_route_specific:
            return
        # Guardrail: if a phrase also looks like a stop name, only keep it when
        # it carries route-specific cues.
        if norm in stop_name_phrases and not is_route_specific:
            return
        self.route_phrase_to_id.setdefault(norm, route_id)
        self.route_phrase_to_agency.setdefault(norm, agency_id)

    def _is_route_specific_phrase(self, norm_phrase: str, route_id: str) -> bool:
        words = set(norm_phrase.split())
        route_cues = {
            "shuttle",
            "bus",
            "buses",
            "train",
            "trains",
            "line",
            "link",
            "sbs",
            "railway",
            "express",
            # Commuter rail — LIRR uses "Branch" in every route long_name
            "branch",
        }
        if words & route_cues:
            return True

        rid_norm = self._normalize_route_phrase(route_id.replace("+", " sbs "))
        if rid_norm and rid_norm in words:
            return True
        return bool(re.search(r"\b[a-z]+\d+\b", norm_phrase))

    def _stop_name_phrase_index(self) -> set[str]:
        out: set[str] = set()
        for _node, attrs in self.G.nodes(data=True):
            if attrs.get("type") != "Stop":
                continue
            stop_name = str(attrs.get("name", "") or "").strip()
            norm = self._normalize_route_phrase(stop_name)
            if norm:
                out.add(norm)
            for part in re.split(r"\s*/\s*|\s+at\s+", stop_name, flags=re.IGNORECASE):
                part_norm = self._normalize_route_phrase(part)
                if part_norm and len(part_norm.split()) >= 2:
                    out.add(part_norm)
        return out

    def _build_stop_indexes(self) -> None:
        self.stop_nodes_by_id.clear()
        self.stop_agency_by_id.clear()

        for node, attrs in self.G.nodes(data=True):
            if attrs.get("type") != "Stop":
                continue
            stop_id = self.to_public_stop_id(node)
            self.stop_nodes_by_id.setdefault(stop_id, []).append(node)

            namespace = self._namespace_from_node(node)
            agency_id = AGENCY_ID_BY_GTFS_NAMESPACE.get(namespace, "MTA NYCT")
            self.stop_agency_by_id.setdefault(stop_id, agency_id)

        for sid in self.stop_nodes_by_id:
            self.stop_nodes_by_id[sid].sort()

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

    @classmethod
    def to_station_stop_id(cls, stop_node_or_id: str) -> str:
        value = cls.to_public_stop_id(stop_node_or_id)
        if re.fullmatch(r"[A-Z0-9]+[NS]", value):
            return value[:-1]
        return value

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

    @staticmethod
    def _normalize_route_phrase(text: str) -> str:
        out = (text or "").strip().lower()
        if not out:
            return ""
        out = re.sub(r"[^a-z0-9 ]+", " ", out)
        replacements = (
            (" avenue ", " av "),
            (" street ", " st "),
        )
        out = f" {out} "
        for src, dst in replacements:
            out = out.replace(src, dst)
        out = re.sub(r"\s{2,}", " ", out).strip()
        return out

    def route_alias_matches(self, text: str) -> List[str]:
        norm_text = self._normalize_route_phrase(text)
        if not norm_text:
            return []
        hay = f" {norm_text} "
        hits: List[tuple[int, str]] = []
        for phrase, route_id in self.route_phrase_to_id.items():
            pos = self._route_phrase_position_with_context(hay, phrase)
            if pos >= 0:
                hits.append((pos, route_id))
        hits.sort(key=lambda x: x[0])
        out: List[str] = []
        seen = set()
        for _, rid in hits:
            if rid in seen:
                continue
            seen.add(rid)
            out.append(rid)
        return out

    def is_route_name_phrase(self, text: str) -> bool:
        norm = self._normalize_route_phrase(text)
        if not norm:
            return False
        return norm in self.route_phrase_to_id

    def link_route_mentions(self, mentions: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for mention in mentions or []:
            for route_id, _agency_id in self._route_links_from_mention(str(mention or "")):
                if route_id in seen:
                    continue
                seen.add(route_id)
                out.append(route_id)
        return out

    def link_route_mention_entities(self, mentions: Sequence[str]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        seen = set()
        for mention in mentions or []:
            for route_id, agency_id in self._route_links_from_mention(str(mention or "")):
                key = (agency_id, route_id)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"agency_id": agency_id, "route_id": route_id})
        return out

    def _route_links_from_mention(self, mention: str) -> List[tuple[str, str]]:
        raw = str(mention or "").strip()
        if not raw:
            return []

        route_id = self.normalize_route_id(raw)
        if route_id in self.route_nodes_by_id:
            norm = self._normalize_route_phrase(raw)
            agency_id = self.route_phrase_to_agency.get(norm) or self.agency_for_route_id(route_id)
            return [(route_id, agency_id)]

        out: List[tuple[str, str]] = []
        seen = set()
        for token in self._route_code_tokens_from_mention(raw):
            route_id = self.normalize_route_id(token)
            if route_id not in self.route_nodes_by_id or route_id in seen:
                continue
            seen.add(route_id)
            out.append((route_id, self.agency_for_route_id(route_id)))
        return out

    def _route_code_tokens_from_mention(self, mention: str) -> List[str]:
        raw = str(mention or "")
        if not raw:
            return []

        # Only inspect a span the LLM already classified as an affected route
        # mention. This recovers route codes inside descriptive spans such as
        # "downtown 4 local" without scanning unrelated alert text.
        pattern = re.compile(
            r"\[[A-Za-z0-9+ -]+\]"
            r"|[A-Za-z]{1,3}\d+[A-Za-z]?(?:[- ]?SBS)?"
            r"|\b\d{1,2}\b"
            r"|\b[A-Z]\b"
        )
        tokens: List[str] = []
        seen = set()
        for match in pattern.finditer(raw):
            token = match.group(0).strip()
            if not self._route_code_token_allowed_in_mention(raw, token, match.start(), match.end()):
                continue
            norm = self._normalize_route_token(token)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            tokens.append(token)
        return tokens

    @staticmethod
    def _route_code_token_allowed_in_mention(
        mention: str,
        token: str,
        start: int = -1,
        end: int = -1,
    ) -> bool:
        bare = str(token or "").strip().strip("[](){}.,;:")
        if not bare:
            return False
        if re.search(r"[A-Za-z]{1,3}\d", bare):
            return True
        if re.fullmatch(r"[A-Z]", bare):
            return True
        if re.fullmatch(r"\d{1,2}", bare):
            prev_char = mention[start - 1] if start > 0 else ""
            next_char = mention[end] if 0 <= end < len(mention) else ""
            following = mention[end : end + 4].lower() if end >= 0 else ""
            if prev_char == ":" or next_char == ":" or re.match(r"\s*(am|pm)\b", following):
                return False
            return bool(
                re.search(
                    r"\b(train|trains|service|line|local|express|uptown|downtown|bound)\b",
                    mention,
                    flags=re.IGNORECASE,
                )
            )
        return bool(token.strip().startswith("[") and token.strip().endswith("]"))

    @staticmethod
    def _route_phrase_position_with_context(hay: str, phrase: str) -> int:
        needle = f" {phrase} "
        pos = hay.find(needle)
        if pos < 0:
            return -1

        tokens = phrase.split()
        # Guardrail: pure numeric aliases (e.g., "3") should only match when the
        # surrounding language implies a route reference, not street numbers.
        if len(tokens) == 1 and tokens[0].isdigit():
            token = re.escape(tokens[0])
            context_pattern = re.compile(
                rf"(?:\[{token}\])|(?:\b{token}\b\s+(?:train|trains|line|service|express)\b)",
                flags=re.IGNORECASE,
            )
            if not context_pattern.search(hay):
                return -1
        return pos

    @staticmethod
    def _normalize_stop_token(token: str) -> str:
        t = str(token or "").strip().upper()
        return t.strip("[](){}.,;:#")

    def normalize_route_id(self, token: str) -> str:
        rid = self._normalize_route_token(token)
        rid = self.route_code_aliases.get(rid, rid)
        if rid in self.route_nodes_by_id:
            return rid
        if rid.endswith("+") and rid[:-1] in self.route_nodes_by_id:
            return rid[:-1]

        phrase = self._normalize_route_phrase(str(token or ""))
        if phrase and phrase in self.route_phrase_to_id:
            return self.route_phrase_to_id[phrase]

        return rid

    def normalize_stop_id(self, token: str) -> str:
        return self._normalize_stop_token(token)

    def is_valid_route_id(self, route_id: str) -> bool:
        return self.normalize_route_id(route_id) in self.route_nodes_by_id

    def is_valid_stop_id(self, stop_id: str) -> bool:
        return self.normalize_stop_id(stop_id) in self.stop_nodes_by_id

    def validate_route_ids(self, route_ids: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for token in route_ids or []:
            rid = self.normalize_route_id(token)
            if not rid or rid in seen:
                continue
            if rid in self.route_nodes_by_id:
                seen.add(rid)
                out.append(rid)
        return out

    def validate_stop_ids(self, stop_ids: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for token in stop_ids or []:
            sid = self.normalize_stop_id(token)
            if not sid or sid in seen:
                continue
            if sid in self.stop_nodes_by_id:
                seen.add(sid)
                out.append(sid)
        return out

    def route_patterns_for_node(self, route_node_id: str) -> List[RoutePattern]:
        return list(self.route_patterns_by_node.get(route_node_id, []))

    def agency_for_route_id(self, route_id: str) -> str:
        rid = self.normalize_route_id(route_id)
        nodes = self.route_nodes_by_id.get(rid, [])
        if not nodes:
            return "MTA NYCT"
        return self.route_agency_by_node.get(nodes[0], "MTA NYCT")

    def agency_for_stop_id(self, stop_id: str) -> str:
        sid = self.normalize_stop_id(stop_id)
        return self.stop_agency_by_id.get(sid, "MTA NYCT")

    def stop_name_for_id(self, stop_id: str) -> Optional[str]:
        sid = self.normalize_stop_id(stop_id)
        nodes = self.stop_nodes_by_id.get(sid, [])
        if not nodes:
            return None
        attrs = self.G.nodes.get(nodes[0], {})
        name = str(attrs.get("name", "")).strip()
        return name or None

    def route_display_name_for_id(self, route_id: str) -> Optional[str]:
        rid = self.normalize_route_id(route_id)
        return self.route_display_name_by_id.get(rid)

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
