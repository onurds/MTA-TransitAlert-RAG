from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


class TextRenderer:
    def __init__(self, retriever: Any):
        self.retriever = retriever

    @staticmethod
    def clean_header_text(text: str) -> str:
        out = (text or "").strip()
        if not out:
            return out

        stop_markers = [
            r"\bwhat(?:['’])?s\s+happening\??",
            r"\bsee\s+a\s+map\b",
            r"\bplan\s+your\s+trip\b",
            r"\b(?:the\s+)?dates?\s+(?:should\s+be|are|is)\b",
            r"\bdates?\s*:",
            r"\btime(?:\s*frame)?\s+(?:is|are)\b",
        ]
        cut_positions = []
        for marker in stop_markers:
            m = re.search(marker, out, flags=re.IGNORECASE)
            if m:
                cut_positions.append(m.start())
        if cut_positions:
            out = out[: min(cut_positions)].strip()

        return re.sub(r"\s{2,}", " ", out).strip(" ,.;:-")

    @staticmethod
    def strip_instruction_meta(text: str) -> str:
        out = (text or "").strip()
        if not out:
            return out
        out = re.sub(r"\b(?:the\s+)?dates?\s+(?:should\s+be|are|is)\s+.+$", "", out, flags=re.IGNORECASE | re.DOTALL).strip()
        out = re.sub(r"\bdates?\s*:\s*.+$", "", out, flags=re.IGNORECASE | re.DOTALL).strip()
        out = re.sub(r"\btime(?:\s*frame)?\s+(?:is|are)\s+.+$", "", out, flags=re.IGNORECASE | re.DOTALL).strip()
        out = re.sub(r"\s{2,}", " ", out).strip(" ,")
        return out

    def replace_stop_ids_with_names(self, text: str, entities: Sequence[Dict[str, Any]]) -> str:
        out = (text or "").strip()
        if not out:
            return out

        candidate_ids = set(re.findall(r"\bstop\s*id\s*([A-Za-z0-9]{3,10}[NS]?)\b", out, flags=re.IGNORECASE))
        for e in entities or []:
            if isinstance(e, dict) and e.get("stop_id"):
                candidate_ids.add(str(e.get("stop_id")))

        for sid_raw in sorted(candidate_ids, key=len, reverse=True):
            sid = self.retriever.normalize_stop_id(sid_raw)
            if not sid:
                continue
            stop_name = self.retriever.stop_name_for_id(sid)
            if not stop_name:
                continue
            pretty_name = self._humanize_stop_name(stop_name)
            escaped = re.escape(sid_raw)
            out = re.sub(rf"\bat\s+stop\s*id\s*{escaped}\b", f"at {pretty_name}", out, flags=re.IGNORECASE)
            out = re.sub(rf"\bstop\s*id\s*{escaped}\b", pretty_name, out, flags=re.IGNORECASE)

        return re.sub(r"\s{2,}", " ", out).strip()

    @staticmethod
    def _humanize_stop_name(stop_name: str) -> str:
        name = (stop_name or "").strip()
        if not name:
            return name
        name = name.replace("/", " at ")
        name = re.sub(r"\s{2,}", " ", name)
        titled = name.title()
        titled = re.sub(r"\bN\b", "N", titled)
        titled = re.sub(r"\bS\b", "S", titled)
        titled = re.sub(r"\bE\b", "E", titled)
        titled = re.sub(r"\bW\b", "W", titled)
        return titled

    def format_route_bullets(self, text: str, route_ids: Sequence[str]) -> str:
        out = (text or "").strip()
        if not out:
            return out

        unique_ids: List[str] = []
        seen = set()
        for rid in route_ids:
            token = str(rid or "").strip().upper()
            if not token or token in seen:
                continue
            seen.add(token)
            unique_ids.append(token)

        unique_ids.sort(key=len, reverse=True)
        display_by_route = {
            rid: self.retriever.route_display_name_for_id(rid)
            for rid in unique_ids
        }

        # Replace canonical internal codes with rider-facing names when available.
        for rid in unique_ids:
            display = display_by_route.get(rid)
            if not display:
                continue
            variants = [rid]
            if "+" in rid:
                variants.append(rid.replace("+", "-SBS"))
            for variant in variants:
                escaped = re.escape(variant)
                out = re.sub(rf"\[{escaped}\]", display, out, flags=re.IGNORECASE)
                out = re.sub(rf"\b{escaped}\b", display, out, flags=re.IGNORECASE)

        for rid in unique_ids:
            # Already substituted to long display name; do not bracket internal code.
            if display_by_route.get(rid):
                continue
            variants = [rid]
            if "+" in rid:
                variants.append(rid.replace("+", "-SBS"))

            for variant in variants:
                escaped = re.escape(variant)
                pattern = re.compile(rf"(?<!\[)\b{escaped}\b(?!\])", flags=re.IGNORECASE)
                out = pattern.sub(lambda m: f"[{m.group(0)}]", out)

        return out

    def collapse_route_parenthetical_duplicates(self, text: str) -> str:
        out = (text or "").strip()
        if not out:
            return out

        # Collapse patterns like "42 St Shuttle (42 St Shuttle)" when both
        # sides resolve to the same canonical GTFS route ID.
        pattern = re.compile(
            r"([A-Za-z0-9][A-Za-z0-9/&\-\.' ]{1,80}?)\s*\(\s*([A-Za-z0-9][A-Za-z0-9/&\-\.' ]{1,80}?)\s*\)"
        )

        def _replace(m: re.Match[str]) -> str:
            left = re.sub(r"\s{2,}", " ", m.group(1)).strip()
            right = re.sub(r"\s{2,}", " ", m.group(2)).strip()
            if not left or not right:
                return m.group(0)

            left_id = self.retriever.normalize_route_id(left)
            right_id = self.retriever.normalize_route_id(right)
            if (
                left_id
                and right_id
                and left_id == right_id
                and left_id in self.retriever.route_nodes_by_id
            ):
                return self.retriever.route_display_name_for_id(left_id) or left
            return m.group(0)

        # Apply twice for nested/adjacent occurrences.
        for _ in range(2):
            new_out = pattern.sub(_replace, out)
            if new_out == out:
                break
            out = new_out
        return out
