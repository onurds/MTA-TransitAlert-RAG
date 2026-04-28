from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from .constants import AFFECTED_LINE_MARKERS, ALT_LINE_MARKERS, LOCATION_PATTERNS


class LocationHintMixin:
    @staticmethod
    def _split_affected_and_alternative_segments(text: str) -> Tuple[List[str], List[str]]:
        parts = [p.strip() for p in re.split(r"[\n\r\.;]+", text or "") if p.strip()]
        affected: List[str] = []
        alternative: List[str] = []

        for part in parts:
            lower = part.lower()
            has_alt = any(marker in lower for marker in ALT_LINE_MARKERS)
            has_affected = any(marker in lower for marker in AFFECTED_LINE_MARKERS)

            if has_alt and has_affected:
                split_match = re.search(
                    r"\b(?:use\s+the\s+stop|use\s+stops|take\s+the|instead)\b",
                    lower,
                )
                if split_match:
                    split_idx = split_match.start()
                    left = part[:split_idx].strip(" ,;:-")
                    right = part[split_idx:].strip(" ,;:-")
                    if left:
                        affected.append(left)
                    if right:
                        alternative.append(right)
                    continue
                affected.append(part)
                continue

            if has_alt:
                alternative.append(part)
                continue
            if has_affected:
                affected.append(part)
                continue
            affected.append(part)

        return affected, alternative

    @staticmethod
    def _clean_location_hint(hint: str) -> str:
        txt = str(hint or "").strip()
        if not txt:
            return ""
        txt = txt.replace(" & ", " and ")
        txt = " ".join(txt.split()).strip(" ,.;:-")
        return txt

    @staticmethod
    def _clean_extracted_location_hint(hint: str) -> str:
        txt = str(hint or "").strip()
        if not txt:
            return ""
        txt = re.sub(r"\*{1,3}|_{1,3}", "", txt).strip()
        txt = re.split(r"\b(?:has|have|had|is|are|was|were|will|would|should|can|could|may|might|must)\b", txt, 1, flags=re.IGNORECASE)[0]
        txt = re.sub(r"\s*-\s*no\s+stops?.*$", "", txt, flags=re.IGNORECASE).strip()
        txt = re.sub(r"\bno\s+stops?\s+will\s+be\s+missed\b.*$", "", txt, flags=re.IGNORECASE).strip()
        txt = re.sub(r"\bno\s+stops?\b.*$", "", txt, flags=re.IGNORECASE).strip()
        txt = re.sub(r"^(?:the\s+)?(?:stop\s+on|stops?\s+on|on|at|near)\s+", "", txt, flags=re.IGNORECASE).strip()
        txt = txt.replace(" & ", " and ")
        txt = " ".join(txt.split()).strip(" ,.;:-")
        return txt

    @staticmethod
    def _looks_like_location_hint(hint: str) -> bool:
        txt = (hint or "").strip().lower()
        if not txt:
            return False

        if " to " in txt:
            return False
        if "no stop" in txt:
            return False

        temporal_tokens = (
            "today",
            "tomorrow",
            "tonight",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "am",
            "pm",
            "hour",
            "hours",
            "minute",
            "minutes",
            "week",
            "weeks",
            "time frame",
            "time frames",
        )
        if any(tok in txt for tok in temporal_tokens):
            return False

        road_markers = (
            " st",
            " street",
            " ave",
            " avenue",
            " blvd",
            " boulevard",
            " rd",
            " road",
            " pkwy",
            " parkway",
            " dr",
            " drive",
            " ln",
            " lane",
            " pl",
            " place",
            " expy",
            " expressway",
            " hwy",
            " highway",
            " station",
            " terminal",
        )
        if any(marker in f" {txt}" for marker in road_markers):
            return True

        return bool(re.search(r"\b\d{1,3}(?:st|nd|rd|th)\b", txt))

    @staticmethod
    def _extract_location_hints(text: str) -> List[str]:
        hints: List[str] = []
        for pattern in LOCATION_PATTERNS:
            for match in pattern.findall(text or ""):
                if isinstance(match, tuple):
                    pieces = [LocationHintMixin._clean_extracted_location_hint(m) for m in match if m and str(m).strip()]
                    for piece in pieces:
                        hints.extend(LocationHintMixin._expand_location_hint(piece))
                else:
                    candidate = LocationHintMixin._clean_extracted_location_hint(match)
                    hints.extend(LocationHintMixin._expand_location_hint(candidate))

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
    def _merge_location_hints(extracted_hints: Sequence[str], override_hints: Optional[Sequence[str]]) -> List[str]:
        merged: List[str] = []
        seen = set()
        # Extracted hints (from text) go through _expand_location_hint which applies
        # the road-marker filter — appropriate for noisy text extraction.
        for raw in extracted_hints or []:
            cleaned = LocationHintMixin._clean_extracted_location_hint(str(raw or ""))
            for expanded in LocationHintMixin._expand_location_hint(cleaned):
                key = expanded.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(expanded)
        # Override hints are explicitly provided by the orchestrator (from the LLM's
        # location_phrases or from GTFS entity IDs) — trust them directly without the
        # road-marker filter, which rejects valid station names like "Court Sq".
        # Still apply the "to" split so "X to Y" produces two separate hints.
        for raw in override_hints or []:
            cleaned = LocationHintMixin._clean_location_hint(str(raw or ""))
            if not cleaned:
                continue
            if " to " in cleaned.lower():
                parts = [LocationHintMixin._clean_location_hint(p) for p in re.split(r"\bto\b", cleaned, maxsplit=1, flags=re.IGNORECASE)]
                candidates = [p for p in parts if p]
            else:
                candidates = [cleaned]
            for candidate in candidates:
                key = candidate.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(candidate)
        return merged

    @staticmethod
    def _expand_location_hint(hint: str) -> List[str]:
        txt = LocationHintMixin._clean_location_hint(hint)
        if not txt:
            return []
        if LocationHintMixin._looks_like_location_hint(txt):
            return [txt]

        if " to " in txt.lower():
            parts = [LocationHintMixin._clean_location_hint(p) for p in re.split(r"\bto\b", txt, maxsplit=1, flags=re.IGNORECASE)]
            out: List[str] = []
            for part in parts:
                if part and LocationHintMixin._looks_like_location_hint(part):
                    out.append(part)
            if out:
                return out
        return []

    @staticmethod
    def _has_stop_indication(text: str, location_hints: Sequence[str]) -> bool:
        if location_hints:
            return True
        lower = (text or "").lower()
        stop_markers = (
            " at ",
            " between ",
            " from ",
            " near ",
            " stop on",
            " stop id",
            " station",
            " intersection",
        )
        return any(marker in lower for marker in stop_markers)

    @staticmethod
    def _has_strong_stop_intent(text: str) -> bool:
        lower = (text or "").lower()
        if not lower.strip():
            return False

        if re.search(r"\bstop(?:\s*id)?\b", lower):
            return True

        # High-signal stop-action cues.
        action_markers = (
            "bypass",
            "bypassing",
            "not make stop",
            "not make stops",
            "skip stop",
            "skip stops",
            "stop moved",
            "stop relocated",
            "board or exit",
            "use the stop",
            "detour",
            "detoured",
        )
        if any(m in lower for m in action_markers):
            return True

        # Explicit geometric location cues.
        if " near " in lower or " intersection " in lower:
            return True
        if re.search(r"\bon\s+.+\s+at\s+.+", lower):
            return True
        if re.search(r"\bbetween\s+.+\s+and\s+.+", lower):
            return True
        if re.search(r"\bfrom\s+.+\s+to\s+.+", lower):
            return True
        return False

    @staticmethod
    def _parse_hint_constraint(hint: str) -> Dict[str, set]:
        tokens = LocationHintMixin._norm_text_tokens(hint)
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
        stop_tokens = set(LocationHintMixin._norm_text_tokens(stop_name))
        numbers = constraint.get("numbers", set())
        road_tokens = constraint.get("road_tokens", set())

        if numbers and not (numbers & stop_tokens):
            return False
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
