from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from .models import EvidenceUnit


CONTROL_PATTERNS = (
    r"\btimeframe\s+is\b",
    r"\bdates?\s+(?:will\s+be|are|should\s+be)\b",
    r"\bset\s+the\s+dates?\b",
    r"\bmake\s+the\s+timeframe\b",
    r"\buse\s+this\s+as\s+the\s+header\b",
    r"\bmake\s+this\s+bold\b",
    r"\bmake\s+sure\s+to\s+get\s+it\s+right\b",
    r"\bverify\s+the\s+route\s+information\b",
    r"\bsee\s+ticket\s*#?\w+\b",
    r"\btkt-\w+\b",
    r"\bheader\s*:",
    r"\bdescription\s*:",
)

RIDER_GUIDANCE_PATTERNS = (
    r"\bwhat'?s happening\??\b",
    r"\bnote:\b",
    r"\bsee a map of this stop change\b",
    r"\bbuilding construction\b",
    r"\bbus arrival information may not be available\b",
)

TEMPORAL_PATTERNS = (
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
    "from now",
    "until",
    "through",
    "for the next",
    "week",
    "weeks",
    "hour",
    "hours",
    "minute",
    "minutes",
    "am",
    "pm",
)

LOCATION_PATTERNS = (
    re.compile(
        r"\bon\s+([A-Za-z0-9\-\./&' ]+?)\s+at\s+([A-Za-z0-9\-\./&' ]+?)(?=\.|,|;|\n|$|\b(?:has|have|is|are|was|were|will)\b)",
        re.IGNORECASE,
    ),
    re.compile(r"\bat\s+([A-Za-z0-9\-\./&' ]+?)(?=\.|,|;|\n|$)", re.IGNORECASE),
    re.compile(
        r"\bbetween\s+([A-Za-z0-9\-\./&' ]+?)\s+and\s+([A-Za-z0-9\-\./&' ]+?)(?=\.|,|;|\n|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfrom\s+([A-Za-z0-9\-\./&' ]+?)\s+to\s+([A-Za-z0-9\-\./&' ]+?)(?=\.|,|;|\n|$)",
        re.IGNORECASE,
    ),
)


def decompose_instruction(instruction: str) -> List[EvidenceUnit]:
    text = str(instruction or "").strip()
    if not text:
        return []

    units: List[EvidenceUnit] = []
    parts = [p.strip() for p in re.split(r"[\n\r]+|(?<=[.!?;])\s+", text) if p.strip()]
    for part in parts:
        kind = _classify_part(part)
        units.append(EvidenceUnit(unit_type=kind, text=part))
        if kind in {"affected_service", "alternative_service", "rider_guidance"}:
            for hint in _extract_location_hints(part):
                units.append(EvidenceUnit(unit_type="location_evidence", text=hint, source=kind))
    return _dedupe_units(units)


def summarize_evidence_units(units: Sequence[EvidenceUnit]) -> List[dict]:
    return [{"type": u.unit_type, "text": u.text, "source": u.source} for u in units]


def evidence_text(units: Sequence[EvidenceUnit], include_types: Iterable[str]) -> str:
    allowed = set(include_types)
    parts = [u.text for u in units if u.unit_type in allowed and str(u.text).strip()]
    return "\n".join(parts).strip()


def rider_text(units: Sequence[EvidenceUnit]) -> str:
    return evidence_text(units, ("affected_service", "alternative_service", "rider_guidance"))


def command_stripped_instruction(units: Sequence[EvidenceUnit]) -> str:
    return rider_text(units)


def _classify_part(part: str) -> str:
    lower = f" {part.lower()} "
    if _looks_temporal(part) and any(
        token in lower for token in ("timeframe", "date", "dates", "until", "from now", "today", "tomorrow")
    ):
        return "temporal_directive"
    if any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in CONTROL_PATTERNS):
        return "operator_control"
    if any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in RIDER_GUIDANCE_PATTERNS):
        return "rider_guidance"
    if _looks_temporal(part):
        return "temporal_directive"
    if _looks_alternative(part):
        return "alternative_service"
    return "affected_service"


def _looks_temporal(text: str) -> bool:
    lower = (text or "").lower()
    if re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", lower):
        return True
    return any(token in lower for token in TEMPORAL_PATTERNS)


def _looks_alternative(text: str) -> bool:
    lower = (text or "").lower()
    markers = (
        "use the stop",
        "take the",
        "instead",
        "board or exit at",
        "customers may use",
        "no stops will be missed",
    )
    return any(marker in lower for marker in markers)


def _extract_location_hints(text: str) -> List[str]:
    out: List[str] = []
    for pattern in LOCATION_PATTERNS:
        for match in pattern.findall(text or ""):
            if isinstance(match, tuple):
                for piece in match:
                    hint = _clean_hint(piece)
                    if hint:
                        out.append(hint)
            else:
                hint = _clean_hint(match)
                if hint:
                    out.append(hint)
    deduped: List[str] = []
    seen = set()
    for hint in out:
        key = hint.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hint)
    return deduped


def _clean_hint(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^(?:the\s+)?(?:stop\s+on|stops?\s+on|on|at|near)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bno\s+stops?\s+will\s+be\s+missed\b.*$", "", value, flags=re.IGNORECASE)
    value = " ".join(value.replace(" & ", " and ").split()).strip(" ,.;:-")
    return value


def _dedupe_units(units: Sequence[EvidenceUnit]) -> List[EvidenceUnit]:
    out: List[EvidenceUnit] = []
    seen = set()
    for unit in units:
        key = (unit.unit_type, unit.text.strip().lower(), unit.source)
        if not unit.text.strip() or key in seen:
            continue
        seen.add(key)
        out.append(unit)
    return out
