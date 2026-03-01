"""
GTFS/MTA alert guardrails and rule-based enum inference.

The enum sets are aligned with the GTFS-Realtime Alert reference
and used as hard validation guardrails for API output.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Tuple


# GTFS-Realtime Alert.Cause enum (official values).
CAUSE_ENUMS = {
    "UNKNOWN_CAUSE",
    "OTHER_CAUSE",
    "TECHNICAL_PROBLEM",
    "STRIKE",
    "DEMONSTRATION",
    "ACCIDENT",
    "HOLIDAY",
    "WEATHER",
    "MAINTENANCE",
    "CONSTRUCTION",
    "POLICE_ACTIVITY",
    "MEDICAL_EMERGENCY",
}

# GTFS-Realtime Alert.Effect enum (official values).
EFFECT_ENUMS = {
    "NO_SERVICE",
    "REDUCED_SERVICE",
    "SIGNIFICANT_DELAYS",
    "DETOUR",
    "ADDITIONAL_SERVICE",
    "MODIFIED_SERVICE",
    "OTHER_EFFECT",
    "UNKNOWN_EFFECT",
    "STOP_MOVED",
    "NO_EFFECT",
    "ACCESSIBILITY_ISSUE",
}

UNKNOWN_CAUSE = "UNKNOWN_CAUSE"
UNKNOWN_EFFECT = "UNKNOWN_EFFECT"


@dataclass(frozen=True)
class EnumInference:
    cause: str
    effect: str
    cause_confidence: float
    effect_confidence: float
    cause_signals: Tuple[str, ...]
    effect_signals: Tuple[str, ...]


def _normalize(s: str) -> str:
    return " ".join((s or "").lower().replace("/", " ").replace("-", " ").split())


def normalize_cause(value: str) -> str:
    token = (value or "").strip().upper()
    return token if token in CAUSE_ENUMS else UNKNOWN_CAUSE


def normalize_effect(value: str) -> str:
    token = (value or "").strip().upper()
    return token if token in EFFECT_ENUMS else UNKNOWN_EFFECT


def validate_enum(value: str, allowed: Iterable[str], fallback: str) -> str:
    token = (value or "").strip().upper()
    return token if token in set(allowed) else fallback


def infer_cause_effect_rule_first(text: str) -> EnumInference:
    """
    Rule-first inference for cause/effect with transparent confidence scores.
    Values are confidence-gated by the caller.
    """
    normalized = _normalize(text)
    if not normalized:
        return EnumInference(
            cause=UNKNOWN_CAUSE,
            effect=UNKNOWN_EFFECT,
            cause_confidence=0.0,
            effect_confidence=0.0,
            cause_signals=tuple(),
            effect_signals=tuple(),
        )

    cause_rules: Dict[str, List[Tuple[str, float]]] = {
        "MAINTENANCE": [
            ("utility work", 0.95),
            ("track work", 0.92),
            ("maintenance", 0.9),
            ("planned work", 0.9),
            ("broken rail", 0.88),
            ("replaced a broken rail", 0.92),
        ],
        "CONSTRUCTION": [
            ("construction", 0.92),
            ("roadwork", 0.9),
            ("road work", 0.9),
            ("street reconstruction", 0.9),
        ],
        "WEATHER": [
            ("weather", 0.95),
            ("snow", 0.92),
            ("rain", 0.8),
            ("ice", 0.9),
            ("flood", 0.9),
        ],
        "POLICE_ACTIVITY": [
            ("police activity", 0.95),
            ("nypd", 0.9),
            ("investigation", 0.75),
            ("security condition", 0.85),
        ],
        "MEDICAL_EMERGENCY": [
            ("medical emergency", 0.95),
            ("customer injury", 0.85),
            ("sick passenger", 0.85),
        ],
        "TECHNICAL_PROBLEM": [
            ("signal problem", 0.92),
            ("signal malfunction", 0.95),
            ("mechanical", 0.88),
            ("switch problem", 0.9),
            ("power problem", 0.86),
            ("equipment malfunction", 0.86),
        ],
        "ACCIDENT": [
            ("accident", 0.95),
            ("collision", 0.92),
            ("derailment", 0.95),
            ("fdny activity", 0.75),
            ("fire department activity", 0.75),
        ],
        "HOLIDAY": [
            ("holiday schedule", 0.95),
            ("holiday", 0.85),
        ],
        "DEMONSTRATION": [
            ("demonstration", 0.95),
            ("protest", 0.95),
        ],
        "STRIKE": [
            ("strike", 0.95),
            ("labor action", 0.9),
        ],
    }

    effect_rules: Dict[str, List[Tuple[str, float]]] = {
        "DETOUR": [
            ("detour", 0.96),
            ("detoured", 0.96),
            ("operate via", 0.88),
            ("extra turns", 0.82),
        ],
        "SIGNIFICANT_DELAYS": [
            ("running with delays", 0.95),
            ("significant delays", 0.95),
            ("expect delays", 0.92),
            ("wait longer", 0.9),
            ("longer waits", 0.9),
            ("service has resumed", 0.7),
        ],
        "NO_SERVICE": [
            ("no service", 0.97),
            ("suspended", 0.95),
            ("not running", 0.95),
            ("service is suspended", 0.97),
        ],
        "REDUCED_SERVICE": [
            ("reduced service", 0.95),
            ("running as much service as we can", 0.92),
            ("fewer trains", 0.9),
        ],
        "MODIFIED_SERVICE": [
            ("bypassing", 0.9),
            ("skip", 0.82),
            ("not make stops", 0.95),
            ("stop will be closed", 0.9),
            ("service runs in two sections", 0.8),
        ],
        "STOP_MOVED": [
            ("stop moved", 0.95),
            ("board or exit at", 0.86),
            ("use the stop on", 0.8),
        ],
        "ADDITIONAL_SERVICE": [
            ("extra train", 0.95),
            ("additional service", 0.95),
        ],
        "NO_EFFECT": [
            ("no stops will be missed", 0.96),
            ("no impact to service", 0.92),
        ],
        "ACCESSIBILITY_ISSUE": [
            ("elevator", 0.85),
            ("escalator", 0.85),
            ("accessible", 0.75),
            ("ada", 0.75),
        ],
    }

    cause, cause_conf, cause_signals = _score_rule_set(normalized, cause_rules, UNKNOWN_CAUSE)
    effect, effect_conf, effect_signals = _score_rule_set(normalized, effect_rules, UNKNOWN_EFFECT)

    # Basic interaction rules.
    if "no stops will be missed" in normalized and effect == "DETOUR":
        effect = "NO_EFFECT"
        effect_conf = max(effect_conf, 0.94)
        effect_signals = tuple(sorted(set(effect_signals + ("no stops will be missed",))))

    return EnumInference(
        cause=cause,
        effect=effect,
        cause_confidence=max(0.0, min(1.0, cause_conf)),
        effect_confidence=max(0.0, min(1.0, effect_conf)),
        cause_signals=cause_signals,
        effect_signals=effect_signals,
    )


def _score_rule_set(
    text: str,
    rules: Dict[str, List[Tuple[str, float]]],
    unknown_token: str,
) -> Tuple[str, float, Tuple[str, ...]]:
    best_label = unknown_token
    best_conf = 0.0
    best_signals: List[str] = []

    for label, label_rules in rules.items():
        matched = [(kw, score) for kw, score in label_rules if _contains_phrase(text, kw)]
        if not matched:
            continue
        matched.sort(key=lambda x: x[1], reverse=True)
        top_score = matched[0][1]
        # Small bonus for multi-signal consistency.
        bonus = min(0.08, 0.02 * (len(matched) - 1))
        conf = min(1.0, top_score + bonus)
        if conf > best_conf:
            best_label = label
            best_conf = conf
            best_signals = [m[0] for m in matched]

    return best_label, best_conf, tuple(best_signals)


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
    return pattern.search(text) is not None
