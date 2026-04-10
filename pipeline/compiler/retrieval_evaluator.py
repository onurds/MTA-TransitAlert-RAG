from __future__ import annotations

from typing import Any, Dict, Sequence

from .models import EvidenceUnit, RetrievalEvaluationResult


class RetrievalEvaluator:
    def evaluate(
        self,
        retrieval: Dict[str, Any],
        evidence_units: Sequence[EvidenceUnit],
        route_confidence: float,
        stop_confidence: float,
        temporal_override: bool,
    ) -> RetrievalEvaluationResult:
        location_hint_count = len(retrieval.get("location_hints", []) or [])
        matched_stop_count = int(retrieval.get("matched_stop_count", 0) or 0)
        has_stop_intent = any(
            unit.unit_type == "location_evidence" for unit in evidence_units
        ) or bool(location_hint_count) or bool(retrieval.get("fallback_needed"))

        location_hint_quality = self._location_hint_quality(location_hint_count, has_stop_intent)
        evidence_agreement = self._evidence_agreement(
            matched_stop_count=matched_stop_count,
            stop_confidence=stop_confidence,
            location_hint_count=location_hint_count,
            fallback_needed=bool(retrieval.get("fallback_needed")),
        )
        temporal_bonus = 0.05 if temporal_override else 0.0
        score = (
            (0.45 * max(0.0, route_confidence))
            + (0.30 * max(0.0, stop_confidence))
            + (0.15 * location_hint_quality)
            + (0.10 * evidence_agreement)
            + temporal_bonus
        )
        score = max(0.0, min(1.0, score))

        if not has_stop_intent and route_confidence >= 0.7:
            return RetrievalEvaluationResult(
                state="ACCEPT",
                score=score,
                route_confidence=route_confidence,
                stop_confidence=stop_confidence,
                location_hint_quality=location_hint_quality,
                evidence_agreement=evidence_agreement,
                temporal_bonus=temporal_bonus,
                trigger_reason="route_grounding_strong_no_stop_required",
                location_hint_count=location_hint_count,
                matched_stop_count=matched_stop_count,
            )

        if route_confidence >= 0.7 and (stop_confidence >= 0.62 or not has_stop_intent):
            reason = "route_and_stop_grounding_strong" if has_stop_intent else "route_grounding_strong"
            state = "ACCEPT"
        elif route_confidence >= 0.55:
            reason = "route_grounding_acceptable_stop_grounding_weak"
            state = "AMBIGUOUS"
        else:
            reason = "route_or_stop_grounding_below_threshold"
            state = "CORRECTIVE_FALLBACK"

        if has_stop_intent and matched_stop_count == 0 and location_hint_count > 0:
            state = "CORRECTIVE_FALLBACK"
            reason = "stop_intent_without_grounded_stop"
        elif has_stop_intent and matched_stop_count > 0 and stop_confidence < 0.55 and state == "ACCEPT":
            state = "AMBIGUOUS"
            reason = "grounded_stop_confidence_weak"

        return RetrievalEvaluationResult(
            state=state,
            score=score,
            route_confidence=route_confidence,
            stop_confidence=stop_confidence,
            location_hint_quality=location_hint_quality,
            evidence_agreement=evidence_agreement,
            temporal_bonus=temporal_bonus,
            trigger_reason=reason,
            location_hint_count=location_hint_count,
            matched_stop_count=matched_stop_count,
        )

    @staticmethod
    def _location_hint_quality(location_hint_count: int, has_stop_intent: bool) -> float:
        if not has_stop_intent:
            return 1.0
        if location_hint_count <= 0:
            return 0.2
        return min(1.0, 0.45 + (0.18 * min(location_hint_count, 3)))

    @staticmethod
    def _evidence_agreement(
        matched_stop_count: int,
        stop_confidence: float,
        location_hint_count: int,
        fallback_needed: bool,
    ) -> float:
        if matched_stop_count > 0:
            return min(1.0, 0.5 + (0.12 * min(matched_stop_count, 3)) + (0.2 * max(0.0, stop_confidence - 0.5)))
        if fallback_needed and location_hint_count > 0:
            return 0.2
        return 0.45 if location_hint_count == 0 else 0.3
