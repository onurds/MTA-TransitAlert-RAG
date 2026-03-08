from __future__ import annotations

from typing import Any, Optional, Tuple

from pipeline.gtfs_rules import (
    CAUSE_ENUMS,
    EFFECT_ENUMS,
    UNKNOWN_CAUSE,
    UNKNOWN_EFFECT,
    normalize_cause,
    normalize_effect,
)

from .confidence import coerce_confidence
from .models import CauseEffectResult
from .utils import extract_first_json_object, extract_llm_text_content


class EnumResolver:
    def __init__(self, ensure_llm, llm_getter, min_confidence: float = 0.6):
        self.ensure_llm = ensure_llm
        self.llm_getter = llm_getter
        self.min_confidence = coerce_confidence(min_confidence)

    def resolve(
        self,
        text: str,
        cause_override: Optional[str],
        effect_override: Optional[str],
    ) -> CauseEffectResult:
        # Parser overrides are treated as hints. Invalid hints do not force UNKNOWN.
        cause = ""
        effect = ""
        cause_conf = 0.0
        effect_conf = 0.0

        if cause_override:
            hint = normalize_cause(cause_override)
            if hint != UNKNOWN_CAUSE:
                cause = hint
                cause_conf = self.min_confidence

        if effect_override:
            hint = normalize_effect(effect_override)
            if hint != UNKNOWN_EFFECT:
                effect = hint
                effect_conf = self.min_confidence

        llm_result, llm_has_values = self._llm_classify_enums(text)

        # Use LLM enums when present and sufficiently confident.
        if llm_result.cause and llm_result.cause_confidence >= self.min_confidence:
            cause = llm_result.cause
            cause_conf = llm_result.cause_confidence

        if llm_result.effect and llm_result.effect_confidence >= self.min_confidence:
            effect = llm_result.effect
            effect_conf = llm_result.effect_confidence

        # Unknown fallback only when enum fields are empty after resolution.
        if not cause:
            cause = UNKNOWN_CAUSE
            cause_conf = 0.0 if not llm_has_values else cause_conf

        if not effect:
            effect = UNKNOWN_EFFECT
            effect_conf = 0.0 if not llm_has_values else effect_conf

        return CauseEffectResult(
            cause=cause,
            effect=effect,
            cause_confidence=cause_conf,
            effect_confidence=effect_conf,
        )

    def _llm_classify_enums(self, text: str) -> Tuple[CauseEffectResult, bool]:
        if not self.ensure_llm():
            return CauseEffectResult(cause="", effect="", cause_confidence=0.0, effect_confidence=0.0), False

        llm = self.llm_getter()
        if llm is None:
            return CauseEffectResult(cause="", effect="", cause_confidence=0.0, effect_confidence=0.0), False

        prompt = (
            "/no_think\n"
            "Classify the transit alert into GTFS enums. "
            f"Allowed cause enums: {sorted(CAUSE_ENUMS)}. "
            f"Allowed effect enums: {sorted(EFFECT_ENUMS)}. "
            "Return strict JSON only with keys: cause, effect, cause_confidence, effect_confidence. "
            "Confidence must be 0..1. If uncertain use UNKNOWN_CAUSE/UNKNOWN_EFFECT.\n\n"
            f"Alert text:\n{text}"
        )

        try:
            response = llm.invoke(prompt)
            content = extract_llm_text_content(response)
            json_obj = extract_first_json_object(content)
            raw_cause = str(json_obj.get("cause", "") or "").strip()
            raw_effect = str(json_obj.get("effect", "") or "").strip()
            has_values = bool(raw_cause or raw_effect)
            return CauseEffectResult(
                cause=self._coerce_cause(raw_cause),
                effect=self._coerce_effect(raw_effect),
                cause_confidence=coerce_confidence(json_obj.get("cause_confidence", 0.0)),
                effect_confidence=coerce_confidence(json_obj.get("effect_confidence", 0.0)),
            ), has_values
        except Exception:
            return CauseEffectResult(cause="", effect="", cause_confidence=0.0, effect_confidence=0.0), False

    @staticmethod
    def _coerce_cause(raw_value: str) -> str:
        token = str(raw_value or "").strip().upper()
        if not token:
            return ""
        if token in CAUSE_ENUMS:
            return token
        return "OTHER_CAUSE"

    @staticmethod
    def _coerce_effect(raw_value: str) -> str:
        token = str(raw_value or "").strip().upper()
        if not token:
            return ""
        if token in EFFECT_ENUMS:
            return token
        return "OTHER_EFFECT"
