from __future__ import annotations

from typing import Any, Optional

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
from .utils import extract_first_json_object


class EnumResolver:
    def __init__(self, ensure_llm, llm_getter):
        self.ensure_llm = ensure_llm
        self.llm_getter = llm_getter

    def resolve(
        self,
        text: str,
        cause_override: Optional[str],
        effect_override: Optional[str],
    ) -> CauseEffectResult:
        # If explicit overrides were provided by the intent parser, use them.
        if cause_override:
            cause = normalize_cause(cause_override)
            cause_conf = 1.0 if cause in CAUSE_ENUMS else 0.0
        else:
            cause = UNKNOWN_CAUSE
            cause_conf = 0.0

        if effect_override:
            effect = normalize_effect(effect_override)
            effect_conf = 1.0 if effect in EFFECT_ENUMS else 0.0
        else:
            effect = UNKNOWN_EFFECT
            effect_conf = 0.0

        # LLM-only enum classification (no rule-based fallback).
        llm_result = self._llm_classify_enums(text)

        if cause == UNKNOWN_CAUSE and llm_result.cause_confidence > cause_conf:
            cause = llm_result.cause
            cause_conf = llm_result.cause_confidence

        if effect == UNKNOWN_EFFECT and llm_result.effect_confidence > effect_conf:
            effect = llm_result.effect
            effect_conf = llm_result.effect_confidence

        return CauseEffectResult(
            cause=cause,
            effect=effect,
            cause_confidence=cause_conf,
            effect_confidence=effect_conf,
        )

    def _llm_classify_enums(self, text: str) -> CauseEffectResult:
        if not self.ensure_llm():
            return CauseEffectResult()

        llm = self.llm_getter()
        if llm is None:
            return CauseEffectResult()

        prompt = (
            "Classify the transit alert into GTFS enums. "
            f"Allowed cause enums: {sorted(CAUSE_ENUMS)}. "
            f"Allowed effect enums: {sorted(EFFECT_ENUMS)}. "
            "Return strict JSON only with keys: cause, effect, cause_confidence, effect_confidence. "
            "Confidence must be 0..1. If uncertain use UNKNOWN_CAUSE/UNKNOWN_EFFECT.\n\n"
            f"Alert text:\n{text}"
        )

        try:
            response = llm.invoke(prompt)
            content = getattr(response, "content", "") if response is not None else ""
            if not isinstance(content, str):
                content = str(content)
            json_obj = extract_first_json_object(content)
            return CauseEffectResult(
                cause=normalize_cause(str(json_obj.get("cause", UNKNOWN_CAUSE))),
                effect=normalize_effect(str(json_obj.get("effect", UNKNOWN_EFFECT))),
                cause_confidence=coerce_confidence(json_obj.get("cause_confidence", 0.0)),
                effect_confidence=coerce_confidence(json_obj.get("effect_confidence", 0.0)),
            )
        except Exception:
            return CauseEffectResult()

