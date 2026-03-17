from __future__ import annotations

import json
import re
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence

from .confidence import coerce_confidence
from .models import IntentParseResult
from .utils import extract_llm_text_content


class IntentParser:
    def __init__(
        self,
        retriever: Any,
        ensure_llm: Callable[[], bool],
        llm_getter: Callable[[], Any],
        bump_telemetry: Callable[[str, int], None],
    ):
        self.retriever = retriever
        self.ensure_llm = ensure_llm
        self.llm_getter = llm_getter
        self.bump_telemetry = bump_telemetry

    def parse(self, instruction: str) -> IntentParseResult:
        text = (instruction or "").strip()
        if not text:
            return IntentParseResult(alert_text=None, temporal_text=None)

        heuristic_routes = self._extract_explicit_route_ids(text)
        heuristic_stops = self._extract_explicit_stop_ids(text)

        if not self.ensure_llm():
            raise RuntimeError("LLM is required for intent parsing but is unavailable.")

        parsed = self._llm_extract_intent(text, compact=False)
        if parsed is None:
            parsed = self._llm_extract_intent(text, compact=True)
        if parsed is not None:
            route_ids = self._merge_unique_tokens(parsed.explicit_route_ids, heuristic_routes)
            stop_ids = self._merge_unique_tokens(parsed.explicit_stop_ids, heuristic_stops)
            return IntentParseResult(
                alert_text=parsed.alert_text,
                temporal_text=parsed.temporal_text,
                explicit_route_ids=tuple(route_ids),
                explicit_stop_ids=tuple(stop_ids),
                location_phrases=tuple(parsed.location_phrases),
                effect_hint=parsed.effect_hint,
                cause_hint=parsed.cause_hint,
                style_intent=parsed.style_intent,
                parse_confidence=max(parsed.parse_confidence, 0.7),
            )

        self.bump_telemetry("parse_fail", 1)
        raise RuntimeError(
            "LLM intent extraction failed after two attempts. "
            "Cannot compile without structured intent."
        )

    def _llm_extract_intent(self, instruction: str, compact: bool = False) -> Optional[IntentParseResult]:
        llm = self.llm_getter()
        if llm is None:
            return None

        if compact:
            prompt = (
                "/no_think\n"
                "Extract transit intent from operator text. Return strict JSON only with keys:\n"
                "alert_text, temporal_text, explicit_route_ids, explicit_stop_ids, "
                "location_phrases, effect_hint, cause_hint, style_intent, parse_confidence.\n"
                "Rules: no prose, no markdown; use null or [] when absent. style_intent='moderate'.\n"
                "Normalize markdown or emphasized text into plain text before extracting fields.\n"
                "Expand route-family prefix lists into full route IDs. Example: "
                "'B: 4, 4A, 6, 41-SBS, 42' means 'B4', 'B4A', 'B6', 'B41-SBS', 'B42'.\n"
                "Treat operator authoring directives as control text, not rider text. Examples: "
                "'timeframe is ...', 'dates will be ...', 'make sure to get it right'. Put the "
                "underlying time phrase in temporal_text when relevant, and do not echo the directive.\n"
                "Return clean location_phrases only: concise physical place phrases, no leading "
                "prepositions, no route names unless they are physical places, and no trailing "
                "service/meta clauses such as 'no stops will be missed'.\n"
                f"INPUT:\n{instruction}"
            )
        else:
            prompt = (
                "/no_think\n"
                "You are an MTA alert intent extractor. "
                "Operators write fully free-form text; extract structured intent.\n"
                "Return strict JSON only with keys:\n"
                "alert_text (string|null): optional header hint only, inferred from the input as the main rider-facing summary.\n"
                "temporal_text (string|null): raw time/recurrence phrase(s).\n"
                "explicit_route_ids (array of strings): route IDs explicitly present in input.\n"
                "explicit_stop_ids (array of strings): stop IDs explicitly present in input.\n"
                "location_phrases (array of strings): clean place phrases that can help stop grounding.\n"
                "effect_hint (string|null): GTFS effect guess if explicit in text.\n"
                "cause_hint (string|null): GTFS cause guess if explicit in text.\n"
                "style_intent (string): use 'moderate'.\n"
                "parse_confidence (number 0..1).\n"
                "Never invent IDs not present in input. Use [] or null when missing.\n\n"
                "Normalization rules:\n"
                "- Convert markdown or emphasized text to plain text.\n"
                "- Infer a possible header-level summary hint only; do not try to fully rewrite the alert body here.\n"
                "- If a route-family prefix applies to a list, expand it in explicit_route_ids. Example: "
                "'B: 4, 4A, 6, 41-SBS, 42' means route IDs like 'B4', 'B4A', 'B6', 'B41-SBS', 'B42'.\n"
                "- If the operator uses cues like 'header:' or 'description:', treat them as strong hints for the split, but do not require them and do not copy the labels.\n"
                "- alert_text should already exclude scheduling/meta/UI clauses and description-only rider detail.\n"
                "- Do not copy operator authoring directives into alert_text or location_phrases. "
                "Examples: 'timeframe is ...', 'dates will be ...', 'make sure to get it right'. "
                "Use their actual time content only in temporal_text when relevant.\n"
                "- location_phrases must be concise physical locations only, with no leading "
                "prepositions like 'at', 'on', or 'near', and no trailing rider-guidance clauses.\n"
                "- Exclude alternative-stop recommendations from location_phrases unless they are "
                "part of the affected location itself.\n\n"
                f"Operator input:\n{instruction}"
            )

        try:
            print(f"[DEBUG] Calling LLM with prompt (compact={compact})...")
            response = llm.invoke(prompt)
            print(f"[DEBUG] LLM response received: {type(response)}")
            content = extract_llm_text_content(response)
            print(f"[DEBUG] Extracted content: {content[:100]}...")
            parsed = self._extract_first_json_object(content)
            print(f"[DEBUG] Parsed JSON: {parsed}")
            alert_text = str(parsed.get("alert_text") or "").strip() or None
            temporal_text = str(parsed.get("temporal_text") or "").strip() or None
            if temporal_text and temporal_text.lower() in {"null", "none"}:
                temporal_text = None

            route_ids: List[str] = []
            if isinstance(parsed.get("explicit_route_ids"), list):
                route_ids = [str(x).strip() for x in parsed.get("explicit_route_ids", []) if str(x).strip()]
            stop_ids: List[str] = []
            if isinstance(parsed.get("explicit_stop_ids"), list):
                stop_ids = [str(x).strip() for x in parsed.get("explicit_stop_ids", []) if str(x).strip()]
            locations: List[str] = []
            if isinstance(parsed.get("location_phrases"), list):
                locations = [str(x).strip() for x in parsed.get("location_phrases", []) if str(x).strip()]

            effect_hint = str(parsed.get("effect_hint") or "").strip().upper() or None
            cause_hint = str(parsed.get("cause_hint") or "").strip().upper() or None
            style_intent = str(parsed.get("style_intent") or "").strip() or "moderate"
            parse_conf = coerce_confidence(parsed.get("parse_confidence", 0.0))

            if not alert_text and not temporal_text and not route_ids and not stop_ids and not locations:
                return None

            return IntentParseResult(
                alert_text=alert_text,
                temporal_text=temporal_text,
                explicit_route_ids=tuple(route_ids),
                explicit_stop_ids=tuple(stop_ids),
                location_phrases=tuple(locations),
                effect_hint=effect_hint,
                cause_hint=cause_hint,
                style_intent=style_intent,
                parse_confidence=parse_conf,
            )
        except Exception as e:
            print(f"[DEBUG] Error in _llm_extract_intent: {e}")
            traceback.print_exc()
            return None

    def _extract_explicit_stop_ids(self, text: str) -> List[str]:
        ids: List[str] = []
        for m in re.finditer(r"\bstop(?:\s*id)?\s*[:#]?\s*([A-Za-z0-9]{3,10}[NS]?)\b", text, flags=re.IGNORECASE):
            ids.append(m.group(1).strip().upper())
        return self._merge_unique_tokens(ids)

    def _extract_explicit_route_ids(self, text: str) -> List[str]:
        tokens = self.retriever._route_tokens_from_text(text)
        return self._merge_unique_tokens(tokens)

    @staticmethod
    def _extract_first_json_object(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            pass

        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    @staticmethod
    def _merge_unique_tokens(*groups: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for group in groups:
            for value in group or []:
                token = str(value or "").strip().upper()
                if not token or token in seen:
                    continue
                seen.add(token)
                out.append(token)
        return out
