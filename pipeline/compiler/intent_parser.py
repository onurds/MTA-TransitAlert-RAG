from __future__ import annotations

import re
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence

from .confidence import coerce_confidence
from .models import EntityMention, IntentParseResult
from .utils import invoke_json_with_repair


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
        self.last_parse_report: Dict[str, Any] = {
            "source": "deterministic",
            "repair_used": False,
        }

    def parse(self, instruction: str) -> IntentParseResult:
        text = (instruction or "").strip()
        if not text:
            return IntentParseResult(alert_text=None, temporal_text=None)

        if not self.ensure_llm():
            raise RuntimeError("LLM is required for intent parsing but is unavailable.")

        parsed = self._llm_extract_intent(text, compact=False)
        if parsed is None:
            parsed = self._llm_extract_intent(text, compact=True)
        if parsed is not None:
            parsed = self._apply_replacement_route_policy(text, parsed)
            route_ids = self.retriever.link_route_mentions(
                mention.text for mention in parsed.affected_route_mentions
            )
            affected_stop_evidence = " ".join(
                f"{mention.text} {mention.source_span}"
                for mention in parsed.affected_stop_mentions
            ).lower()
            stop_ids = tuple(
                stop_id
                for stop_id in parsed.explicit_stop_ids
                if stop_id.lower() in affected_stop_evidence
            )
            return IntentParseResult(
                alert_text=parsed.alert_text,
                temporal_text=parsed.temporal_text,
                explicit_route_ids=tuple(route_ids),
                explicit_stop_ids=stop_ids,
                location_phrases=tuple(parsed.location_phrases),
                affected_route_mentions=parsed.affected_route_mentions,
                affected_stop_mentions=parsed.affected_stop_mentions,
                alternative_route_mentions=parsed.alternative_route_mentions,
                alternative_stop_mentions=parsed.alternative_stop_mentions,
                corridor_endpoints=parsed.corridor_endpoints,
                effect_hint=parsed.effect_hint,
                cause_hint=parsed.cause_hint,
                style_intent=parsed.style_intent,
                parse_confidence=max(parsed.parse_confidence, 0.7),
                alternative_service_text=parsed.alternative_service_text,
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
                "affected_route_mentions, affected_stop_mentions, alternative_route_mentions, "
                "alternative_stop_mentions, corridor_endpoints, effect_hint, cause_hint, style_intent, parse_confidence, "
                "alternative_service_text.\n"
                "Rules: no prose, no markdown; use null or [] when absent. style_intent='moderate'.\n"
                "Normalize markdown or emphasized text into plain text before extracting fields.\n"
                "Expand route-family prefix lists into full route IDs. Example: "
                "'B: 4, 4A, 6, 41-SBS, 42' means 'B4', 'B4A', 'B6', 'B41-SBS', 'B42'.\n"
                "Treat operator authoring directives as control text, not rider text. Examples: "
                "'timeframe is ...', 'dates will be ...', 'make sure to get it right'. Put the "
                "underlying time phrase in temporal_text when relevant, and do not echo the directive.\n"
                "Do not leave affected_route_mentions empty when an affected route code/name is present. "
                "Classify every service route mention as either affected_route or alternative_route.\n"
                "Each mention item must be {text, source_span, role}. text must be copied from source_span, "
                "and source_span must be copied verbatim from INPUT. Never infer an unmentioned name.\n"
                "affected_stop_mentions and corridor_endpoints must contain only primary affected-service places. "
                "Do NOT include locations from alternative/shuttle/replacement service there.\n"
                "Examples: 'downtown 4 local' and 'Coney Island-bound F' are affected route mentions; "
                "'Take the D instead' is alternative route D.\n"
                "alternative_service_text: the portion of the input describing what riders should do "
                "INSTEAD (shuttles, transfer routes, replacement service). Return null if none.\n"
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
                "affected_route_mentions (array): primary affected route mentions as {text, source_span, role='affected_route'}.\n"
                "affected_stop_mentions (array): primary affected stop/place mentions as {text, source_span, role='affected_stop'}.\n"
                "alternative_route_mentions (array): alternative/replacement route mentions as {text, source_span, role='alternative_route'}.\n"
                "alternative_stop_mentions (array): alternative/replacement stop mentions as {text, source_span, role='alternative_stop'}.\n"
                "corridor_endpoints (array): ordered affected endpoints as {text, source_span, role='corridor_endpoint'}; return at most two.\n"
                "effect_hint (string|null): GTFS effect guess if explicit in text.\n"
                "cause_hint (string|null): GTFS cause guess if explicit in text.\n"
                "style_intent (string): use 'moderate'.\n"
                "parse_confidence (number 0..1).\n"
                "alternative_service_text (string|null): the portion of the input that describes "
                "alternative, replacement, or shuttle service for riders — i.e., what riders should do "
                "INSTEAD of the disrupted service. Examples: shuttle bus routes, transfer instructions, "
                "nearby station recommendations, 'use [route] instead' clauses, 'D/F/N/R trains serve "
                "the same stops'. Return null if none exists.\n"
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
                "- Every mention text must occur inside its source_span, and every source_span must occur verbatim in the operator input.\n"
                "- Copy concise names, not whole sentences. Examples: route='Montauk Branch', stop='Babylon'.\n"
                "- Classify 'No F between Church Av and Coney Island-Stillwell Av. Take the D instead.' as "
                "affected route F, affected stops/corridor endpoints Church Av and Coney Island-Stillwell Av, "
                "and alternative route D.\n"
                "- Classify 'Buses replace trains between Babylon and Montauk' with affected stop mentions "
                "Babylon and Montauk and those same two corridor endpoints; do not invent a branch name.\n"
                "- Exclude alternative/shuttle/replacement locations from affected_stop_mentions and corridor_endpoints.\n\n"
                f"Operator input:\n{instruction}"
            )

        try:
            print(f"[DEBUG] Calling LLM with prompt (compact={compact})...")
            parsed, repair_used = invoke_json_with_repair(
                llm=llm,
                prompt=prompt,
                required_keys=(
                    "alert_text",
                    "temporal_text",
                    "explicit_route_ids",
                    "explicit_stop_ids",
                    "affected_route_mentions",
                    "affected_stop_mentions",
                    "alternative_route_mentions",
                    "alternative_stop_mentions",
                    "corridor_endpoints",
                    "effect_hint",
                    "cause_hint",
                    "style_intent",
                    "parse_confidence",
                    "alternative_service_text",
                ),
                repair_prompt_builder=lambda bad: (
                    "/no_think\n"
                    "Repair the following malformed or incomplete JSON for the transit intent extractor. "
                    "Return strict JSON only with the required keys and use null or [] when missing.\n"
                    f"RAW_OUTPUT:\n{bad}"
                ),
            )
            print(f"[DEBUG] Parsed JSON: {parsed}")
            alert_text = str(parsed.get("alert_text") or "").strip() or None
            temporal_text = str(parsed.get("temporal_text") or "").strip() or None
            if temporal_text and temporal_text.lower() in {"null", "none"}:
                temporal_text = None

            route_ids: List[str] = []
            if isinstance(parsed.get("explicit_route_ids"), list):
                route_ids = self._ground_explicit_tokens(
                    parsed.get("explicit_route_ids"),
                    instruction,
                )
            stop_ids: List[str] = []
            if isinstance(parsed.get("explicit_stop_ids"), list):
                stop_ids = self._ground_explicit_tokens(
                    parsed.get("explicit_stop_ids"),
                    instruction,
                )
            affected_routes = self._ground_mentions(
                parsed.get("affected_route_mentions"),
                instruction,
                expected_role="affected_route",
            )
            affected_stops = self._ground_mentions(
                parsed.get("affected_stop_mentions"),
                instruction,
                expected_role="affected_stop",
            )
            alternative_routes = self._ground_mentions(
                parsed.get("alternative_route_mentions"),
                instruction,
                expected_role="alternative_route",
            )
            alternative_stops = self._ground_mentions(
                parsed.get("alternative_stop_mentions"),
                instruction,
                expected_role="alternative_stop",
            )
            corridor_endpoints = self._ground_mentions(
                parsed.get("corridor_endpoints"),
                instruction,
                expected_role="corridor_endpoint",
            )[:2]
            locations = self._merge_mention_texts(corridor_endpoints, affected_stops)

            effect_hint = str(parsed.get("effect_hint") or "").strip().upper() or None
            cause_hint = str(parsed.get("cause_hint") or "").strip().upper() or None
            style_intent = str(parsed.get("style_intent") or "").strip() or "moderate"
            parse_conf = coerce_confidence(parsed.get("parse_confidence", 0.0))
            alt_service_text = str(parsed.get("alternative_service_text") or "").strip() or None
            if alt_service_text and alt_service_text.lower() in {"null", "none"}:
                alt_service_text = None

            if not alert_text and not temporal_text and not route_ids and not stop_ids and not locations and not affected_routes:
                return None

            self.last_parse_report = {
                "source": "llm",
                "repair_used": repair_used,
            }
            return IntentParseResult(
                alert_text=alert_text,
                temporal_text=temporal_text,
                explicit_route_ids=tuple(route_ids),
                explicit_stop_ids=tuple(stop_ids),
                location_phrases=tuple(locations),
                affected_route_mentions=tuple(affected_routes),
                affected_stop_mentions=tuple(affected_stops),
                alternative_route_mentions=tuple(alternative_routes),
                alternative_stop_mentions=tuple(alternative_stops),
                corridor_endpoints=tuple(corridor_endpoints),
                effect_hint=effect_hint,
                cause_hint=cause_hint,
                style_intent=style_intent,
                parse_confidence=parse_conf,
                alternative_service_text=alt_service_text,
            )
        except Exception as e:
            print(f"[DEBUG] Error in _llm_extract_intent: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def _ground_mentions(
        raw_mentions: Any,
        instruction: str,
        expected_role: str,
    ) -> List[EntityMention]:
        if not isinstance(raw_mentions, list):
            return []
        normalized_source = IntentParser._normalize_evidence_text(instruction)
        out: List[EntityMention] = []
        seen = set()
        for raw in raw_mentions:
            if not isinstance(raw, dict):
                continue
            text = " ".join(str(raw.get("text") or "").split()).strip()
            source_span = str(raw.get("source_span") or "").strip()
            role = str(raw.get("role") or expected_role).strip().lower()
            if not text or not source_span or role != expected_role:
                continue
            normalized_span = IntentParser._normalize_evidence_text(source_span)
            normalized_text = IntentParser._normalize_evidence_text(text)
            if not normalized_span or normalized_span not in normalized_source:
                continue
            if not normalized_text or normalized_text not in normalized_span:
                continue
            key = (text.lower(), source_span.lower(), role)
            if key in seen:
                continue
            seen.add(key)
            out.append(EntityMention(text=text, source_span=source_span, role=role))
        return out

    @staticmethod
    def _normalize_evidence_text(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
        return " ".join(normalized.split())

    @staticmethod
    def _merge_mention_texts(*groups: Sequence[EntityMention]) -> List[str]:
        out: List[str] = []
        seen = set()
        for group in groups:
            for mention in group:
                key = mention.text.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(mention.text)
        return out

    @staticmethod
    def _apply_replacement_route_policy(
        instruction: str,
        parsed: IntentParseResult,
    ) -> IntentParseResult:
        match = re.search(
            r"\b([A-Z0-9][A-Z0-9+-]*(?:-SBS)?)\s+replaces\s+([A-Z0-9][A-Z0-9+-]*(?:-SBS)?)\s+service\b",
            instruction,
            flags=re.IGNORECASE,
        )
        if not match:
            return parsed

        replacement = match.group(1)
        original = match.group(2)
        replacement_mention = EntityMention(
            text=replacement,
            source_span=replacement,
            role="affected_route",
        )
        original_mention = EntityMention(
            text=original,
            source_span=original,
            role="alternative_route",
        )
        affected = (replacement_mention,) + tuple(
            mention
            for mention in parsed.affected_route_mentions
            if mention.text.strip().lower() != original.lower()
        )
        alternatives = (original_mention,) + tuple(
            mention
            for mention in parsed.alternative_route_mentions
            if mention.text.strip().lower() != replacement.lower()
        )
        return IntentParseResult(
            alert_text=parsed.alert_text,
            temporal_text=parsed.temporal_text,
            explicit_route_ids=parsed.explicit_route_ids,
            explicit_stop_ids=parsed.explicit_stop_ids,
            location_phrases=parsed.location_phrases,
            affected_route_mentions=affected,
            affected_stop_mentions=parsed.affected_stop_mentions,
            alternative_route_mentions=alternatives,
            alternative_stop_mentions=parsed.alternative_stop_mentions,
            corridor_endpoints=parsed.corridor_endpoints,
            effect_hint=parsed.effect_hint,
            cause_hint=parsed.cause_hint,
            style_intent=parsed.style_intent,
            parse_confidence=parsed.parse_confidence,
            alternative_service_text=parsed.alternative_service_text,
        )

    @staticmethod
    def _ground_explicit_tokens(raw_tokens: Any, instruction: str) -> List[str]:
        if not isinstance(raw_tokens, list):
            return []
        source_lower = instruction.lower()
        out: List[str] = []
        seen = set()
        for raw in raw_tokens:
            token = str(raw or "").strip()
            key = token.lower()
            if not token or key in seen or key not in source_lower:
                continue
            seen.add(key)
            out.append(token)
        return out

    def _extract_explicit_stop_ids(self, text: str) -> List[str]:
        ids: List[str] = []
        for m in re.finditer(r"\bstop(?:\s*id)?\s*[:#]?\s*([A-Za-z0-9]{3,10}[NS]?)\b", text, flags=re.IGNORECASE):
            ids.append(m.group(1).strip().upper())
        return self._merge_unique_tokens(ids)

    def _extract_explicit_route_ids(self, text: str) -> List[str]:
        tokens = self.retriever._route_tokens_from_text(text)
        return self._merge_unique_tokens(tokens)

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
