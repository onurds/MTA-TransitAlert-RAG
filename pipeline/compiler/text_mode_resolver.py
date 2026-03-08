from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from .utils import extract_llm_text_content


TextMode = Literal["default", "rewrite"]


class TextModeResolver:
    def __init__(self, bump_telemetry: Optional[Callable[[str, int], None]] = None):
        self._bump_telemetry = bump_telemetry

    def resolve(
        self,
        llm: Any,
        instruction: str,
        route_ids: Sequence[str],
        cause: str,
        effect: str,
        text_mode: TextMode,
        header_hint: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        source_text = self._normalize_blocks(instruction)
        if not source_text:
            return "", None
        effective_header_hint = header_hint if text_mode == "rewrite" else None
        rider_source = source_text
        if llm is not None:
            rider_source = self._llm_extract_rider_source(
                llm=llm,
                source_text=source_text,
                text_mode=text_mode,
            )

        if llm is not None:
            for attempt in ("base", "repair"):
                if attempt == "repair":
                    self._bump("text_layout_retry")
                prompt = self._build_prompt(
                    operator_input=source_text,
                    rider_source=rider_source,
                    route_ids=route_ids,
                    cause=cause,
                    effect=effect,
                    text_mode=text_mode,
                    header_hint=effective_header_hint,
                    attempt=attempt,
                )
                try:
                    resp = llm.invoke(prompt)
                    content = extract_llm_text_content(resp)
                    parsed = self._extract_first_json_object(content)
                    header = self._normalize_blocks(parsed.get("header_text") or parsed.get("header") or "")
                    description_raw = parsed.get("description_text")
                    if description_raw is None:
                        description_raw = parsed.get("description")
                    description = self._normalize_blocks(description_raw or "") or None
                    header, description = self._llm_cleanup_operator_commands(
                        llm=llm,
                        source_text=source_text,
                        header=header,
                        description=description,
                        text_mode=text_mode,
                    )
                    ok, reason = self._validate_layout(
                        header=header,
                        description=description,
                        source_text=rider_source,
                        route_ids=route_ids,
                        text_mode=text_mode,
                    )
                    if ok:
                        return header, description
                    self._bump("text_layout_validation_fail")
                    if reason:
                        self._bump(f"text_layout_validation_fail_{reason}")
                except Exception:
                    continue

        self._bump("text_layout_fallback_used")
        return self._deterministic_fallback(
            source_text=rider_source,
            header_hint=effective_header_hint,
            text_mode=text_mode,
        )

    def _build_prompt(
        self,
        operator_input: str,
        rider_source: str,
        route_ids: Sequence[str],
        cause: str,
        effect: str,
        text_mode: TextMode,
        header_hint: Optional[str],
        attempt: str,
    ) -> str:
        mode_instructions = self._default_mode_rules() if text_mode == "default" else self._rewrite_mode_rules()
        repair_block = ""
        if attempt == "repair":
            repair_block = (
                "Repair requirements:\n"
                "- The previous attempt was invalid because content was dropped, commands remained, or header/description split was malformed.\n"
                "- Preserve all non-command rider-facing information.\n"
                "- Never split a source sentence across header_text and description_text.\n"
                "- If a sentence contains both a summary and rider instruction, keep the whole sentence together.\n\n"
            )
        hint_block = f"HEADER_HINT: {header_hint}\n" if header_hint else "HEADER_HINT: null\n"
        return (
            "/no_think\n"
            "You are laying out transit alert text into header_text and description_text.\n"
            "Return strict JSON only with keys: header_text, description_text, confidence.\n"
            "Rules:\n"
            "- Preserve all facts.\n"
            "- Do NOT invent stops, routes, times, causes, effects, or rider guidance.\n"
            "- Use only RIDER_SOURCE as rider-facing content. Treat OPERATOR_INPUT as context.\n"
            "- Treat cues like 'header:' and 'description:' as strong hints, but do not copy those labels into output.\n"
            "- Keep route tokens unchanged when present.\n"
            "- description_text may be null only if there is no rider-facing content beyond the header.\n"
            f"{mode_instructions}"
            f"{repair_block}"
            f"{hint_block}"
            f"OPERATOR_INPUT:\n{operator_input}\n"
            f"RIDER_SOURCE:\n{rider_source}\n"
            f"ROUTES: {list(route_ids)}\n"
            f"CAUSE: {cause}\n"
            f"EFFECT: {effect}\n"
        )

    @staticmethod
    def _default_mode_rules() -> str:
        return (
            "- Mode is DEFAULT.\n"
            "- Preserve wording, order, and sentence boundaries as closely as possible.\n"
            "- Do not rewrite for style.\n"
            "- If you use the first rider-facing sentence as header_text, keep that sentence whole.\n"
            "- Prefer natural whole-sentence or whole-section splits over summary-style shortening.\n"
            "- Split header_text and description_text only at completed sentence boundaries or explicit labeled sections.\n"
            "- Do not split a single sentence across header_text and description_text.\n"
            "- If a sentence contains an actionable rider instruction after punctuation like ';' or ',', keep that whole sentence together.\n"
            "- Remove authoring commands semantically even when they are phrased indirectly, such as 'make the timeframe from now until 5 hours ahead'.\n"
            "- Keep section labels in place, including 'See a map of this stop change.', 'What's happening?', 'NOTE:', and missed-stop labels, unless they are operator/control commands.\n"
            "- header_text should be a primary rider-facing summary in the source's own wording, not a rewritten abstraction.\n"
            "- description_text should contain the remaining rider-facing details in original order.\n"
        )

    @staticmethod
    def _rewrite_mode_rules() -> str:
        return (
            "- Mode is REWRITE.\n"
            "- Rewrite into transit-alert style while preserving all facts.\n"
            "- You may rephrase and reorder for readability, but do not drop rider-facing information.\n"
            "- Do not split a single sentence across header_text and description_text.\n"
            "- Remove operator/control commands semantically rather than echoing them.\n"
            "- header_text should be the primary summary.\n"
            "- description_text should contain all remaining rider-facing details.\n"
        )

    def _llm_cleanup_operator_commands(
        self,
        llm: Any,
        source_text: str,
        header: str,
        description: Optional[str],
        text_mode: TextMode,
    ) -> Tuple[str, Optional[str]]:
        prompt = (
            "/no_think\n"
            "Remove only operator/control commands from this already-generated transit alert text.\n"
            "Return strict JSON only with keys: header_text, description_text.\n"
            "Rules:\n"
            "- Preserve rider-facing text, wording, order, and sentence boundaries as closely as possible.\n"
            "- Remove internal drafting or authoring instructions such as timeframe/date-setting instructions, "
            "header/description instructions, or quality-control instructions.\n"
            "- Example commands to remove: 'timeframe is ...', 'make the timeframe from now until 5 hours ahead', "
            "'dates will be ...', 'set the dates ...', 'use this as the header', 'make sure to get it right'.\n"
            "- Do not remove legitimate rider-facing MTA text like detour notes, map callouts, or 'What's happening?'.\n"
            "- Do not add new facts or rewrite for style.\n"
            f"MODE: {text_mode}\n"
            f"SOURCE_FACTS:\n{source_text}\n"
            f"CURRENT_HEADER:\n{header}\n"
            f"CURRENT_DESCRIPTION:\n{description or 'null'}\n"
        )
        try:
            resp = llm.invoke(prompt)
            content = extract_llm_text_content(resp)
            parsed = self._extract_first_json_object(content)
            cleaned_header = self._normalize_blocks(parsed.get("header_text") or header)
            cleaned_description_raw = parsed.get("description_text")
            if cleaned_description_raw is None:
                cleaned_description = description
            else:
                cleaned_description = self._normalize_blocks(cleaned_description_raw or "") or None
            return cleaned_header or header, cleaned_description
        except Exception:
            return header, description

    def _llm_extract_rider_source(
        self,
        llm: Any,
        source_text: str,
        text_mode: TextMode,
    ) -> str:
        prompt = (
            "/no_think\n"
            "Remove only operator/control commands from this operator-written transit alert input.\n"
            "Return strict JSON only with key: rider_text.\n"
            "Rules:\n"
            "- Preserve all rider-facing text, wording, order, and sentence boundaries as closely as possible.\n"
            "- Remove internal drafting or authoring instructions such as timeframe/date-setting instructions, "
            "header/description instructions, or quality-control instructions.\n"
            "- Examples of commands to remove: 'timeframe is ...', 'make the timeframe from now until 5 hours ahead', "
            "'it should start from now until 5 hours ahead', 'dates will be ...', 'set the dates ...', "
            "'use this as the header', 'make sure to get it right'.\n"
            "- Keep legitimate rider-facing MTA text like map callouts, notes, operational explanations, and 'What's happening?'.\n"
            "- Do not rewrite for style.\n"
            f"MODE: {text_mode}\n"
            f"OPERATOR_INPUT:\n{source_text}\n"
        )
        try:
            resp = llm.invoke(prompt)
            content = extract_llm_text_content(resp)
            parsed = self._extract_first_json_object(content)
            rider_text = self._normalize_blocks(parsed.get("rider_text") or "")
            return rider_text or source_text
        except Exception:
            return source_text

    def _validate_layout(
        self,
        header: str,
        description: Optional[str],
        source_text: str,
        route_ids: Sequence[str],
        text_mode: TextMode,
    ) -> Tuple[bool, Optional[str]]:
        if not header:
            return False, "empty_header"
        if not self._route_bullets_are_safe(header, route_ids):
            return False, "route_bullet"
        if description and not self._route_bullets_are_safe(description, route_ids):
            return False, "route_bullet"

        rider_source = self._normalize_blocks(source_text)
        if not rider_source:
            return True, None
        combined = self._normalize_blocks("\n".join([header, description or ""]))
        if not combined:
            return False, "empty_output"
        if description and self._starts_with_mid_sentence_fragment(description, rider_source):
            return False, "split_unit"
        if not self._covers_required_source_content(
            header=header,
            description=description or "",
            sanitized_source=rider_source,
            text_mode=text_mode,
        ):
            return False, "coverage"
        if text_mode == "default" and self._drops_first_sentence_content(
            header=header,
            description=description or "",
            sanitized_source=rider_source,
        ):
            return False, "first_sentence_loss"
        return True, None

    def _covers_required_source_content(
        self,
        header: str,
        description: str,
        sanitized_source: str,
        text_mode: TextMode,
    ) -> bool:
        combined_tokens = set(self._tokenize(" ".join([header, description])))
        min_overlap = 0.75 if text_mode == "default" else 0.55
        for unit in self._source_units_for_coverage(sanitized_source):
            unit_tokens = set(self._tokenize(unit))
            if len(unit_tokens) < 4:
                continue
            overlap = len(unit_tokens & combined_tokens) / max(1, len(unit_tokens))
            if overlap < min_overlap:
                return False
        return True

    def _drops_first_sentence_content(
        self,
        header: str,
        description: str,
        sanitized_source: str,
    ) -> bool:
        first_sentence = self._first_sentence(sanitized_source)
        if not first_sentence:
            return False
        first_tokens = set(self._tokenize(first_sentence))
        if len(first_tokens) < 4:
            return False
        combined_tokens = set(self._tokenize(" ".join([header, description])))
        overlap = len(first_tokens & combined_tokens) / max(1, len(first_tokens))
        if overlap < 0.75:
            return True
        header_norm = self._canonical_unit_text(header)
        first_norm = self._canonical_unit_text(first_sentence)
        return bool(header_norm and first_norm.startswith(header_norm) and header_norm != first_norm and not description)

    @staticmethod
    def _source_units_for_coverage(source_text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", source_text or "").strip()
        if not normalized:
            return []
        parts = re.split(
            r"(?<=[.!?])\s+|"
            r"(?<=[.!?])(?=[A-Z])|"
            r"\n+",
            normalized,
            flags=re.IGNORECASE,
        )
        return [part.strip(" ,.;:-") for part in parts if part and part.strip(" ,.;:-")]

    def _default_structural_units(self, source_text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", source_text or "").strip()
        if not normalized:
            return []
        parts = re.split(r"(?<=[.!?])\s+|(?<=[.!?])(?=[A-Z])|\n+", normalized)
        units = [part.strip() for part in parts if part and part.strip()]
        return units

    def _starts_with_mid_sentence_fragment(self, description: str, source_text: str) -> bool:
        desc = (description or "").strip()
        if not desc or not source_text:
            return False
        leading = self._leading_words(desc, max_words=8)
        if len(leading.split()) < 2:
            return False
        pattern = r"\b" + r"\W+".join(re.escape(word) for word in leading.split()) + r"\b"
        match = re.search(pattern, source_text, flags=re.IGNORECASE)
        if not match:
            return False
        prefix = source_text[:match.start()].rstrip()
        if not prefix:
            return False
        return prefix[-1] not in ".!?\n"

    @staticmethod
    def _route_bullets_are_safe(text: str, route_ids: Sequence[str]) -> bool:
        allowed = {str(r).upper() for r in route_ids if r}
        if not allowed:
            return True
        bullets = re.findall(r"\[([A-Za-z0-9+\-]+)\]", text or "")
        for bullet in bullets:
            token = bullet.strip().upper().replace("-SBS", "+")
            if token not in allowed:
                return False
        return True

    @staticmethod
    def _extract_first_json_object(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}

    @staticmethod
    def _normalize_blocks(text: Any) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        blocks = [block for block in re.split(r"(?:\r?\n){2,}", value) if block.strip()]
        out: List[str] = []
        for block in blocks:
            lines = [" ".join(line.split()) for line in block.splitlines() if line.strip()]
            if lines:
                out.append("\n".join(lines))
        if not out:
            return " ".join(value.split())
        return "\n\n".join(out)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
        return [token for token in cleaned.split() if token]

    @staticmethod
    def _leading_words(text: str, max_words: int = 8) -> str:
        words = re.findall(r"[A-Za-z0-9']+", text or "")
        return " ".join(words[:max_words])

    @staticmethod
    def _canonical_unit_text(text: str) -> str:
        value = re.sub(r"\s+", " ", str(text or "")).strip()
        if not value:
            return ""
        value = re.sub(r"([.!?])(?=\s|$)", "", value)
        return value.strip()

    @staticmethod
    def _first_sentence(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return ""
        parts = re.split(r"(?<=[.!?])\s+|(?<=[.!?])(?=[A-Z])|\n+", normalized, maxsplit=1)
        return parts[0].strip() if parts and parts[0].strip() else ""

    def _bump(self, key: str, amount: int = 1) -> None:
        if self._bump_telemetry is not None:
            self._bump_telemetry(key, amount)

    def _deterministic_fallback(
        self,
        source_text: str,
        header_hint: Optional[str],
        text_mode: TextMode,
    ) -> Tuple[str, Optional[str]]:
        cleaned_source = self._normalize_blocks(source_text)
        if not cleaned_source:
            return "", None

        if text_mode == "rewrite":
            if header_hint:
                header = self._normalize_blocks(header_hint).rstrip(".")
                if header:
                    return header, cleaned_source if cleaned_source != header else None
            first_sentence = re.split(r"(?<=[.!?])\s+", cleaned_source, maxsplit=1)[0].strip()
            remainder = cleaned_source[len(first_sentence):].strip()
            return first_sentence, remainder or None

        units = self._default_structural_units(cleaned_source)
        if not units:
            return cleaned_source, None
        header = units[0].strip()
        description = " ".join(unit.strip() for unit in units[1:] if unit and unit.strip()).strip()
        return header, description or None
