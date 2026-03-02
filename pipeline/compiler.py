from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, model_validator

from pipeline.description_generator import DescriptionGenerator
from pipeline.graph_retriever import GraphRetriever
from pipeline.gtfs_rules import (
    CAUSE_ENUMS,
    EFFECT_ENUMS,
    UNKNOWN_CAUSE,
    UNKNOWN_EFFECT,
    infer_cause_effect_rule_first,
    normalize_cause,
    normalize_effect,
)
from pipeline.llm_config import build_langchain_chat_model, load_llm_config, with_overrides
from pipeline.temporal_resolver import TemporalResolver


CONFIDENCE_THRESHOLD = 0.85
MULTI_LANG_CODES = ("zh", "es")


class InformedEntity(BaseModel):
    agency_id: str
    route_id: Optional[str] = None
    stop_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_any_selector(self) -> "InformedEntity":
        self.agency_id = (self.agency_id or "").strip() or "MTA NYCT"
        if self.route_id:
            self.route_id = str(self.route_id).strip().upper()
        if self.stop_id:
            self.stop_id = str(self.stop_id).strip().upper()
        if not self.route_id and not self.stop_id:
            raise ValueError("informed_entity requires at least route_id or stop_id")
        return self


class ActivePeriod(BaseModel):
    start: str
    end: Optional[str] = None


class Translation(BaseModel):
    text: str
    language: Optional[str] = None


class TranslatedString(BaseModel):
    translation: List[Translation] = Field(default_factory=list)


class LegacyAlertPayload(BaseModel):
    id: str = ""
    header: str = ""
    description: Optional[str] = None
    effect: str = UNKNOWN_EFFECT
    cause: str = UNKNOWN_CAUSE
    active_periods: List[ActivePeriod] = Field(default_factory=list)
    informed_entities: List[InformedEntity] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_gtfs_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        out = dict(data)

        # Accept GTFS-style request alert payloads.
        if "active_period" in out and "active_periods" not in out:
            out["active_periods"] = out.get("active_period")
        if "informed_entity" in out and "informed_entities" not in out:
            out["informed_entities"] = out.get("informed_entity")

        if not out.get("header") and isinstance(out.get("header_text"), dict):
            tr = out["header_text"].get("translation") or []
            if isinstance(tr, list):
                for item in tr:
                    if isinstance(item, dict) and item.get("text"):
                        out["header"] = str(item.get("text"))
                        if str(item.get("language", "")).lower() == "en":
                            break

        if not out.get("description") and isinstance(out.get("description_text"), dict):
            tr = out["description_text"].get("translation") or []
            if isinstance(tr, list):
                for item in tr:
                    if isinstance(item, dict) and item.get("text"):
                        out["description"] = str(item.get("text"))
                        if str(item.get("language", "")).lower() == "en":
                            break

        return out


class CompileRequest(BaseModel):
    instruction: str = ""
    alert: Optional[LegacyAlertPayload] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


@dataclass(frozen=True)
class ParsedInstruction:
    header: Optional[str]
    description: Optional[str]
    dates_text: Optional[str]
    cause_override: Optional[str]
    effect_override: Optional[str]


@dataclass(frozen=True)
class IntentParseResult:
    alert_text: Optional[str]
    temporal_text: Optional[str]
    explicit_route_ids: Tuple[str, ...] = field(default_factory=tuple)
    explicit_stop_ids: Tuple[str, ...] = field(default_factory=tuple)
    location_phrases: Tuple[str, ...] = field(default_factory=tuple)
    effect_hint: Optional[str] = None
    cause_hint: Optional[str] = None
    style_intent: Optional[str] = None
    parse_confidence: float = 0.0


class AlertCompiler:
    def __init__(
        self,
        graph_path: str = "data/mta_knowledge_graph.gpickle",
        calendar_path: str = "data/2026_english_calendar.csv",
        timezone: str = "America/New_York",
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.retriever = GraphRetriever(graph_path=graph_path)
        self.temporal_resolver: Optional[TemporalResolver]
        self.tz = ZoneInfo(timezone)
        self.confidence_threshold = confidence_threshold

        try:
            self.temporal_resolver = TemporalResolver(calendar_path=calendar_path, timezone=timezone)
        except Exception:
            self.temporal_resolver = None

        self.description_generator = DescriptionGenerator(examples_path="data/mta_alerts.json")
        self._llm_config = load_llm_config()
        self._llm_config_active = self._llm_config
        self.llm = None
        self.telemetry: Dict[str, int] = {
            "parse_fail": 0,
            "explicit_id_invalid_drop": 0,
            "fallback_used": 0,
            "inferred_stop_accept": 0,
        }

    def compile(self, request: CompileRequest) -> Dict[str, Any]:
        self._set_request_llm_config(request.llm_provider, request.llm_model)

        base_alert = request.alert
        instruction = self._strip_markdown_formatting((request.instruction or "").strip())
        directive_hint = ParsedInstruction(None, None, None, None, None)
        intent = self._parse_intent_llm_first(instruction, directive_hint)

        base_header = (base_alert.header if base_alert else "") or ""
        base_description = (base_alert.description if base_alert else "") or ""

        header = (
            intent.alert_text
            or base_header
            or self._derive_header_from_text(self._strip_directive_clauses(instruction))
        )
        description = base_description
        header = self._clean_header_text(header)

        if not header and not description:
            if instruction:
                header = self._derive_header_from_text(self._strip_directive_clauses(instruction))
                description = ""
            else:
                raise ValueError("Either instruction text or alert header/description must be provided.")

        # Temporal resolution: LLM intent -> deterministic resolver.
        active_periods = self._normalize_active_periods(base_alert.active_periods if base_alert else [])
        temporal_text = intent.temporal_text or instruction or f"{header}\n{description}"
        temporal_override = bool(intent.temporal_text or self._has_temporal_hint(instruction))
        temporal_confidence = 0.4

        resolved_periods: List[Dict[str, Any]] = []
        if self.temporal_resolver:
            for p in self.temporal_resolver.resolve_all(temporal_text):
                resolved_periods.append({"start": p.start, "end": p.end})

        if temporal_override and resolved_periods:
            active_periods = resolved_periods
            temporal_confidence = 0.98
        elif active_periods:
            temporal_confidence = 0.9
        elif resolved_periods:
            active_periods = resolved_periods
            temporal_confidence = 0.9
        else:
            now_iso = datetime.now(self.tz).strftime("%Y-%m-%dT%H:%M:%S")
            active_periods = [{"start": now_iso}]
            temporal_confidence = 0.4

        explicit_route_ids = self.retriever.validate_route_ids(intent.explicit_route_ids)
        explicit_stop_ids = self.retriever.validate_stop_ids(intent.explicit_stop_ids)
        dropped_explicit = max(0, len(intent.explicit_stop_ids) - len(explicit_stop_ids))
        if dropped_explicit:
            self._bump_telemetry("explicit_id_invalid_drop", dropped_explicit)

        explicit_route_entities = [self._build_route_entity(rid) for rid in explicit_route_ids]
        explicit_stop_entities = [self._build_stop_entity(sid) for sid in explicit_stop_ids]

        seed_entities = [e.model_dump(exclude_none=True) for e in (base_alert.informed_entities if base_alert else [])]
        seed_entities = self._dedupe_entities(seed_entities + explicit_route_entities + explicit_stop_entities)

        affected_segments, alternative_segments = self.retriever._split_affected_and_alternative_segments(instruction)
        affected_hints = self.retriever._extract_location_hints(". ".join(affected_segments))
        alternative_hints = {h.lower() for h in self.retriever._extract_location_hints(". ".join(alternative_segments))}
        intent_hints = [h for h in intent.location_phrases if str(h).strip().lower() not in alternative_hints]
        location_hints_override = self._merge_text_tokens(affected_hints, intent_hints)

        retriever_text = (instruction or "").strip() or "\n".join(
            x for x in [header, description, instruction] if x
        ).strip()
        retrieval = self.retriever.retrieve_affected_entities(
            retriever_text,
            seed_entities=seed_entities,
            route_ids_override=explicit_route_ids or None,
            location_hints_override=location_hints_override or None,
            max_stop_candidates=20,
        )

        retrieved_entities = retrieval.get("informed_entities", []) if retrieval.get("status") == "success" else []
        inferred_route_entities = [e for e in retrieved_entities if e.get("route_id") and not e.get("stop_id")]
        inferred_stop_entities = [e for e in retrieved_entities if e.get("stop_id")]
        stop_candidates = retrieval.get("stop_candidates", []) if retrieval.get("status") == "success" else []

        allowed_route_ids = self._merge_unique_tokens(
            explicit_route_ids,
            retrieval.get("route_ids", []),
            [e.get("route_id") for e in inferred_route_entities if e.get("route_id")],
        )
        llm_route_ids: List[str] = []
        llm_stop_ids: List[str] = []
        llm_entity_conf = 0.0
        if self._ensure_llm() and (allowed_route_ids or stop_candidates):
            llm_route_ids, llm_stop_ids, llm_entity_conf = self._llm_select_entities(
                text="\n".join([header, description, instruction]).strip(),
                allowed_route_ids=allowed_route_ids,
                stop_candidates=stop_candidates,
                locked_route_ids=explicit_route_ids,
                locked_stop_ids=explicit_stop_ids,
                location_hints=retrieval.get("location_hints", []),
            )

        if llm_stop_ids and llm_entity_conf >= 0.65:
            self._bump_telemetry("inferred_stop_accept", len(llm_stop_ids))

        # Explicit stop IDs are hard-locked; inferred stops cannot replace them.
        locked_stop_ids = {sid.upper() for sid in explicit_stop_ids}
        if locked_stop_ids:
            filtered_inferred_stops = [e for e in inferred_stop_entities if str(e.get("stop_id", "")).upper() in locked_stop_ids]
            inferred_stop_entities = filtered_inferred_stops
        elif llm_stop_ids and llm_entity_conf >= 0.65:
            inferred_stop_entities = [self._build_stop_entity(sid) for sid in llm_stop_ids]

        inferred_stop_entities = self._prune_stops_for_single_location(
            stop_entities=inferred_stop_entities,
            source_text="\n".join([header, description]).strip() or header,
            stop_candidates=stop_candidates,
        )

        route_entities = self._dedupe_entities(
            explicit_route_entities
            + inferred_route_entities
            + [self._build_route_entity(rid) for rid in llm_route_ids]
        )
        stop_entities = self._dedupe_entities(explicit_stop_entities + inferred_stop_entities)
        informed_entities = self._dedupe_entities(route_entities + stop_entities)
        informed_entities = self._normalize_entities_for_output(
            informed_entities,
            "\n".join([header, description, instruction]),
        )

        route_conf = float(retrieval.get("route_confidence", 0.0))
        stop_conf = float(retrieval.get("stop_confidence", 0.0))
        if explicit_route_ids:
            route_conf = max(route_conf, 0.95)
        if explicit_stop_ids:
            stop_conf = max(stop_conf, 1.0)
        elif llm_entity_conf >= 0.65 and llm_stop_ids:
            stop_conf = max(stop_conf, llm_entity_conf)
        if not route_conf and informed_entities:
            route_conf = 0.75

        base_cause = normalize_cause(base_alert.cause if base_alert else UNKNOWN_CAUSE)
        base_effect = normalize_effect(base_alert.effect if base_alert else UNKNOWN_EFFECT)
        cause_override = intent.cause_hint
        effect_override = intent.effect_hint

        cause, effect, cause_conf, effect_conf = self._resolve_cause_effect(
            text="\n".join([header, description, instruction]).strip(),
            base_cause=base_cause,
            base_effect=base_effect,
            cause_override=cause_override,
            effect_override=effect_override,
        )

        schema_conf = 1.0
        confidence = self._global_confidence(route_conf, stop_conf, temporal_confidence, schema_conf)

        should_try_fallback = bool(retrieval.get("fallback_needed", False)) or (confidence < self.confidence_threshold)
        fallback_used = False
        fallback_conf = 0.0
        if should_try_fallback:
            fallback = self.retriever.geocode_fallback_entities(
                location_hints=retrieval.get("location_hints", []),
                route_ids=[e.get("route_id") for e in informed_entities if e.get("route_id")],
            )
            if fallback.get("status") == "success":
                fallback_used = True
                self._bump_telemetry("fallback_used")
                fallback_conf = float(fallback.get("confidence", 0.0))
                fallback_entities = fallback.get("entities", [])
                if explicit_stop_ids:
                    # Never replace explicit validated stop IDs.
                    fallback_entities = [
                        e for e in fallback_entities if str(e.get("stop_id", "")).upper() in locked_stop_ids
                    ]
                informed_entities = self._dedupe_entities(informed_entities + fallback_entities)
                stop_conf = max(stop_conf, float(fallback.get("confidence", 0.0)))
                confidence = self._global_confidence(route_conf, stop_conf, temporal_confidence, schema_conf)

        # Conservative guardrail when confidence remains below threshold.
        preserve_stops = self._should_preserve_stops_under_low_confidence(
            entities=informed_entities,
            stop_conf=stop_conf,
            retrieval=retrieval,
            fallback_used=fallback_used,
            fallback_conf=fallback_conf,
        )
        if confidence < self.confidence_threshold and not preserve_stops and not explicit_stop_ids:
            informed_entities = self._conservative_entities(informed_entities, seed_entities)

        route_ids_for_text = [e.get("route_id") for e in informed_entities if isinstance(e, dict) and e.get("route_id")]
        header = self._replace_stop_ids_with_names(header, informed_entities)
        header = self.description_generator.render_header_mta(
            llm=self.llm if self._ensure_llm() else None,
            header=header,
            source_text=instruction,
            route_ids=route_ids_for_text,
            effect=effect,
            style_intent=intent.style_intent,
        )
        header = self._clean_header_text(header)
        header = self._format_route_bullets(header, route_ids_for_text)
        if not (header or "").strip():
            fallback_header = (
                (intent.alert_text or "").strip()
                or (base_header or "").strip()
                or self._derive_header_from_text(self._strip_directive_clauses(instruction))
                or self._derive_header_from_text(instruction)
            )
            header = self._format_route_bullets(self._clean_header_text(fallback_header).strip(), route_ids_for_text)
        if not header:
            raise ValueError("Unable to derive a non-empty header from the request.")

        generated_description = self.description_generator.generate_or_null(
            llm=self.llm if self._ensure_llm() else None,
            header=header,
            source_text="\n".join([instruction, description, header]).strip(),
            route_ids=route_ids_for_text,
            cause=cause if cause_conf >= self.confidence_threshold else UNKNOWN_CAUSE,
            effect=effect if effect_conf >= self.confidence_threshold else UNKNOWN_EFFECT,
            current_description=description,
        )
        description = generated_description
        if description:
            description = self._replace_stop_ids_with_names(description, informed_entities)
            description = self._format_route_bullets(description, route_ids_for_text)

        alert_id = self._resolve_alert_id(base_alert.id if base_alert else None, header, description)

        payload = LegacyAlertPayload(
            id=alert_id,
            header=header.strip(),
            description=description.strip() if description else None,
            effect=normalize_effect(effect if effect_conf >= self.confidence_threshold else UNKNOWN_EFFECT),
            cause=normalize_cause(cause if cause_conf >= self.confidence_threshold else UNKNOWN_CAUSE),
            active_periods=[ActivePeriod(**p) for p in active_periods],
            informed_entities=[InformedEntity(**e) for e in informed_entities],
        )

        # exclude_none=True keeps nested objects clean (no stop_id:null, end:null).
        dumped = payload.model_dump(exclude_none=True)
        variants = self._generate_text_variants_bundle(
            header_text=dumped["header"],
            description_text=dumped.get("description"),
        )
        header_text = self._build_translated_string(
            dumped["header"],
            multi=variants.get("header_multi", {}),
        )
        description_text = (
            self._build_translated_string(
                dumped["description"],
                multi=variants.get("description_multi", {}),
            )
            if dumped.get("description")
            else None
        )
        tts_header_text = self._build_tts_translated_string(
            dumped["header"],
            kind="header",
            precomputed=variants.get("tts_header"),
        )
        tts_description_text = (
            self._build_tts_translated_string(
                dumped["description"],
                kind="description",
                precomputed=variants.get("tts_description"),
            )
            if dumped.get("description")
            else None
        )

        data = {
            "id": dumped["id"],
            "active_period": dumped["active_periods"],
            "informed_entity": dumped["informed_entities"],
            "cause": dumped["cause"],
            "effect": dumped["effect"],
            "header_text": header_text,
            "description_text": description_text,
            "tts_header_text": tts_header_text,
            "tts_description_text": tts_description_text,
        }
        return data

    def _build_translated_string(self, text: Optional[str], multi: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        value = str(text or "").strip()
        if not value:
            return None
        html_value = AlertCompiler._to_en_html(value)
        multi_map = dict(multi or {})
        out = [
            {"text": value, "language": "en"},
            {"text": html_value, "language": "en-html"},
        ]
        for code in MULTI_LANG_CODES:
            t = str(multi_map.get(code, "") or "").strip()
            if not t:
                t = value
            out.append({"text": t, "language": code})
        return {
            "translation": out
        }

    def _generate_text_variants_bundle(self, header_text: str, description_text: Optional[str]) -> Dict[str, Any]:
        header = str(header_text or "").strip()
        description = str(description_text or "").strip()
        out = {
            "header_multi": {c: "" for c in MULTI_LANG_CODES},
            "description_multi": {c: "" for c in MULTI_LANG_CODES},
            "tts_header": "",
            "tts_description": "",
        }
        if not header or not self._ensure_llm():
            return out

        prompt = (
            "You are generating localization + TTS variants for a GTFS transit alert.\n"
            "Return strict JSON only with keys:\n"
            "header_zh, header_es, description_zh, description_es, tts_header, tts_description, confidence.\n"
            "Rules:\n"
            "- Preserve facts exactly.\n"
            "- Keep route tokens and IDs unchanged when possible.\n"
            "- `header_*` and `description_*` must be plain text translations.\n"
            "- `tts_*` must be English text optimized for speech (expanded abbreviations where natural).\n"
            "- If description is empty, return empty strings for description_* and tts_description.\n\n"
            f"HEADER_EN: {header}\n"
            f"DESCRIPTION_EN: {description}\n"
        )
        try:
            resp = self.llm.invoke(prompt)
            content = getattr(resp, "content", "") if resp is not None else ""
            if not isinstance(content, str):
                content = str(content)
            parsed = self._extract_first_json_object(content)
            conf = self._coerce_confidence(parsed.get("confidence", 0.0))
            if conf < 0.6:
                return out

            out["header_multi"]["zh"] = str(parsed.get("header_zh", "") or "").strip()
            out["header_multi"]["es"] = str(parsed.get("header_es", "") or "").strip()
            out["description_multi"]["zh"] = str(parsed.get("description_zh", "") or "").strip()
            out["description_multi"]["es"] = str(parsed.get("description_es", "") or "").strip()
            out["tts_header"] = str(parsed.get("tts_header", "") or "").strip()
            out["tts_description"] = str(parsed.get("tts_description", "") or "").strip()
            return out
        except Exception:
            return out

    @staticmethod
    def _to_en_html(text: str) -> str:
        blocks = [b.strip() for b in re.split(r"(?:\r?\n){2,}", text or "") if b.strip()]
        if not blocks:
            return ""
        rendered: List[str] = []
        for block in blocks:
            safe = html.escape(block, quote=False)
            safe = safe.replace("\n", "<br/>")
            rendered.append(f"<p>{safe}</p>")
        return "".join(rendered)

    def _build_tts_translated_string(
        self,
        text: Optional[str],
        kind: str,
        precomputed: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        source = str(text or "").strip()
        if not source:
            return None

        tts_text = str(precomputed or "").strip() or self._deterministic_tts_fallback(source)
        if not tts_text:
            return None
        return {"translation": [{"text": tts_text, "language": "en"}]}

    @staticmethod
    def _deterministic_tts_fallback(text: str) -> str:
        out = str(text or "").strip()
        if not out:
            return ""
        out = out.replace("[", "").replace("]", "")
        replacements = [
            (r"\bSt\b", "Street"),
            (r"\bAve\b", "Avenue"),
            (r"\bAv\b", "Avenue"),
            (r"\bBlvd\b", "Boulevard"),
            (r"\bPkwy\b", "Parkway"),
            (r"\bExpy\b", "Expressway"),
            (r"\bRd\b", "Road"),
            (r"\bDr\b", "Drive"),
            (r"\bN\b", "North"),
            (r"\bS\b", "South"),
            (r"\bE\b", "East"),
            (r"\bW\b", "West"),
        ]
        for pattern, repl in replacements:
            out = re.sub(pattern, repl, out)
        out = re.sub(r"\s{2,}", " ", out).strip()
        return out

    # ------------------------------------------------------------------
    # Cause / effect resolution
    # ------------------------------------------------------------------
    def _resolve_cause_effect(
        self,
        text: str,
        base_cause: str,
        base_effect: str,
        cause_override: Optional[str],
        effect_override: Optional[str],
    ) -> Tuple[str, str, float, float]:
        if cause_override:
            cause = normalize_cause(cause_override)
            cause_conf = 1.0 if cause in CAUSE_ENUMS else 0.0
        else:
            cause = base_cause
            cause_conf = 1.0 if cause != UNKNOWN_CAUSE else 0.0

        if effect_override:
            effect = normalize_effect(effect_override)
            effect_conf = 1.0 if effect in EFFECT_ENUMS else 0.0
        else:
            effect = base_effect
            effect_conf = 1.0 if effect != UNKNOWN_EFFECT else 0.0

        llm_cause, llm_effect, llm_cause_conf, llm_effect_conf = (UNKNOWN_CAUSE, UNKNOWN_EFFECT, 0.0, 0.0)
        if self._ensure_llm():
            llm_cause, llm_effect, llm_cause_conf, llm_effect_conf = self._llm_enum_fallback(text)

        inferred = infer_cause_effect_rule_first(text)

        # Pick the strongest bounded signal for each enum.
        if cause == UNKNOWN_CAUSE:
            if llm_cause_conf >= inferred.cause_confidence and llm_cause_conf > cause_conf:
                cause = llm_cause
                cause_conf = llm_cause_conf
            elif inferred.cause_confidence > cause_conf:
                cause = inferred.cause
                cause_conf = inferred.cause_confidence

        if effect == UNKNOWN_EFFECT:
            if llm_effect_conf >= inferred.effect_confidence and llm_effect_conf > effect_conf:
                effect = llm_effect
                effect_conf = llm_effect_conf
            elif inferred.effect_confidence > effect_conf:
                effect = inferred.effect
                effect_conf = inferred.effect_confidence

        return cause, effect, cause_conf, effect_conf

    def _llm_enum_fallback(self, text: str) -> Tuple[str, str, float, float]:
        if not self._ensure_llm():
            return UNKNOWN_CAUSE, UNKNOWN_EFFECT, 0.0, 0.0

        prompt = (
            "Classify the transit alert into GTFS enums. "
            f"Allowed cause enums: {sorted(CAUSE_ENUMS)}. "
            f"Allowed effect enums: {sorted(EFFECT_ENUMS)}. "
            "Return strict JSON only with keys: cause, effect, cause_confidence, effect_confidence. "
            "Confidence must be 0..1. If uncertain use UNKNOWN_CAUSE/UNKNOWN_EFFECT.\n\n"
            f"Alert text:\n{text}"
        )

        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", "") if response is not None else ""
            if not isinstance(content, str):
                content = str(content)
            json_obj = self._extract_first_json_object(content)
            cause = normalize_cause(str(json_obj.get("cause", UNKNOWN_CAUSE)))
            effect = normalize_effect(str(json_obj.get("effect", UNKNOWN_EFFECT)))
            cause_conf = self._coerce_confidence(json_obj.get("cause_confidence", 0.0))
            effect_conf = self._coerce_confidence(json_obj.get("effect_confidence", 0.0))
            return cause, effect, cause_conf, effect_conf
        except Exception:
            return UNKNOWN_CAUSE, UNKNOWN_EFFECT, 0.0, 0.0

    def _ensure_llm(self) -> bool:
        if self.llm is not None:
            return True
        try:
            self.llm = build_langchain_chat_model(config=self._llm_config_active, temperature=0.0)
            return self.llm is not None
        except Exception:
            self.llm = None
            return False

    def _set_request_llm_config(self, provider: Optional[str], model: Optional[str]) -> None:
        next_config = with_overrides(self._llm_config, provider=provider, model=model)
        if next_config != self._llm_config_active:
            self._llm_config_active = next_config
            self.llm = None

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

    def _llm_select_entities(
        self,
        text: str,
        allowed_route_ids: Sequence[str],
        stop_candidates: Sequence[Dict[str, Any]],
        locked_route_ids: Sequence[str],
        locked_stop_ids: Sequence[str],
        location_hints: Sequence[str],
    ) -> Tuple[List[str], List[str], float]:
        if not self._ensure_llm():
            return [], [], 0.0

        allowed_routes = self._merge_unique_tokens(allowed_route_ids)
        allowed_stop_ids: List[str] = []
        pretty_candidates = []
        for c in stop_candidates:
            stop_id = str(c.get("stop_id", "")).upper().strip()
            route_id = str(c.get("route_id", "")).upper().strip()
            if not stop_id:
                continue
            allowed_stop_ids.append(stop_id)
            pretty_candidates.append(
                {
                    "stop_id": stop_id,
                    "route_id": route_id,
                    "stop_name": str(c.get("stop_name", "")),
                    "score": c.get("score", 0.0),
                }
            )

        prompt = (
            "You are a transit entity selector.\n"
            "Return strict JSON only with keys: selected_route_ids (array), selected_stop_ids (array), confidence (0..1).\n"
            "Rules:\n"
            "- Select only from allowed IDs.\n"
            "- Keep locked IDs if present.\n"
            "- Do not hallucinate.\n"
            "- If uncertain return empty arrays.\n\n"
            f"Alert text: {text}\n"
            f"Location hints: {list(location_hints)}\n"
            f"Allowed route IDs: {allowed_routes}\n"
            f"Allowed stop candidates: {json.dumps(pretty_candidates, ensure_ascii=False)}\n"
            f"Locked route IDs: {list(locked_route_ids)}\n"
            f"Locked stop IDs: {list(locked_stop_ids)}\n"
        )

        try:
            resp = self.llm.invoke(prompt)
            content = getattr(resp, "content", "") if resp is not None else ""
            if not isinstance(content, str):
                content = str(content)
            parsed = self._extract_first_json_object(content)
            routes = parsed.get("selected_route_ids", [])
            stops = parsed.get("selected_stop_ids", [])
            confidence = self._coerce_confidence(parsed.get("confidence", 0.0))
            if not isinstance(routes, list):
                routes = []
            if not isinstance(stops, list):
                stops = []

            route_allow = {r.upper() for r in allowed_routes}
            stop_allow = {s.upper() for s in allowed_stop_ids}
            selected_routes = []
            for rid in routes:
                token = str(rid).strip().upper()
                if token and token in route_allow and token not in selected_routes:
                    selected_routes.append(token)

            selected_stops = []
            for sid in stops:
                token = str(sid).strip().upper()
                if token and token in stop_allow and token not in selected_stops:
                    selected_stops.append(token)

            return selected_routes, selected_stops, confidence
        except Exception:
            return [], [], 0.0

    def _llm_select_stops(
        self,
        text: str,
        route_entities: Sequence[Dict[str, Any]],
        stop_candidates: Sequence[Dict[str, Any]],
        location_hints: Sequence[str],
    ) -> Tuple[List[Dict[str, Any]], float]:
        route_ids = [str(e.get("route_id", "")).upper() for e in route_entities if e.get("route_id")]
        _, stop_ids, confidence = self._llm_select_entities(
            text=text,
            allowed_route_ids=route_ids,
            stop_candidates=stop_candidates,
            locked_route_ids=[],
            locked_stop_ids=[],
            location_hints=location_hints,
        )
        agency = route_entities[0].get("agency_id", "MTA NYCT") if route_entities else "MTA NYCT"
        selected_entities = [{"agency_id": agency, "stop_id": sid} for sid in stop_ids]
        return selected_entities, confidence

    @staticmethod
    def _prune_stops_for_single_location(
        stop_entities: Sequence[Dict[str, Any]],
        source_text: str,
        stop_candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        entities = [dict(e) for e in stop_entities if isinstance(e, dict) and e.get("stop_id")]
        if len(entities) <= 1:
            return entities

        lower = (source_text or "").lower()
        is_single_point = (
            (" near " in lower or " at " in lower)
            and (" between " not in lower)
            and (" from " not in lower or " to " not in lower)
        )
        if not is_single_point:
            return entities

        score_by_stop: Dict[str, float] = {}
        for c in stop_candidates:
            sid = str(c.get("stop_id", "")).upper()
            if not sid:
                continue
            score = float(c.get("score", 0.0))
            score_by_stop[sid] = max(score_by_stop.get(sid, 0.0), score)

        entities.sort(key=lambda e: score_by_stop.get(str(e.get("stop_id", "")).upper(), 0.0), reverse=True)
        return [entities[0]]

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        try:
            out = float(value)
            if out < 0:
                return 0.0
            if out > 1:
                return 1.0
            return out
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Input parsing / merge
    # ------------------------------------------------------------------
    @staticmethod
    def _derive_header_from_text(text: str, max_len: int = 160) -> str:
        first = re.split(r"[\n\r]", text.strip())[0]
        first = first.strip()
        return first[:max_len]

    def _parse_instruction(self, instruction: str) -> ParsedInstruction:
        """Backward-compatible parser wrapper used by older call sites/tests."""
        if not instruction:
            return ParsedInstruction(None, None, None, None, None)

        directive_hint = self._parse_directives(instruction) if self._has_labeled_directives(instruction) else ParsedInstruction(
            None,
            None,
            None,
            None,
            None,
        )
        intent = self._parse_intent_llm_first(instruction, directive_hint)
        return ParsedInstruction(
            header=intent.alert_text or directive_hint.header,
            description=directive_hint.description,
            dates_text=intent.temporal_text or directive_hint.dates_text,
            cause_override=directive_hint.cause_override or intent.cause_hint,
            effect_override=directive_hint.effect_override or intent.effect_hint,
        )

    def _parse_directives(self, text: str) -> ParsedInstruction:
        """Extract structured fields when operator used explicit labeled directives."""
        header = self._extract_labeled_block(
            text,
            "header",
            [
                "description",
                "make description use",
                "dates",
                "cause",
                "effect",
                "what's happening",
                "whats happening",
                "plan your trip",
            ],
        )

        description = self._extract_labeled_block(text, "description", ["dates", "cause", "effect"])
        if not description:
            description = self._extract_phrase_block(text, r"make\s+description\s+use\s*:", ["dates", "cause", "effect"])
        if description:
            description = re.sub(
                r"\bdates?\s+(?:are|is)\s+.+$",
                "",
                description,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip() or description
            description = re.sub(
                r"\btime(?:\s*frame)?\s+(?:is|are)\s+.+$",
                "",
                description,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip() or description

        dates_text = self._extract_labeled_block(text, "dates", ["cause", "effect"])
        if not dates_text:
            m_dates = re.search(r"\bdates?\s+(?:are|is)\s+(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
            if m_dates:
                dates_text = m_dates.group(1).strip()
        if not dates_text:
            m_tf = re.search(r"\btime(?:\s*frame)?\s+(?:is|are)\s+(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
            if m_tf:
                dates_text = m_tf.group(1).strip()

        cause_override = None
        effect_override = None
        m_cause = re.search(r"\bcause\s*(?:to|=|:)\s*([A-Za-z_]+)", text, flags=re.IGNORECASE)
        if m_cause:
            cause_override = m_cause.group(1).strip().upper()
        m_effect = re.search(r"\beffect\s*(?:to|=|:)\s*([A-Za-z_]+)", text, flags=re.IGNORECASE)
        if m_effect:
            effect_override = m_effect.group(1).strip().upper()

        return ParsedInstruction(
            header=header,
            description=description,
            dates_text=dates_text,
            cause_override=cause_override,
            effect_override=effect_override,
        )

    @staticmethod
    def _extract_labeled_block(text: str, label: str, stop_labels: Sequence[str]) -> Optional[str]:
        stop_pattern = "|".join(re.escape(lbl) + r"\s*:" for lbl in stop_labels)
        pattern = re.compile(
            rf"\b{re.escape(label)}\s*:\s*(.+?)(?=(?:\b(?:{stop_pattern})|$))",
            flags=re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(text)
        if not m:
            return None
        out = m.group(1).strip()
        return out or None

    @staticmethod
    def _extract_phrase_block(text: str, phrase_regex: str, stop_labels: Sequence[str]) -> Optional[str]:
        stop_pattern = "|".join(re.escape(lbl) + r"\s*:" for lbl in stop_labels)
        pattern = re.compile(
            rf"{phrase_regex}\s*(.+?)(?=(?:\b(?:{stop_pattern})|$))",
            flags=re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(text)
        if not m:
            return None
        out = m.group(1).strip()
        return out or None

    def _parse_intent_llm_first(self, instruction: str, directive_hint: ParsedInstruction) -> IntentParseResult:
        """LLM-first intent parsing with one retry, then deterministic fallback."""
        if not instruction:
            return IntentParseResult(alert_text=None, temporal_text=None)

        heuristic_routes = self._extract_explicit_route_ids(instruction)
        heuristic_stops = self._extract_explicit_stop_ids(instruction)

        if self._ensure_llm():
            parsed = self._llm_extract_intent_v2(instruction, compact=False)
            if parsed is None:
                parsed = self._llm_extract_intent_v2(instruction, compact=True)
            if parsed is not None:
                route_ids = self._merge_unique_tokens(parsed.explicit_route_ids, heuristic_routes)
                stop_ids = self._merge_unique_tokens(parsed.explicit_stop_ids, heuristic_stops)
                return IntentParseResult(
                    alert_text=parsed.alert_text or directive_hint.header,
                    temporal_text=parsed.temporal_text or directive_hint.dates_text,
                    explicit_route_ids=tuple(route_ids),
                    explicit_stop_ids=tuple(stop_ids),
                    location_phrases=tuple(parsed.location_phrases),
                    effect_hint=parsed.effect_hint,
                    cause_hint=parsed.cause_hint,
                    style_intent=parsed.style_intent,
                    parse_confidence=max(parsed.parse_confidence, 0.7),
                )

            self._bump_telemetry("parse_fail")

        fallback = self._heuristic_extract_intent(instruction)
        return IntentParseResult(
            alert_text=fallback.header or directive_hint.header,
            temporal_text=fallback.dates_text or directive_hint.dates_text,
            explicit_route_ids=tuple(heuristic_routes),
            explicit_stop_ids=tuple(heuristic_stops),
            location_phrases=tuple(self.retriever._extract_location_hints(instruction)),
            effect_hint=directive_hint.effect_override,
            cause_hint=directive_hint.cause_override,
            style_intent="moderate",
            parse_confidence=0.35,
        )

    @staticmethod
    def _strip_markdown_formatting(text: str) -> str:
        """Strip common markdown inline markers (bold/italic) from operator input."""
        out = re.sub(r"\*{1,3}|_{1,3}", "", text or "")
        # Normalize missing spaces after punctuation from operator copy/paste.
        out = re.sub(r"([?!\.])([A-Za-z])", r"\1 \2", out)
        return out.strip()

    @staticmethod
    def _has_labeled_directives(text: str) -> bool:
        """Return True if any explicit labeled directive keyword is present."""
        return bool(re.search(
            r"\b(?:header|description|dates?)\s*:"
            r"|\bdates?\s+(?:are|is)\s"
            r"|\bcause\s*(?:to|=|:)\s"
            r"|\beffect\s*(?:to|=|:)\s",
            text,
            flags=re.IGNORECASE,
        ))

    def _llm_extract_intent_v2(self, instruction: str, compact: bool = False) -> Optional[IntentParseResult]:
        """Bounded LLM extraction of intent slots for free-form operator input."""
        if not self._ensure_llm():
            return None

        if compact:
            prompt = (
                "Extract transit intent from operator text. Return strict JSON only with keys:\n"
                "alert_text, temporal_text, explicit_route_ids, explicit_stop_ids, "
                "location_phrases, effect_hint, cause_hint, style_intent, parse_confidence.\n"
                "Rules: no prose, no markdown; use null or [] when absent. style_intent='moderate'.\n"
                f"INPUT:\n{instruction}"
            )
        else:
            prompt = (
                "You are an MTA alert intent extractor. "
                "Operators write fully free-form text; extract structured intent.\n"
                "Return strict JSON only with keys:\n"
                "alert_text (string|null): rider-facing core alert text without scheduling/meta clauses.\n"
                "temporal_text (string|null): raw time/recurrence phrase(s).\n"
                "explicit_route_ids (array of strings): route IDs explicitly present in input.\n"
                "explicit_stop_ids (array of strings): stop IDs explicitly present in input.\n"
                "location_phrases (array of strings): place phrases that can help stop grounding.\n"
                "effect_hint (string|null): GTFS effect guess if explicit in text.\n"
                "cause_hint (string|null): GTFS cause guess if explicit in text.\n"
                "style_intent (string): use 'moderate'.\n"
                "parse_confidence (number 0..1).\n"
                "Never invent IDs not present in input. Use [] or null when missing.\n\n"
                f"Operator input:\n{instruction}"
            )

        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", "") if response is not None else ""
            if not isinstance(content, str):
                content = str(content)
            parsed = self._extract_first_json_object(content)
            alert_text = str(parsed.get("alert_text") or "").strip() or None
            temporal_text = str(parsed.get("temporal_text") or "").strip() or None
            if temporal_text and temporal_text.lower() in {"null", "none"}:
                temporal_text = None

            route_ids = []
            if isinstance(parsed.get("explicit_route_ids"), list):
                route_ids = [str(x).strip() for x in parsed.get("explicit_route_ids", []) if str(x).strip()]
            stop_ids = []
            if isinstance(parsed.get("explicit_stop_ids"), list):
                stop_ids = [str(x).strip() for x in parsed.get("explicit_stop_ids", []) if str(x).strip()]
            locations = []
            if isinstance(parsed.get("location_phrases"), list):
                locations = [str(x).strip() for x in parsed.get("location_phrases", []) if str(x).strip()]

            effect_hint = str(parsed.get("effect_hint") or "").strip().upper() or None
            cause_hint = str(parsed.get("cause_hint") or "").strip().upper() or None
            style_intent = str(parsed.get("style_intent") or "").strip() or "moderate"
            parse_conf = self._coerce_confidence(parsed.get("parse_confidence", 0.0))

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
        except Exception:
            return None

    def _heuristic_extract_intent(self, instruction: str) -> ParsedInstruction:
        """Deterministic fallback when no labeled directives and no LLM are available."""
        text = (instruction or "").strip()
        if not text:
            return ParsedInstruction(None, None, None, None, None)

        dates_text = None
        date_match = re.search(
            r"(?:\b(?:the\s+)?dates?\s+(?:should\s+be|are|is)\s+(.+)$)|(?:\bdates?\s*:\s*(.+)$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if date_match:
            dates_text = (date_match.group(1) or date_match.group(2) or "").strip() or None

        header_text = self._clean_header_text(text)
        if not header_text:
            return ParsedInstruction(None, None, dates_text, None, None)

        return ParsedInstruction(
            header=header_text,
            description=None,
            dates_text=dates_text,
            cause_override=None,
            effect_override=None,
        )

    def _extract_explicit_stop_ids(self, text: str) -> List[str]:
        ids: List[str] = []
        for m in re.finditer(r"\bstop(?:\s*id)?\s*[:#]?\s*([A-Za-z0-9]{3,10}[NS]?)\b", text, flags=re.IGNORECASE):
            ids.append(m.group(1).strip().upper())
        return self._merge_unique_tokens(ids)

    def _extract_explicit_route_ids(self, text: str) -> List[str]:
        tokens = self.retriever._route_tokens_from_text(text)
        return self._merge_unique_tokens(tokens)

    @staticmethod
    def _has_temporal_hint(text: str) -> bool:
        lower = (text or "").lower()
        hints = [
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
            "from",
            "until",
        ]
        return any(h in lower for h in hints)

    @staticmethod
    def _strip_directive_clauses(text: str) -> str:
        out = (text or "").strip()
        if not out:
            return out

        # Remove trailing directive clauses often used in instruction mode.
        out = re.sub(r"\b(?:the\s+)?dates?\s+should\s+be\s+.+$", "", out, flags=re.IGNORECASE | re.DOTALL).strip()
        out = re.sub(r"\bdates?\s+(?:are|is)\s+.+$", "", out, flags=re.IGNORECASE | re.DOTALL).strip()
        out = re.sub(r"\bdates?\s*:\s*.+$", "", out, flags=re.IGNORECASE | re.DOTALL).strip()
        out = re.sub(r"\bcause\s*(?:to|=|:)\s*[A-Za-z_]+\b", "", out, flags=re.IGNORECASE).strip()
        out = re.sub(r"\beffect\s*(?:to|=|:)\s*[A-Za-z_]+\b", "", out, flags=re.IGNORECASE).strip()
        return re.sub(r"\s{2,}", " ", out).strip(" ,")

    @staticmethod
    def _clean_header_text(text: str) -> str:
        out = (text or "").strip()
        if not out:
            return out

        # Drop common narrative/meta sections from the header.
        stop_markers = [
            r"\bdescription\s*:",
            r"\bwhat(?:['’])?s\s+happening\??",
            r"\bsee\s+a\s+map\b",
            r"\bplan\s+your\s+trip\b",
            r"\b(?:the\s+)?dates?\s+(?:should\s+be|are|is)\b",
            r"\bdates?\s*:",
        ]
        cut_positions = []
        for marker in stop_markers:
            m = re.search(marker, out, flags=re.IGNORECASE)
            if m:
                cut_positions.append(m.start())
        if cut_positions:
            out = out[: min(cut_positions)].strip()

        return re.sub(r"\s{2,}", " ", out).strip(" ,.;:-")

    def _replace_stop_ids_with_names(self, text: str, entities: Sequence[Dict[str, Any]]) -> str:
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
        # Title-case while keeping short directional tokens uppercase when standalone.
        titled = name.title()
        titled = re.sub(r"\bN\b", "N", titled)
        titled = re.sub(r"\bS\b", "S", titled)
        titled = re.sub(r"\bE\b", "E", titled)
        titled = re.sub(r"\bW\b", "W", titled)
        return titled

    @staticmethod
    def _format_route_bullets(text: str, route_ids: Sequence[str]) -> str:
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

        # Longer IDs first to avoid partial masking.
        unique_ids.sort(key=len, reverse=True)

        for rid in unique_ids:
            variants = [rid]
            if "+" in rid:
                variants.append(rid.replace("+", "-SBS"))

            for variant in variants:
                escaped = re.escape(variant)
                # Wrap only standalone route mentions that are not already bracketed.
                pattern = re.compile(
                    rf"(?<!\[)\b{escaped}\b(?!\])",
                    flags=re.IGNORECASE,
                )
                out = pattern.sub(lambda m: f"[{m.group(0)}]", out)

        return out

    # ------------------------------------------------------------------
    # Validation / normalization
    # ------------------------------------------------------------------
    def _normalize_active_periods(self, periods: Sequence[Any]) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        for period in periods or []:
            if isinstance(period, ActivePeriod):
                start = self._normalize_iso(period.start)
                end = self._normalize_iso(period.end) if period.end else None
            elif isinstance(period, dict):
                start = self._normalize_iso(period.get("start"))
                end = self._normalize_iso(period.get("end")) if period.get("end") is not None else None
            else:
                continue

            if not start:
                continue

            row = {"start": start}
            if end and end >= start:
                row["end"] = end
            normalized.append(row)

        return normalized

    def _normalize_iso(self, value: Any) -> Optional[str]:
        if value is None:
            return None

        # Unix timestamp support.
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=self.tz).strftime("%Y-%m-%dT%H:%M:%S")
            except Exception:
                return None

        text = str(value).strip()
        if not text:
            return None

        # Numeric string unix timestamp.
        if re.fullmatch(r"\d{9,11}", text):
            try:
                return datetime.fromtimestamp(float(text), tz=self.tz).strftime("%Y-%m-%dT%H:%M:%S")
            except Exception:
                return None

        try:
            clean = text.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean)
            if dt.tzinfo is not None:
                dt = dt.astimezone(self.tz)
            return dt.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None

    @staticmethod
    def _dedupe_entities(entities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            agency_id = str(entity.get("agency_id", "")).strip() or "MTA NYCT"
            route_id = str(entity.get("route_id", "")).strip().upper() if entity.get("route_id") else ""
            stop_id = str(entity.get("stop_id", "")).strip().upper() if entity.get("stop_id") else ""

            # Keep route entities and stop entities distinct; stop entities should
            # not repeat route_id in final legacy payload.
            if stop_id:
                route_id = ""
            if not route_id and not stop_id:
                continue
            key = (agency_id, route_id, stop_id)
            if key in seen:
                continue
            seen.add(key)
            row = {"agency_id": agency_id}
            if route_id:
                row["route_id"] = route_id
            if stop_id:
                row["stop_id"] = stop_id
            out.append(row)
        return out

    @staticmethod
    def _normalize_entities_for_output(
        entities: Sequence[Dict[str, Any]],
        source_text: str,
    ) -> List[Dict[str, Any]]:
        text = source_text or ""
        explicit_direction = AlertCompiler._has_explicit_subway_stop_direction(text)
        normalized: List[Dict[str, Any]] = []
        for e in entities:
            if not isinstance(e, dict):
                continue
            row = dict(e)
            agency = str(row.get("agency_id", "")).strip().upper()
            stop_id = str(row.get("stop_id", "")).strip().upper() if row.get("stop_id") else ""
            if (
                agency == "MTASBWY"
                and stop_id
                and not explicit_direction
                and re.fullmatch(r"[A-Z0-9]+[NS]", stop_id)
            ):
                row["stop_id"] = stop_id[:-1]
            normalized.append(row)
        return AlertCompiler._dedupe_entities(normalized)

    @staticmethod
    def _has_explicit_subway_stop_direction(text: str) -> bool:
        lower = (text or "").lower()
        directional_phrases = (
            "north entrance",
            "south entrance",
            "northbound platform",
            "southbound platform",
            "platform n",
            "platform s",
        )
        if any(p in lower for p in directional_phrases):
            return True
        # Explicit stop ID direction e.g. 118N / A17S
        if re.search(r"\b[A-Za-z0-9]{2,5}[NS]\b", text or ""):
            return True
        return False

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

    @staticmethod
    def _merge_text_tokens(*groups: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for group in groups:
            for value in group or []:
                token = str(value or "").strip()
                if not token:
                    continue
                key = token.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(token)
        return out

    def _build_route_entity(self, route_id: str) -> Dict[str, Any]:
        rid = self.retriever.normalize_route_id(route_id)
        return {
            "agency_id": self.retriever.agency_for_route_id(rid),
            "route_id": rid,
        }

    def _build_stop_entity(self, stop_id: str) -> Dict[str, Any]:
        sid = self.retriever.normalize_stop_id(stop_id)
        return {
            "agency_id": self.retriever.agency_for_stop_id(sid),
            "stop_id": sid,
        }

    def _bump_telemetry(self, key: str, amount: int = 1) -> None:
        if key not in self.telemetry:
            self.telemetry[key] = 0
        self.telemetry[key] += max(1, int(amount))

    @staticmethod
    def _conservative_entities(
        current_entities: Sequence[Dict[str, Any]],
        seed_entities: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if current_entities:
            route_only = [
                {"agency_id": e["agency_id"], "route_id": e["route_id"]}
                for e in current_entities
                if e.get("route_id")
            ]
            deduped = AlertCompiler._dedupe_entities(route_only)
            if deduped:
                return deduped

        seed_route_only = [
            {"agency_id": e.get("agency_id", "MTA NYCT"), "route_id": e.get("route_id")}
            for e in seed_entities
            if isinstance(e, dict) and e.get("route_id")
        ]
        return AlertCompiler._dedupe_entities(seed_route_only)

    @staticmethod
    def _global_confidence(route_conf: float, stop_conf: float, temporal_conf: float, schema_conf: float) -> float:
        score = (0.35 * route_conf) + (0.35 * stop_conf) + (0.20 * temporal_conf) + (0.10 * schema_conf)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _should_preserve_stops_under_low_confidence(
        entities: Sequence[Dict[str, Any]],
        stop_conf: float,
        retrieval: Dict[str, Any],
        fallback_used: bool,
        fallback_conf: float,
    ) -> bool:
        has_stops = any(isinstance(e, dict) and e.get("stop_id") for e in entities)
        if not has_stops:
            return False

        matched_stop_count = int(retrieval.get("matched_stop_count", 0) or 0)
        if matched_stop_count > 0 and stop_conf >= 0.55:
            return True

        if fallback_used and fallback_conf >= 0.50:
            return True

        return stop_conf >= 0.70

    @staticmethod
    def _resolve_alert_id(existing_id: Optional[str], header: str, description: str) -> str:
        if existing_id and str(existing_id).strip():
            return str(existing_id).strip()
        seed = f"{(header or '').strip()}|{(description or '').strip()}|{datetime.utcnow().isoformat()}"
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        return f"lmm:generated:{digest}"
