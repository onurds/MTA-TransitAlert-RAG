from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
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


class LegacyAlertPayload(BaseModel):
    id: str = ""
    header: str = ""
    description: Optional[str] = None
    effect: str = UNKNOWN_EFFECT
    cause: str = UNKNOWN_CAUSE
    severity: Optional[Any] = None
    active_periods: List[ActivePeriod] = Field(default_factory=list)
    informed_entities: List[InformedEntity] = Field(default_factory=list)


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

    def compile(self, request: CompileRequest) -> Dict[str, Any]:
        self._set_request_llm_config(request.llm_provider, request.llm_model)

        base_alert = request.alert
        instruction = (request.instruction or "").strip()
        parsed = self._parse_instruction(instruction)

        base_header = (base_alert.header if base_alert else "") or ""
        base_description = (base_alert.description if base_alert else "") or ""

        header = parsed.header or base_header
        description = parsed.description or base_description

        if not header and not description:
            if instruction:
                cleaned_instruction = self._strip_directive_clauses(instruction)
                description = ""
                header = self._derive_header_from_text(cleaned_instruction)
            else:
                raise ValueError("Either instruction text or alert header/description must be provided.")

        active_periods = self._normalize_active_periods(base_alert.active_periods if base_alert else [])
        temporal_text = parsed.dates_text or instruction or f"{header}\n{description}"
        temporal_override = bool(parsed.dates_text) or self._has_temporal_hint(instruction)
        temporal_confidence = 0.4

        resolved_period = None
        if self.temporal_resolver:
            resolved_period = self.temporal_resolver.resolve(temporal_text)

        if temporal_override and resolved_period:
            active_periods = [{"start": resolved_period.start, "end": resolved_period.end}]
            temporal_confidence = 0.98
        elif active_periods:
            temporal_confidence = 0.9
        elif resolved_period:
            active_periods = [{"start": resolved_period.start, "end": resolved_period.end}]
            temporal_confidence = 0.9
        else:
            now_iso = datetime.now(self.tz).strftime("%Y-%m-%dT%H:%M:%S")
            active_periods = [{"start": now_iso}]
            temporal_confidence = 0.4

        seed_entities = [e.model_dump(exclude_none=True) for e in (base_alert.informed_entities if base_alert else [])]
        retriever_text = "\n".join(x for x in [header, description, instruction] if x).strip()
        retrieval = self.retriever.retrieve_affected_entities(retriever_text, seed_entities=seed_entities)

        retrieved_entities = retrieval.get("informed_entities", []) if retrieval.get("status") == "success" else []
        if retrieved_entities:
            informed_entities = self._dedupe_entities(retrieved_entities)
        else:
            informed_entities = self._dedupe_entities(seed_entities)

        route_entities = [e for e in informed_entities if e.get("route_id") and not e.get("stop_id")]
        stop_entities = [e for e in informed_entities if e.get("stop_id")]
        stop_candidates = retrieval.get("stop_candidates", []) if retrieval.get("status") == "success" else []

        llm_stop_conf = 0.0
        if stop_candidates and self._ensure_llm():
            llm_selected_stops, llm_stop_conf = self._llm_select_stops(
                text="\n".join([header, description, instruction]).strip(),
                route_entities=route_entities,
                stop_candidates=stop_candidates,
                location_hints=retrieval.get("location_hints", []),
            )
            if llm_selected_stops and llm_stop_conf >= 0.7:
                stop_entities = llm_selected_stops

        stop_entities = self._prune_stops_for_single_location(
            stop_entities=stop_entities,
            source_text="\n".join([header, description, instruction]).strip(),
            stop_candidates=stop_candidates,
        )

        informed_entities = self._dedupe_entities(route_entities + stop_entities)

        informed_entities = self._normalize_entities_for_output(
            informed_entities,
            "\n".join([header, description, instruction]),
        )

        route_conf = float(retrieval.get("route_confidence", 0.0))
        stop_conf = max(float(retrieval.get("stop_confidence", 0.0)), llm_stop_conf)
        if not route_conf and informed_entities:
            route_conf = 0.75

        base_cause = normalize_cause(base_alert.cause if base_alert else UNKNOWN_CAUSE)
        base_effect = normalize_effect(base_alert.effect if base_alert else UNKNOWN_EFFECT)

        cause, effect, cause_conf, effect_conf = self._resolve_cause_effect(
            text="\n".join([header, description, instruction]).strip(),
            base_cause=base_cause,
            base_effect=base_effect,
            cause_override=parsed.cause_override,
            effect_override=parsed.effect_override,
        )

        schema_conf = 1.0
        confidence = self._global_confidence(route_conf, stop_conf, temporal_confidence, schema_conf)

        should_try_fallback = bool(retrieval.get("fallback_needed", False)) or (confidence < self.confidence_threshold)
        if should_try_fallback:
            fallback = self.retriever.geocode_fallback_entities(
                location_hints=retrieval.get("location_hints", []),
                route_ids=[e.get("route_id") for e in informed_entities if e.get("route_id")],
            )
            if fallback.get("status") == "success":
                informed_entities = self._dedupe_entities(informed_entities + fallback.get("entities", []))
                stop_conf = max(stop_conf, float(fallback.get("confidence", 0.0)))
                confidence = self._global_confidence(route_conf, stop_conf, temporal_confidence, schema_conf)

        # Conservative guardrail when confidence remains below threshold.
        if confidence < self.confidence_threshold:
            informed_entities = self._conservative_entities(informed_entities, seed_entities)

        route_ids_for_text = [e.get("route_id") for e in informed_entities if isinstance(e, dict) and e.get("route_id")]
        header = self._format_route_bullets(header, route_ids_for_text)
        if not (header or "").strip():
            fallback_header = (
                (parsed.header or "").strip()
                or (base_header or "").strip()
                or self._derive_header_from_text(self._strip_directive_clauses(instruction))
                or self._derive_header_from_text(instruction)
            )
            header = fallback_header.strip()
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
            description = self._format_route_bullets(description, route_ids_for_text)

        alert_id = self._resolve_alert_id(base_alert.id if base_alert else None, header, description)

        payload = LegacyAlertPayload(
            id=alert_id,
            header=header.strip(),
            description=description.strip() if description else None,
            effect=normalize_effect(effect if effect_conf >= self.confidence_threshold else UNKNOWN_EFFECT),
            cause=normalize_cause(cause if cause_conf >= self.confidence_threshold else UNKNOWN_CAUSE),
            severity=base_alert.severity if base_alert else None,
            active_periods=[ActivePeriod(**p) for p in active_periods],
            informed_entities=[InformedEntity(**e) for e in informed_entities],
        )

        data = payload.model_dump(exclude_none=True)
        data.setdefault("description", None)
        data.setdefault("severity", None)
        return data

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

    def _llm_select_stops(
        self,
        text: str,
        route_entities: Sequence[Dict[str, Any]],
        stop_candidates: Sequence[Dict[str, Any]],
        location_hints: Sequence[str],
    ) -> Tuple[List[Dict[str, Any]], float]:
        if not self._ensure_llm():
            return [], 0.0

        route_ids = [str(e.get("route_id", "")).upper() for e in route_entities if e.get("route_id")]
        allowed_stop_ids = []
        pretty_candidates = []
        for c in stop_candidates:
            stop_id = str(c.get("stop_id", "")).upper()
            if not stop_id:
                continue
            allowed_stop_ids.append(stop_id)
            pretty_candidates.append(
                {
                    "stop_id": stop_id,
                    "route_id": str(c.get("route_id", "")).upper(),
                    "stop_name": str(c.get("stop_name", "")),
                    "score": c.get("score", 0.0),
                }
            )
        if not pretty_candidates:
            return [], 0.0

        prompt = (
            "You are a transit alert entity selector. "
            "Pick ONLY affected stop_ids from provided candidates. "
            "Do not hallucinate stop IDs. "
            "If uncertain, return an empty list. "
            "Return strict JSON only with keys: selected_stop_ids (array), confidence (0..1).\\n\\n"
            f"Routes: {route_ids}\\n"
            f"Location hints: {list(location_hints)}\\n"
            f"Alert text: {text}\\n"
            f"Allowed candidates: {json.dumps(pretty_candidates, ensure_ascii=False)}"
        )

        try:
            resp = self.llm.invoke(prompt)
            content = getattr(resp, "content", "") if resp is not None else ""
            if not isinstance(content, str):
                content = str(content)
            parsed = self._extract_first_json_object(content)
            selected = parsed.get("selected_stop_ids", [])
            confidence = self._coerce_confidence(parsed.get("confidence", 0.0))
            if not isinstance(selected, list):
                return [], 0.0

            allowed = set(allowed_stop_ids)
            selected_ids: List[str] = []
            for sid in selected:
                token = str(sid).strip().upper()
                if token in allowed and token not in selected_ids:
                    selected_ids.append(token)

            selected_entities = []
            agency = route_entities[0].get("agency_id", "MTA NYCT") if route_entities else "MTA NYCT"
            for sid in selected_ids:
                selected_entities.append({"agency_id": agency, "stop_id": sid})
            return selected_entities, confidence
        except Exception:
            return [], 0.0

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
        if not instruction:
            return ParsedInstruction(None, None, None, None, None)

        text = instruction.strip()

        header = self._extract_labeled_block(
            text,
            "header",
            ["description", "make description use", "dates", "cause", "effect"],
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

        dates_text = self._extract_labeled_block(text, "dates", ["cause", "effect"])
        if not dates_text:
            m_dates = re.search(r"\bdates?\s+(?:are|is)\s+(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
            if m_dates:
                dates_text = m_dates.group(1).strip()

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
        out = re.sub(r"\bdates?\s+(?:are|is)\s+.+$", "", out, flags=re.IGNORECASE | re.DOTALL).strip()
        out = re.sub(r"\bdates?\s*:\s*.+$", "", out, flags=re.IGNORECASE | re.DOTALL).strip()
        out = re.sub(r"\bcause\s*(?:to|=|:)\s*[A-Za-z_]+\b", "", out, flags=re.IGNORECASE).strip()
        out = re.sub(r"\beffect\s*(?:to|=|:)\s*[A-Za-z_]+\b", "", out, flags=re.IGNORECASE).strip()
        return re.sub(r"\s{2,}", " ", out).strip(" ,")

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
    def _resolve_alert_id(existing_id: Optional[str], header: str, description: str) -> str:
        if existing_id and str(existing_id).strip():
            return str(existing_id).strip()
        seed = f"{(header or '').strip()}|{(description or '').strip()}|{datetime.utcnow().isoformat()}"
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        return f"lmm:generated:{digest}"
