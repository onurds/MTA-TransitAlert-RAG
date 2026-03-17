from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from pipeline.graph import GraphRetriever
from pipeline.gtfs_rules import UNKNOWN_CAUSE, UNKNOWN_EFFECT, normalize_cause, normalize_effect
from pipeline.llm_config import build_langchain_chat_model, load_llm_config, with_overrides
from pipeline.temporal_resolver import TemporalResolver

from .confidence import global_confidence, should_preserve_stops_under_low_confidence
from .entity_selector import EntitySelector
from .enum_resolver import EnumResolver
from .intent_parser import IntentParser
from .mercury_resolver import MercuryResolver
from .models import CompileRequest
from .output_builder import OutputBuilder
from .temporal_selector import TemporalSelector
from .text_renderer import TextRenderer
from .text_mode_resolver import TextModeResolver
from .utils import (
    build_route_entity,
    build_stop_entity,
    conservative_entities,
    dedupe_entities,
    derive_header_from_text,
    has_temporal_hint,
    merge_text_tokens,
    merge_unique_tokens,
    normalize_entities_for_output,
    resolve_alert_id,
)

CONFIDENCE_THRESHOLD = 0.85
ENUM_CONFIDENCE_THRESHOLD = 0.6


class AlertCompiler:
    def __init__(
        self,
        graph_path: str = "data/mta_knowledge_graph.gpickle",
        calendar_path: str = "data/2026_english_calendar.csv",
        timezone: str = "America/New_York",
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        enum_confidence_threshold: float = ENUM_CONFIDENCE_THRESHOLD,
    ):
        self.retriever = GraphRetriever(graph_path=graph_path)
        self.temporal_resolver: Optional[TemporalResolver]
        self.tz = ZoneInfo(timezone)
        self.confidence_threshold = confidence_threshold
        self.enum_confidence_threshold = enum_confidence_threshold

        try:
            self.temporal_resolver = TemporalResolver(calendar_path=calendar_path, timezone=timezone)
        except Exception:
            self.temporal_resolver = None

        self._llm_config = load_llm_config()
        self._llm_config_active = self._llm_config
        self.llm = None
        self.telemetry: Dict[str, int] = {
            "parse_fail": 0,
            "explicit_id_invalid_drop": 0,
            "fallback_used": 0,
            "inferred_stop_accept": 0,
            "text_layout_retry": 0,
            "text_layout_validation_fail": 0,
            "text_layout_fallback_used": 0,
        }
        self.last_compile_report: Dict[str, Any] = {"stages": []}

        self.intent_parser = IntentParser(
            retriever=self.retriever,
            ensure_llm=self._ensure_llm,
            llm_getter=lambda: self.llm,
            bump_telemetry=self._bump_telemetry,
        )
        self.entity_selector = EntitySelector(
            ensure_llm=self._ensure_llm,
            llm_getter=lambda: self.llm,
        )
        self.enum_resolver = EnumResolver(
            ensure_llm=self._ensure_llm,
            llm_getter=lambda: self.llm,
            min_confidence=self.enum_confidence_threshold,
        )
        self.mercury_resolver = MercuryResolver(
            ensure_llm=self._ensure_llm,
            llm_getter=lambda: self.llm,
            min_confidence=self.enum_confidence_threshold,
        )
        self.text_renderer = TextRenderer(retriever=self.retriever)
        self.text_mode_resolver = TextModeResolver(bump_telemetry=self._bump_telemetry)
        self.temporal_selector = TemporalSelector()
        self.output_builder = OutputBuilder(
            ensure_llm=self._ensure_llm,
            llm_getter=lambda: self.llm,
        )

    def compile(self, request: CompileRequest) -> Dict[str, Any]:
        print(f"[DEBUG] Starting compilation for instruction: {request.instruction[:50]}...")
        compile_report: Dict[str, Any] = {
            "provider": request.llm_provider or self._llm_config.provider,
            "model": request.llm_model or (
                self._llm_config.openrouter_model_name if (request.llm_provider or self._llm_config.provider) == "openrouter"
                else self._llm_config.gemini_model_name if (request.llm_provider or self._llm_config.provider) == "gemini"
                else self._llm_config.local_model_name
            ),
            "reasoning_effort": request.llm_reasoning_effort or self._llm_config.openrouter_reasoning_effort,
            "stages": [],
        }
        self._set_request_llm_config(
            request.llm_provider,
            request.llm_model,
            request.llm_reasoning_effort,
        )

        instruction = (request.instruction or "").strip()
        if not instruction:
            raise ValueError("instruction is required and cannot be empty.")

        print("[DEBUG] Phase 1: Intent Parsing...")
        intent = self.intent_parser.parse(instruction)
        print(f"[DEBUG] Intent parsed: {intent.parse_confidence}")
        compile_report["stages"].append({
            "stage": "intent_parsing",
            "source": "llm",
            "details": f"LLM extracted structured intent with confidence {intent.parse_confidence:.2f}.",
        })

        header_hint = intent.alert_text or None
        header = header_hint or derive_header_from_text(instruction)
        header = " ".join((header or "").split())
        if not header:
            header = derive_header_from_text(instruction)
        if not header:
            raise ValueError("Unable to derive a non-empty header from the request.")

        temporal_text = intent.temporal_text or instruction or header
        temporal_override = bool(intent.temporal_text or has_temporal_hint(instruction))
        temporal_confidence = 0.4
        active_periods: List[Dict[str, Any]] = []
        reference_dt = datetime.now(self.tz)
        compiled_at_posix = int(reference_dt.timestamp())

        resolved_periods: List[Dict[str, Any]] = []
        if self.temporal_resolver:
            print("[DEBUG] Phase 2: Temporal Resolution...")
            llm_periods = self.temporal_selector.resolve_periods(
                llm=self.llm,
                temporal_text=temporal_text,
                full_instruction=instruction,
                resolver=self.temporal_resolver,
                reference_dt=reference_dt,
            )
            for p in llm_periods:
                resolved_periods.append({"start": p.start, "end": p.end})

        if temporal_override and resolved_periods:
            active_periods = resolved_periods
            temporal_confidence = 0.98
        elif resolved_periods:
            active_periods = resolved_periods
            temporal_confidence = 0.9
        else:
            now_iso = reference_dt.strftime("%Y-%m-%dT%H:%M:%S")
            active_periods = [{"start": now_iso}]
        compile_report["stages"].append({
            "stage": "temporal_resolution",
            "source": "llm" if resolved_periods else "deterministic",
            "details": (
                f"Resolved {len(resolved_periods)} active period(s) with the LLM-backed calendar selector."
                if resolved_periods
                else "No concrete periods were resolved, so a deterministic immediate-start fallback was used."
            ),
        })
        no_timeframe_mentioned = not temporal_override and not resolved_periods

        print("[DEBUG] Phase 3: Entity Retrieval...")
        explicit_route_ids = self.retriever.validate_route_ids(intent.explicit_route_ids)
        explicit_stop_ids = self.retriever.validate_stop_ids(intent.explicit_stop_ids)
        dropped_explicit = max(0, len(intent.explicit_stop_ids) - len(explicit_stop_ids))
        if dropped_explicit:
            self._bump_telemetry("explicit_id_invalid_drop", dropped_explicit)

        explicit_route_entities = [build_route_entity(self.retriever, rid) for rid in explicit_route_ids]
        explicit_stop_entities = [build_stop_entity(self.retriever, sid) for sid in explicit_stop_ids]
        seed_entities = dedupe_entities(explicit_route_entities + explicit_stop_entities)

        affected_segments, alternative_segments = self.retriever._split_affected_and_alternative_segments(instruction)
        affected_hints = self.retriever._extract_location_hints(". ".join(affected_segments))
        alternative_hints = {h.lower() for h in self.retriever._extract_location_hints(". ".join(alternative_segments))}
        intent_hints = [
            h
            for h in intent.location_phrases
            if str(h).strip().lower() not in alternative_hints and not self.retriever.is_route_name_phrase(str(h))
        ]
        if not affected_hints and not self.retriever._has_strong_stop_intent(instruction):
            intent_hints = []
        location_hints_override = merge_text_tokens(affected_hints, intent_hints)

        retrieval = self.retriever.retrieve_affected_entities(
            instruction,
            seed_entities=seed_entities,
            route_ids_override=explicit_route_ids or None,
            location_hints_override=location_hints_override or None,
            max_stop_candidates=20,
        )
        compile_report["stages"].append({
            "stage": "entity_retrieval",
            "source": "deterministic",
            "details": (
                f"Graph retrieval produced {len(retrieval.get('informed_entities', [])) if retrieval.get('status') == 'success' else 0} "
                "candidate informed entities."
            ),
        })

        retrieved_entities = retrieval.get("informed_entities", []) if retrieval.get("status") == "success" else []
        inferred_route_entities = [e for e in retrieved_entities if e.get("route_id") and not e.get("stop_id")]
        inferred_stop_entities = [e for e in retrieved_entities if e.get("stop_id")]
        stop_candidates = retrieval.get("stop_candidates", []) if retrieval.get("status") == "success" else []

        allowed_route_ids = merge_unique_tokens(
            explicit_route_ids,
            retrieval.get("route_ids", []),
            [e.get("route_id") for e in inferred_route_entities if e.get("route_id")],
        )

        llm_route_ids: List[str] = []
        llm_stop_ids: List[str] = []
        llm_entity_conf = 0.0
        if allowed_route_ids or stop_candidates:
            print("[DEBUG] Phase 4: Entity Selection...")
            llm_route_ids, llm_stop_ids, llm_entity_conf = self.entity_selector.llm_select_entities(
                text="\n".join([header, instruction]).strip(),
                allowed_route_ids=allowed_route_ids,
                stop_candidates=stop_candidates,
                locked_route_ids=explicit_route_ids,
                locked_stop_ids=explicit_stop_ids,
                location_hints=retrieval.get("location_hints", []),
            )
        compile_report["stages"].append({
            "stage": "entity_selection",
            "source": "llm" if (allowed_route_ids or stop_candidates) else "deterministic",
            "details": (
                f"LLM selected {len(llm_route_ids)} route(s) and {len(llm_stop_ids)} stop(s) with confidence {llm_entity_conf:.2f}."
                if (allowed_route_ids or stop_candidates)
                else "Entity selection used deterministic retrieval only because there were no candidates to rank."
            ),
        })

        if llm_stop_ids and llm_entity_conf >= 0.65:
            self._bump_telemetry("inferred_stop_accept", len(llm_stop_ids))

        locked_stop_ids = {sid.upper() for sid in explicit_stop_ids}
        if locked_stop_ids:
            inferred_stop_entities = [
                e for e in inferred_stop_entities if str(e.get("stop_id", "")).upper() in locked_stop_ids
            ]
        elif llm_stop_ids and llm_entity_conf >= 0.65:
            inferred_stop_entities = [build_stop_entity(self.retriever, sid) for sid in llm_stop_ids]

        inferred_stop_entities = self.entity_selector.choose_stops_for_single_location(
            stop_entities=inferred_stop_entities,
            source_text="\n".join([header, instruction]),
            stop_candidates=stop_candidates,
            location_hints=retrieval.get("location_hints", []),
        )

        route_entities = dedupe_entities(
            explicit_route_entities + inferred_route_entities + [build_route_entity(self.retriever, rid) for rid in llm_route_ids]
        )
        stop_entities = dedupe_entities(explicit_stop_entities + inferred_stop_entities)
        informed_entities = dedupe_entities(route_entities + stop_entities)
        informed_entities = normalize_entities_for_output(
            informed_entities,
            "\n".join([header, instruction]),
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

        print("[DEBUG] Phase 5: Enum Resolution...")
        cause_effect = self.enum_resolver.resolve(
            text="\n".join([header, instruction]).strip(),
            cause_override=intent.cause_hint,
            effect_override=intent.effect_hint,
        )
        compile_report["stages"].append({
            "stage": "enum_resolution",
            "source": "llm" if (
                cause_effect.cause_confidence >= self.enum_confidence_threshold
                or cause_effect.effect_confidence >= self.enum_confidence_threshold
            ) else "deterministic",
            "details": (
                f"Cause/effect were accepted from the LLM classifier ({cause_effect.cause_confidence:.2f}/{cause_effect.effect_confidence:.2f})."
                if (cause_effect.cause_confidence >= self.enum_confidence_threshold or cause_effect.effect_confidence >= self.enum_confidence_threshold)
                else "Cause/effect fell back to deterministic defaults or low-confidence outputs."
            ),
        })

        schema_conf = 1.0
        confidence = global_confidence(route_conf, stop_conf, temporal_confidence, schema_conf)

        should_try_fallback = bool(retrieval.get("fallback_needed", False)) or (confidence < self.confidence_threshold)
        fallback_attempted = False
        fallback_used = False
        fallback_success = False
        fallback_conf = 0.0
        fallback_added_count = 0
        if should_try_fallback:
            print("[DEBUG] Phase 6: Fallback...")
            fallback_attempted = True
            fallback = self.retriever.geocode_fallback_entities(
                location_hints=retrieval.get("location_hints", []),
                route_ids=[str(e["route_id"]) for e in informed_entities if isinstance(e, dict) and e.get("route_id")],
            )
            if fallback.get("status") == "success":
                fallback_success = True
                fallback_conf = float(fallback.get("confidence", 0.0))
                fallback_entities = fallback.get("entities", [])
                if explicit_stop_ids:
                    fallback_entities = [
                        e for e in fallback_entities if str(e.get("stop_id", "")).upper() in locked_stop_ids
                    ]
                before_count = len(informed_entities)
                informed_entities = dedupe_entities(informed_entities + fallback_entities)
                fallback_added_count = max(0, len(informed_entities) - before_count)
                if fallback_added_count > 0:
                    fallback_used = True
                    self._bump_telemetry("fallback_used")
                    stop_conf = max(stop_conf, float(fallback.get("confidence", 0.0)))
                    confidence = global_confidence(route_conf, stop_conf, temporal_confidence, schema_conf)
        compile_report["stages"].append({
            "stage": "geocode_fallback",
            "source": "deterministic",
            "details": (
                f"External geocode fallback added {fallback_added_count} new "
                f"{'entity' if fallback_added_count == 1 else 'entities'} with confidence {fallback_conf:.2f}."
                if fallback_used
                else (
                    f"External geocode fallback was attempted and returned a match at confidence {fallback_conf:.2f}, "
                    "but it did not change the final entities."
                    if fallback_success
                    else (
                        "External geocode fallback was attempted but did not return a successful result."
                        if fallback_attempted
                        else "External geocode fallback was not needed."
                    )
                )
            ),
        })

        preserve_stops = should_preserve_stops_under_low_confidence(
            entities=informed_entities,
            stop_conf=stop_conf,
            retrieval=retrieval,
            fallback_used=fallback_used,
            fallback_conf=fallback_conf,
        )
        if confidence < self.confidence_threshold and not preserve_stops and not explicit_stop_ids:
            informed_entities = conservative_entities(informed_entities)

        route_ids_for_text: List[str] = [
            str(e["route_id"]) for e in informed_entities if isinstance(e, dict) and e.get("route_id")
        ]
        print("[DEBUG] Phase 7: Text Mode Resolution...")
        header_text, description = self.text_mode_resolver.resolve(
            llm=self.llm,
            instruction=instruction,
            route_ids=route_ids_for_text,
            cause=cause_effect.cause or UNKNOWN_CAUSE,
            effect=cause_effect.effect,
            text_mode=request.text_mode,
            header_hint=header_hint if request.text_mode == "rewrite" else None,
        )
        compile_report["stages"].append({
            "stage": "text_mode_resolution",
            **self.text_mode_resolver.last_resolution_report,
        })
        header = self.text_renderer.replace_stop_ids_with_names(header_text, informed_entities)
        header = self.text_renderer.format_route_bullets(header, route_ids_for_text)
        header = self.text_renderer.collapse_route_parenthetical_duplicates(header)

        if not header.strip():
            fallback_header = header_hint or derive_header_from_text(instruction)
            header = self.text_renderer.format_route_bullets(
                " ".join((fallback_header or "").split()).strip(),
                route_ids_for_text,
            )
        if not header:
            raise ValueError("Unable to derive a non-empty header from the request.")

        if description:
            description = self.text_renderer.replace_stop_ids_with_names(description, informed_entities)
            description = self.text_renderer.format_route_bullets(description, route_ids_for_text)
            description = self.text_renderer.collapse_route_parenthetical_duplicates(description)

        alert_id = resolve_alert_id(header, description)

        cause_out = normalize_cause(cause_effect.cause or UNKNOWN_CAUSE)
        effect_out = normalize_effect(cause_effect.effect or UNKNOWN_EFFECT)
        print("[DEBUG] Phase 8: Mercury Resolution...")
        mercury_selection = self.mercury_resolver.resolve(
            instruction=instruction,
            header_text=header.strip(),
            description_text=description.strip() if description else None,
            cause=cause_out,
            effect=effect_out,
            active_periods=active_periods,
            informed_entities=informed_entities,
        )
        compile_report["stages"].append({
            "stage": "mercury_resolution",
            "source": "llm" if mercury_selection.confidence >= self.enum_confidence_threshold else "deterministic",
            "details": (
                f"Mercury priority came from the LLM classifier with confidence {mercury_selection.confidence:.2f}."
                if mercury_selection.confidence >= self.enum_confidence_threshold
                else "Mercury priority used the deterministic fallback category."
            ),
        })
        informed_entities = self.mercury_resolver.annotate_entities(
            informed_entities=informed_entities,
            priority_number=mercury_selection.priority_number,
        )
        mercury_alert = self.mercury_resolver.build_mercury_alert(
            active_periods=active_periods,
            compiled_at_posix=compiled_at_posix,
            selection=mercury_selection,
            no_timeframe_mentioned=no_timeframe_mentioned,
        )

        print("[DEBUG] Phase 9: Payload Building...")
        payload = self.output_builder.build_payload(
            alert_id=alert_id,
            active_periods=active_periods,
            informed_entities=informed_entities,
            cause=cause_out,
            effect=effect_out,
            header_text=header.strip(),
            description_text=description.strip() if description else None,
            mercury_alert=mercury_alert,
        )
        compile_report["stages"].append({
            "stage": "payload_building",
            **self.output_builder.last_variants_report,
        })
        self.last_compile_report = compile_report
        return payload

    def _ensure_llm(self) -> bool:
        if self.llm is not None:
            return True
        try:
            self.llm = build_langchain_chat_model(config=self._llm_config_active, temperature=0.0)
            if self.llm is None:
                raise RuntimeError(
                    "LLM initialization returned None. Check your provider config and API keys."
                )
            return True
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize LLM ({self._llm_config_active.provider}): {exc}"
            ) from exc

    def _set_request_llm_config(
        self,
        provider: Optional[str],
        model: Optional[str],
        reasoning_effort: Optional[str],
    ) -> None:
        next_config = with_overrides(
            self._llm_config,
            provider=provider,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        if next_config != self._llm_config_active:
            self._llm_config_active = next_config
            self.llm = None

    def _bump_telemetry(self, key: str, amount: int = 1) -> None:
        if key not in self.telemetry:
            self.telemetry[key] = 0
        self.telemetry[key] += max(1, int(amount))
