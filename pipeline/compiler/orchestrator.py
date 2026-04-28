from __future__ import annotations

import json
import threading
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

from pipeline.codex_cli_runner import CodexCliInvocationError, CodexCliRunner
from pipeline.graph import GraphRetriever
from pipeline.gtfs_rules import CAUSE_ENUMS, EFFECT_ENUMS, UNKNOWN_CAUSE, UNKNOWN_EFFECT, normalize_cause, normalize_effect
from pipeline.llm_config import build_langchain_chat_model, load_llm_config, with_overrides
from pipeline.temporal_resolver import TemporalResolver

from .confidence import global_confidence, should_preserve_stops_under_low_confidence
from .evidence import (
    command_stripped_instruction,
    decompose_instruction,
    evidence_text,
    summarize_evidence_units,
)
from .entity_selector import EntitySelector
from .enum_resolver import EnumResolver
from .intent_parser import IntentParser
from .mercury_resolver import MercuryResolver
from .models import ActivePeriod, CompileRequest
from .output_builder import OutputBuilder
from .retrieval_evaluator import RetrievalEvaluator
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
        self.baseline_config: Dict[str, Any] = {
            "graph_path": graph_path,
            "calendar_path": calendar_path,
            "timezone": timezone,
            "confidence_threshold": confidence_threshold,
            "enum_confidence_threshold": enum_confidence_threshold,
        }

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
        self._compile_report_lock = threading.Lock()
        self._compile_reports_by_request: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._compile_report_limit = 2048

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
        self.retrieval_evaluator = RetrievalEvaluator()
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
        self._set_request_llm_config(
            request.llm_provider,
            request.llm_model,
            request.llm_reasoning_effort,
        )
        provider = request.llm_provider or self._llm_config_active.provider
        model = self._resolve_model_name(provider, request.llm_model)
        reasoning_effort = self._resolve_reasoning_effort(provider, request.llm_reasoning_effort)
        compile_report: Dict[str, Any] = {
            "provider": provider,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "baseline_config": dict(self.baseline_config),
            "request": {
                "request_id": request.request_id,
                "text_mode": request.text_mode,
                "reference_time": request.reference_time,
            },
            "stages": [],
        }

        instruction = (request.instruction or "").strip()
        if not instruction:
            raise ValueError("instruction is required and cannot be empty.")
        compile_report["request"]["instruction_preview"] = instruction[:200]

        if provider == "codex_cli":
            return self._compile_codex_cli(
                request=request,
                instruction=instruction,
                compile_report=compile_report,
                model=model,
                reasoning_effort=reasoning_effort,
            )

        evidence_units = decompose_instruction(instruction)
        evidence_summary = summarize_evidence_units(evidence_units)
        affected_source_text = evidence_text(evidence_units, ("affected_service", "location_evidence")) or instruction
        temporal_source_text = evidence_text(evidence_units, ("temporal_directive", "affected_service")) or instruction
        rider_source_text = command_stripped_instruction(evidence_units) or instruction
        compile_report["stages"].append({
            "stage": "evidence_decomposition",
            "source": "deterministic",
            "inputs": {"instruction_length": len(instruction)},
            "outputs": {"evidence_units": evidence_summary},
            "scores": {"unit_count": len(evidence_units)},
            "branch_decision": "typed_evidence_units_ready",
            "details": "Instruction was segmented into typed evidence units before retrieval and text generation.",
        })

        print("[DEBUG] Phase 1: Intent Parsing...")
        intent = self.intent_parser.parse(instruction)
        print(f"[DEBUG] Intent parsed: {intent.parse_confidence}")
        compile_report["stages"].append({
            "stage": "intent_parsing",
            "source": self.intent_parser.last_parse_report.get("source", "llm"),
            "inputs": {"instruction": instruction},
            "outputs": {
                "alert_text": intent.alert_text,
                "temporal_text": intent.temporal_text,
                "explicit_route_ids": list(intent.explicit_route_ids),
                "explicit_stop_ids": list(intent.explicit_stop_ids),
                "location_phrases": list(intent.location_phrases),
                "effect_hint": intent.effect_hint,
                "cause_hint": intent.cause_hint,
            },
            "scores": {"parse_confidence": intent.parse_confidence},
            "branch_decision": "llm_structured_intent",
            "repair_used": self.intent_parser.last_parse_report.get("repair_used", False),
            "details": f"LLM extracted structured intent with confidence {intent.parse_confidence:.2f}.",
        })

        header_hint = intent.alert_text or None
        header = header_hint or derive_header_from_text(instruction)
        header = " ".join((header or "").split())
        if not header:
            header = derive_header_from_text(instruction)
        if not header:
            raise ValueError("Unable to derive a non-empty header from the request.")

        temporal_text = intent.temporal_text or temporal_source_text or header
        temporal_override = bool(intent.temporal_text or has_temporal_hint(instruction))
        temporal_confidence = 0.4
        active_periods: List[Dict[str, Any]] = []
        reference_dt = self._parse_reference_time(request.reference_time)
        compiled_at_posix = int(reference_dt.timestamp())

        resolved_periods: List[Dict[str, Any]] = []
        if self.temporal_resolver:
            print("[DEBUG] Phase 2: Temporal Resolution...")
            llm_periods = self.temporal_selector.resolve_periods(
                llm=self.llm,
                temporal_text=temporal_text,
                full_instruction=temporal_source_text,
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
            "source": self.temporal_selector.last_resolution_report.get("source", "llm" if resolved_periods else "deterministic"),
            "inputs": {
                "temporal_text": temporal_text,
                "reference_dt": reference_dt.isoformat(),
            },
            "outputs": {"active_periods": active_periods},
            "scores": {
                "temporal_confidence": temporal_confidence,
                "period_count": len(resolved_periods),
            },
            "repair_used": self.temporal_selector.last_resolution_report.get("repair_used", False),
            "branch_decision": "resolved_periods" if resolved_periods else "immediate_start_fallback",
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
        evidence_location_hints = [
            u.text
            for u in evidence_units
            if u.unit_type == "location_evidence"
        ]
        affected_hints = [
            u.text
            for u in evidence_units
            if u.unit_type == "location_evidence" and u.source == "affected_service"
        ] or evidence_location_hints
        alternative_hints = {
            str(u.text).strip().lower()
            for u in evidence_units
            if u.unit_type == "location_evidence" and u.source == "alternative_service"
        }
        intent_hints = [
            h
            for h in intent.location_phrases
            if str(h).strip().lower() not in alternative_hints and not self.retriever.is_route_name_phrase(str(h))
        ]
        if not affected_hints and not self.retriever._has_strong_stop_intent(affected_source_text):
            intent_hints = []
        location_hints_override = merge_text_tokens(affected_hints, intent_hints)

        retrieval = self.retriever.retrieve_affected_entities(
            affected_source_text,
            seed_entities=seed_entities,
            route_ids_override=explicit_route_ids or None,
            location_hints_override=location_hints_override or None,
            alternative_hints_override=intent.alternative_service_text or None,
            max_stop_candidates=20,
        )
        high_level_context = retrieval.get("high_level_context", {}) if isinstance(retrieval, dict) else {}
        compile_report["stages"].append({
            "stage": "entity_retrieval",
            "source": "deterministic",
            "inputs": {
                "retrieval_text": affected_source_text,
                "location_hints_override": location_hints_override,
                "seed_entities": seed_entities,
            },
            "outputs": {
                "route_ids": retrieval.get("route_ids", []),
                "location_hints": retrieval.get("location_hints", []),
                "matched_stop_count": retrieval.get("matched_stop_count", 0),
                "high_level_context": high_level_context,
            },
            "scores": {
                "route_confidence": retrieval.get("route_confidence", 0.0),
                "stop_confidence": retrieval.get("stop_confidence", 0.0),
            },
            "branch_decision": "graph_grounding_attempted",
            "details": (
                f"Graph retrieval produced {len(retrieval.get('informed_entities', [])) if retrieval.get('status') == 'success' else 0} "
                "candidate informed entities."
            ),
        })

        retrieved_entities = retrieval.get("informed_entities", []) if retrieval.get("status") == "success" else []
        inferred_route_entities = [e for e in retrieved_entities if e.get("route_id") and not e.get("stop_id")]
        inferred_stop_entities = [e for e in retrieved_entities if e.get("stop_id")]
        stop_candidates = retrieval.get("stop_candidates", []) if retrieval.get("status") == "success" else []
        route_conf = float(retrieval.get("route_confidence", 0.0))
        stop_conf = float(retrieval.get("stop_confidence", 0.0))
        if explicit_route_ids:
            route_conf = max(route_conf, 0.95)
        if explicit_stop_ids:
            stop_conf = max(stop_conf, 1.0)

        retrieval_eval = self.retrieval_evaluator.evaluate(
            retrieval=retrieval,
            evidence_units=evidence_units,
            route_confidence=route_conf,
            stop_confidence=stop_conf,
            temporal_override=temporal_override,
        )
        compile_report["stages"].append({
            "stage": "retrieval_evaluation",
            "source": "deterministic",
            "inputs": {
                "location_hints": retrieval.get("location_hints", []),
                "matched_stop_count": retrieval.get("matched_stop_count", 0),
                "has_stop_intent": retrieval.get("has_stop_intent", False),
            },
            "outputs": retrieval_eval.as_dict(),
            "scores": retrieval_eval.as_dict(),
            "branch_decision": retrieval_eval.state,
            "details": "Retrieval branch was chosen from explicit graph-grounding quality signals.",
        })

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
                text="\n".join([header, affected_source_text]).strip(),
                allowed_route_ids=allowed_route_ids,
                stop_candidates=stop_candidates,
                locked_route_ids=explicit_route_ids,
                locked_stop_ids=explicit_stop_ids,
                location_hints=retrieval.get("location_hints", []),
                high_level_context=high_level_context,
            )
        compile_report["stages"].append({
            "stage": "entity_selection",
            "source": self.entity_selector.last_selection_report.get("source", "llm" if (allowed_route_ids or stop_candidates) else "deterministic"),
            "inputs": {
                "allowed_route_ids": allowed_route_ids,
                "locked_route_ids": list(explicit_route_ids),
                "locked_stop_ids": list(explicit_stop_ids),
            },
            "outputs": {
                "selected_route_ids": llm_route_ids,
                "selected_stop_ids": llm_stop_ids,
            },
            "scores": {"entity_selection_confidence": llm_entity_conf},
            "repair_used": self.entity_selector.last_selection_report.get("repair_used", False),
            "branch_decision": "bounded_llm_selection" if (allowed_route_ids or stop_candidates) else "deterministic_only",
            "details": (
                f"LLM selected {len(llm_route_ids)} route(s) and {len(llm_stop_ids)} stop(s) with confidence {llm_entity_conf:.2f}."
                if (allowed_route_ids or stop_candidates)
                else "Entity selection used deterministic retrieval only because there were no candidates to rank."
            ),
        })

        if llm_stop_ids and llm_entity_conf >= 0.65 and retrieval_eval.state != "CORRECTIVE_FALLBACK":
            self._bump_telemetry("inferred_stop_accept", len(llm_stop_ids))

        locked_stop_ids = {sid.upper() for sid in explicit_stop_ids}
        if locked_stop_ids:
            inferred_stop_entities = [
                e for e in inferred_stop_entities if str(e.get("stop_id", "")).upper() in locked_stop_ids
            ]
        elif llm_stop_ids and llm_entity_conf >= 0.65 and retrieval_eval.state != "CORRECTIVE_FALLBACK":
            inferred_stop_entities = [build_stop_entity(self.retriever, sid) for sid in llm_stop_ids]

        inferred_stop_entities = self.entity_selector.choose_stops_for_single_location(
            stop_entities=inferred_stop_entities,
            source_text="\n".join([header, affected_source_text]),
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
            "\n".join([header, affected_source_text]),
        )

        if explicit_stop_ids:
            stop_conf = max(stop_conf, 1.0)
        elif llm_entity_conf >= 0.65 and llm_stop_ids and retrieval_eval.state != "CORRECTIVE_FALLBACK":
            stop_conf = max(stop_conf, llm_entity_conf)
        if not route_conf and informed_entities:
            route_conf = 0.75

        print("[DEBUG] Phase 5: Enum Resolution...")
        cause_effect = self.enum_resolver.resolve(
            text="\n".join([header, rider_source_text]).strip(),
            cause_override=intent.cause_hint,
            effect_override=intent.effect_hint,
        )
        compile_report["stages"].append({
            "stage": "enum_resolution",
            "source": self.enum_resolver.last_resolution_report.get("source", "deterministic"),
            "inputs": {
                "text": "\n".join([header, rider_source_text]).strip(),
                "cause_override": intent.cause_hint,
                "effect_override": intent.effect_hint,
            },
            "outputs": {
                "cause": cause_effect.cause,
                "effect": cause_effect.effect,
            },
            "scores": {
                "cause_confidence": cause_effect.cause_confidence,
                "effect_confidence": cause_effect.effect_confidence,
            },
            "repair_used": self.enum_resolver.last_resolution_report.get("repair_used", False),
            "branch_decision": "llm_enum_classifier",
            "details": (
                f"Cause/effect were accepted from the LLM classifier ({cause_effect.cause_confidence:.2f}/{cause_effect.effect_confidence:.2f})."
                if (cause_effect.cause_confidence >= self.enum_confidence_threshold or cause_effect.effect_confidence >= self.enum_confidence_threshold)
                else "Cause/effect fell back to deterministic defaults or low-confidence outputs."
            ),
        })

        schema_conf = 1.0
        confidence = global_confidence(route_conf, stop_conf, temporal_confidence, schema_conf)

        should_try_fallback = (
            retrieval_eval.state == "CORRECTIVE_FALLBACK"
            and bool(retrieval.get("has_stop_intent") or retrieval.get("location_hints"))
        )
        fallback_attempted = False
        fallback_used = False
        fallback_success = False
        fallback_conf = 0.0
        fallback_added_count = 0
        fallback_outcome = "unused"
        fallback_trigger_reason = retrieval_eval.trigger_reason if should_try_fallback else "not_needed"
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
                    fallback_outcome = "accepted"
                    self._bump_telemetry("fallback_used")
                    stop_conf = max(stop_conf, float(fallback.get("confidence", 0.0)))
                    confidence = global_confidence(route_conf, stop_conf, temporal_confidence, schema_conf)
                else:
                    fallback_outcome = "attempted_no_change"
            else:
                fallback_outcome = "attempted_no_match"
        compile_report["stages"].append({
            "stage": "geocode_fallback",
            "source": "deterministic",
            "inputs": {
                "location_hints": retrieval.get("location_hints", []),
                "route_ids": [str(e["route_id"]) for e in informed_entities if isinstance(e, dict) and e.get("route_id")],
            },
            "outputs": {
                "fallback_outcome": fallback_outcome,
                "fallback_added_count": fallback_added_count,
            },
            "scores": {
                "fallback_confidence": fallback_conf,
            },
            "branch_decision": fallback_trigger_reason,
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
        if should_try_fallback and fallback_outcome != "accepted" and not explicit_stop_ids:
            informed_entities = conservative_entities(informed_entities)

        preserve_stops = should_preserve_stops_under_low_confidence(
            entities=informed_entities,
            stop_conf=stop_conf,
            retrieval=retrieval,
            fallback_used=fallback_used,
            fallback_conf=fallback_conf,
        )
        if (confidence < self.confidence_threshold or retrieval_eval.state == "CORRECTIVE_FALLBACK") and not preserve_stops and not explicit_stop_ids:
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
            rider_source_override=rider_source_text,
            high_level_context=high_level_context,
        )
        compile_report["stages"].append({
            "stage": "text_mode_resolution",
            "inputs": {
                "route_ids": route_ids_for_text,
                "rider_source": rider_source_text,
            },
            "outputs": {
                "header_text": header_text,
                "description_text": description,
            },
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
            high_level_context=high_level_context,
        )
        compile_report["stages"].append({
            "stage": "mercury_resolution",
            "inputs": {
                "header_text": header.strip(),
                "description_text": description.strip() if description else None,
                "high_level_context": high_level_context,
            },
            "outputs": {
                "priority_key": mercury_selection.priority_key,
                "priority_number": mercury_selection.priority_number,
            },
            "scores": {"confidence": mercury_selection.confidence},
            **self.mercury_resolver.last_resolution_report,
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
            "inputs": {
                "alert_id": alert_id,
                "entity_count": len(informed_entities),
            },
            "outputs": {
                "payload_keys": list(payload.keys()),
            },
            **self.output_builder.last_variants_report,
        })
        compile_report["telemetry"] = {
            "retrieval_state": retrieval_eval.state,
            "fallback_trigger_reason": fallback_trigger_reason,
            "fallback_outcome": fallback_outcome,
            "evidence_units_used": len(evidence_units),
            "schema_repair_used": bool(
                self.intent_parser.last_parse_report.get("repair_used")
                or self.entity_selector.last_selection_report.get("repair_used")
                or self.temporal_selector.last_resolution_report.get("repair_used")
                or self.enum_resolver.last_resolution_report.get("repair_used")
                or self.text_mode_resolver.last_resolution_report.get("repair_used")
                or self.mercury_resolver.last_resolution_report.get("repair_used")
                or self.output_builder.last_variants_report.get("repair_used")
            ),
            "command_strip_applied": rider_source_text != instruction,
            "high_level_context_used": any(bool(v) for v in high_level_context.values()) if isinstance(high_level_context, dict) else False,
        }
        compile_report["final"] = {
            "confidence": round(confidence, 3),
            "route_confidence": round(route_conf, 3),
            "stop_confidence": round(stop_conf, 3),
            "temporal_confidence": round(temporal_confidence, 3),
            "entity_count": len(informed_entities),
            "route_count": len(route_ids_for_text),
        }
        self._store_compile_report(request.request_id, compile_report)
        return payload

    def _compile_codex_cli(
        self,
        request: CompileRequest,
        instruction: str,
        compile_report: Dict[str, Any],
        model: str,
        reasoning_effort: str,
    ) -> Dict[str, Any]:
        compile_report["backend_mode"] = "codex_hybrid"
        runner = CodexCliRunner(config=self._llm_config_active, cwd=".")

        evidence_units = decompose_instruction(instruction)
        evidence_summary = summarize_evidence_units(evidence_units)
        affected_source_text = evidence_text(evidence_units, ("affected_service", "location_evidence")) or instruction
        temporal_source_text = evidence_text(evidence_units, ("temporal_directive", "affected_service")) or instruction
        rider_source_text = command_stripped_instruction(evidence_units) or instruction
        compile_report["stages"].append({
            "stage": "evidence_decomposition",
            "source": "deterministic",
            "inputs": {"instruction_length": len(instruction)},
            "outputs": {"evidence_units": evidence_summary},
            "scores": {"unit_count": len(evidence_units)},
            "branch_decision": "typed_evidence_units_ready",
            "details": "Instruction was segmented into typed evidence units before Codex extraction and deterministic grounding.",
        })

        reference_dt = self._parse_reference_time(request.reference_time)
        compiled_at_posix = int(reference_dt.timestamp())
        header_hint = derive_header_from_text(instruction)
        temporal_override = has_temporal_hint(instruction)

        explicit_route_ids = self.retriever.validate_route_ids(self.intent_parser._extract_explicit_route_ids(instruction))
        raw_explicit_stop_ids = self.intent_parser._extract_explicit_stop_ids(instruction)
        explicit_stop_ids = self.retriever.validate_stop_ids(raw_explicit_stop_ids)
        dropped_explicit = max(0, len(raw_explicit_stop_ids) - len(explicit_stop_ids))
        if dropped_explicit:
            self._bump_telemetry("explicit_id_invalid_drop", dropped_explicit)

        explicit_route_entities = [build_route_entity(self.retriever, rid) for rid in explicit_route_ids]
        explicit_stop_entities = [build_stop_entity(self.retriever, sid) for sid in explicit_stop_ids]
        seed_entities = dedupe_entities(explicit_route_entities + explicit_stop_entities)

        evidence_location_hints = [u.text for u in evidence_units if u.unit_type == "location_evidence"]
        affected_hints = [
            u.text for u in evidence_units if u.unit_type == "location_evidence" and u.source == "affected_service"
        ] or evidence_location_hints
        location_hints_override = merge_text_tokens(affected_hints)

        retrieval = self.retriever.retrieve_affected_entities(
            affected_source_text,
            seed_entities=seed_entities,
            route_ids_override=explicit_route_ids or None,
            location_hints_override=location_hints_override or None,
            max_stop_candidates=20,
        )
        high_level_context = retrieval.get("high_level_context", {}) if isinstance(retrieval, dict) else {}
        compile_report["stages"].append({
            "stage": "entity_retrieval",
            "source": "deterministic",
            "inputs": {
                "retrieval_text": affected_source_text,
                "location_hints_override": location_hints_override,
                "seed_entities": seed_entities,
            },
            "outputs": {
                "route_ids": retrieval.get("route_ids", []),
                "location_hints": retrieval.get("location_hints", []),
                "matched_stop_count": retrieval.get("matched_stop_count", 0),
                "high_level_context": high_level_context,
            },
            "scores": {
                "route_confidence": retrieval.get("route_confidence", 0.0),
                "stop_confidence": retrieval.get("stop_confidence", 0.0),
            },
            "branch_decision": "graph_grounding_attempted",
            "details": (
                f"Graph retrieval produced {len(retrieval.get('informed_entities', [])) if retrieval.get('status') == 'success' else 0} candidate informed entities."
            ),
        })

        retrieved_entities = retrieval.get("informed_entities", []) if retrieval.get("status") == "success" else []
        inferred_route_entities = [e for e in retrieved_entities if e.get("route_id") and not e.get("stop_id")]
        stop_candidates = retrieval.get("stop_candidates", []) if retrieval.get("status") == "success" else []
        route_conf = float(retrieval.get("route_confidence", 0.0))
        stop_conf = float(retrieval.get("stop_confidence", 0.0))
        if explicit_route_ids:
            route_conf = max(route_conf, 0.95)
        if explicit_stop_ids:
            stop_conf = max(stop_conf, 1.0)

        retrieval_eval = self.retrieval_evaluator.evaluate(
            retrieval=retrieval,
            evidence_units=evidence_units,
            route_confidence=route_conf,
            stop_confidence=stop_conf,
            temporal_override=temporal_override,
        )
        compile_report["stages"].append({
            "stage": "retrieval_evaluation",
            "source": "deterministic",
            "inputs": {
                "location_hints": retrieval.get("location_hints", []),
                "matched_stop_count": retrieval.get("matched_stop_count", 0),
                "has_stop_intent": retrieval.get("has_stop_intent", False),
            },
            "outputs": retrieval_eval.as_dict(),
            "scores": retrieval_eval.as_dict(),
            "branch_decision": retrieval_eval.state,
            "details": "Retrieval branch was chosen from explicit graph-grounding quality signals.",
        })

        allowed_route_ids = merge_unique_tokens(
            explicit_route_ids,
            retrieval.get("route_ids", []),
            [e.get("route_id") for e in inferred_route_entities if e.get("route_id")],
        )

        try:
            core_result = runner.run_json_task(
                task_name="core_extraction",
                prompt=self._build_codex_core_prompt(
                    instruction=instruction,
                    reference_dt=reference_dt,
                    evidence_summary=evidence_summary,
                    allowed_route_ids=allowed_route_ids,
                    stop_candidates=stop_candidates,
                    high_level_context=high_level_context,
                ),
                schema=self._codex_core_schema(),
                model=model,
                reasoning_effort=reasoning_effort,
            )
        except CodexCliInvocationError as exc:
            compile_report["stages"].append({
                "stage": "codex_core_extraction",
                "source": "codex_cli",
                "inputs": {
                    "allowed_route_ids": allowed_route_ids,
                    "stop_candidate_count": len(stop_candidates),
                },
                "outputs": {"error_category": exc.category},
                "scores": {},
                "branch_decision": exc.category,
                "details": str(exc),
            })
            compile_report["telemetry"] = {
                "backend_mode": "codex_hybrid",
                "codex_calls": 1,
                "codex_failure_category": exc.category,
            }
            self._store_compile_report(request.request_id, compile_report)
            raise RuntimeError(f"Codex CLI core_extraction failed ({exc.category}): {exc}") from exc

        core_payload = core_result.payload
        selected_route_ids = [rid for rid in self._normalize_token_list(core_payload.get("selected_route_ids")) if rid in set(allowed_route_ids)]
        allowed_stop_ids = {str(c.get("stop_id", "")).strip().upper() for c in stop_candidates if c.get("stop_id")}
        allowed_stop_ids.update(str(sid).strip().upper() for sid in explicit_stop_ids)
        selected_stop_ids = [sid for sid in self._normalize_token_list(core_payload.get("selected_stop_ids")) if sid in allowed_stop_ids]
        active_periods = self._normalize_active_periods(core_payload.get("active_periods"))
        core_confidence = self._coerce_confidence(core_payload.get("confidence", 0.0))
        clean_rider_source = str(core_payload.get("clean_rider_source") or "").strip() or rider_source_text
        cause_out = normalize_cause(str(core_payload.get("cause") or UNKNOWN_CAUSE))
        effect_out = normalize_effect(str(core_payload.get("effect") or UNKNOWN_EFFECT))
        mercury_key = str(core_payload.get("mercury_priority_key") or "").strip()
        mercury_selection = self.mercury_resolver.selection_from_key(mercury_key, core_confidence)
        compile_report["stages"].append({
            "stage": "codex_core_extraction",
            "source": "codex_cli",
            "inputs": {
                "allowed_route_ids": allowed_route_ids,
                "stop_candidate_count": len(stop_candidates),
                "reference_dt": reference_dt.isoformat(),
            },
            "outputs": {
                "selected_route_ids": selected_route_ids,
                "selected_stop_ids": selected_stop_ids,
                "active_periods": active_periods,
                "cause": cause_out,
                "effect": effect_out,
                "mercury_priority_key": mercury_selection.priority_key,
                "clean_rider_source_length": len(clean_rider_source),
            },
            "scores": {"confidence": core_confidence, **core_result.usage.as_dict()},
            "branch_decision": "codex_structured_extraction",
            "details": "Codex produced the structured alert interpretation in one bounded extraction call.",
        })

        route_entities = dedupe_entities(
            explicit_route_entities + inferred_route_entities + [build_route_entity(self.retriever, rid) for rid in selected_route_ids]
        )
        stop_entities = dedupe_entities(explicit_stop_entities + [build_stop_entity(self.retriever, sid) for sid in selected_stop_ids])
        stop_entities = self.entity_selector.choose_stops_for_single_location(
            stop_entities=stop_entities,
            source_text="\n".join([header_hint, affected_source_text]).strip(),
            stop_candidates=stop_candidates,
            location_hints=retrieval.get("location_hints", []),
        )
        informed_entities = normalize_entities_for_output(
            dedupe_entities(route_entities + stop_entities),
            "\n".join([header_hint, affected_source_text]),
        )

        if selected_route_ids:
            route_conf = max(route_conf, core_confidence)
        if selected_stop_ids:
            stop_conf = max(stop_conf, core_confidence)

        should_try_fallback = (
            retrieval_eval.state == "CORRECTIVE_FALLBACK"
            and bool(retrieval.get("has_stop_intent") or retrieval.get("location_hints"))
        )
        fallback_attempted = False
        fallback_used = False
        fallback_success = False
        fallback_conf = 0.0
        fallback_added_count = 0
        fallback_outcome = "unused"
        fallback_trigger_reason = retrieval_eval.trigger_reason if should_try_fallback else "not_needed"
        if should_try_fallback:
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
                    locked_stop_ids = {sid.upper() for sid in explicit_stop_ids}
                    fallback_entities = [
                        e for e in fallback_entities if str(e.get("stop_id", "")).upper() in locked_stop_ids
                    ]
                before_count = len(informed_entities)
                informed_entities = dedupe_entities(informed_entities + fallback_entities)
                fallback_added_count = max(0, len(informed_entities) - before_count)
                if fallback_added_count > 0:
                    fallback_used = True
                    fallback_outcome = "accepted"
                    self._bump_telemetry("fallback_used")
                    stop_conf = max(stop_conf, fallback_conf)
                else:
                    fallback_outcome = "attempted_no_change"
            else:
                fallback_outcome = "attempted_no_match"
        compile_report["stages"].append({
            "stage": "geocode_fallback",
            "source": "deterministic",
            "inputs": {
                "location_hints": retrieval.get("location_hints", []),
                "route_ids": [str(e["route_id"]) for e in informed_entities if isinstance(e, dict) and e.get("route_id")],
            },
            "outputs": {
                "fallback_outcome": fallback_outcome,
                "fallback_added_count": fallback_added_count,
            },
            "scores": {"fallback_confidence": fallback_conf},
            "branch_decision": fallback_trigger_reason,
            "details": (
                f"External geocode fallback added {fallback_added_count} new {'entity' if fallback_added_count == 1 else 'entities'} with confidence {fallback_conf:.2f}."
                if fallback_used
                else (
                    "External geocode fallback was attempted but did not change the final entities."
                    if fallback_attempted
                    else "External geocode fallback was not needed."
                )
            ),
        })
        if should_try_fallback and fallback_outcome != "accepted" and not explicit_stop_ids:
            informed_entities = conservative_entities(informed_entities)

        preserve_stops = should_preserve_stops_under_low_confidence(
            entities=informed_entities,
            stop_conf=stop_conf,
            retrieval=retrieval,
            fallback_used=fallback_used,
            fallback_conf=fallback_conf,
        )
        if (core_confidence < self.confidence_threshold or retrieval_eval.state == "CORRECTIVE_FALLBACK") and not preserve_stops and not explicit_stop_ids:
            informed_entities = conservative_entities(informed_entities)

        route_ids_for_text: List[str] = [
            str(e["route_id"]) for e in informed_entities if isinstance(e, dict) and e.get("route_id")
        ]

        text_call_count = 0
        try:
            text_result = runner.run_json_task(
                task_name="english_text",
                prompt=self._build_codex_text_prompt(
                    instruction=instruction,
                    clean_rider_source=clean_rider_source,
                    route_ids=route_ids_for_text,
                    cause=cause_out,
                    effect=effect_out,
                    text_mode=request.text_mode,
                    header_hint=header_hint if request.text_mode == "rewrite" else None,
                    high_level_context=high_level_context,
                ),
                schema=self._codex_text_schema(),
                model=model,
                reasoning_effort=reasoning_effort,
            )
            text_call_count = 1
            header_text = str(text_result.payload.get("header_text") or "").strip()
            description = str(text_result.payload.get("description_text") or "").strip() or None
            compile_report["stages"].append({
                "stage": "codex_english_text",
                "source": "codex_cli",
                "inputs": {"route_ids": route_ids_for_text, "text_mode": request.text_mode},
                "outputs": {"header_text": header_text, "description_text": description},
                "scores": {"confidence": self._coerce_confidence(text_result.payload.get("confidence", 0.0)), **text_result.usage.as_dict()},
                "branch_decision": "codex_text_layout",
                "details": "Codex produced the English header/description pair in a dedicated second call.",
            })
        except CodexCliInvocationError as exc:
            header_text, description = self.text_mode_resolver.resolve(
                llm=None,
                instruction=instruction,
                route_ids=route_ids_for_text,
                cause=cause_out,
                effect=effect_out,
                text_mode=request.text_mode,
                header_hint=header_hint if request.text_mode == "rewrite" else None,
                rider_source_override=clean_rider_source,
                high_level_context=high_level_context,
            )
            compile_report["stages"].append({
                "stage": "codex_english_text",
                "source": "deterministic",
                "inputs": {"route_ids": route_ids_for_text, "text_mode": request.text_mode},
                "outputs": {"error_category": exc.category, "header_text": header_text, "description_text": description},
                "scores": {},
                "branch_decision": "deterministic_fallback",
                "details": f"Codex English text generation failed ({exc.category}); deterministic fallback was used.",
            })

        header = self.text_renderer.replace_stop_ids_with_names(header_text, informed_entities)
        header = self.text_renderer.format_route_bullets(header, route_ids_for_text)
        header = self.text_renderer.collapse_route_parenthetical_duplicates(header)
        if not header.strip():
            header = self.text_renderer.format_route_bullets(header_hint, route_ids_for_text)
        if description:
            description = self.text_renderer.replace_stop_ids_with_names(description, informed_entities)
            description = self.text_renderer.format_route_bullets(description, route_ids_for_text)
            description = self.text_renderer.collapse_route_parenthetical_duplicates(description)

        alert_id = resolve_alert_id(header, description)
        no_timeframe_mentioned = not temporal_override and not active_periods
        mercury_alert = self.mercury_resolver.build_mercury_alert(
            active_periods=active_periods,
            compiled_at_posix=compiled_at_posix,
            selection=mercury_selection,
            no_timeframe_mentioned=no_timeframe_mentioned,
        )
        informed_entities = self.mercury_resolver.annotate_entities(
            informed_entities=informed_entities,
            priority_number=mercury_selection.priority_number,
        )
        compile_report["stages"].append({
            "stage": "mercury_resolution",
            "source": "codex_cli",
            "inputs": {
                "priority_key": mercury_selection.priority_key,
                "priority_number": mercury_selection.priority_number,
            },
            "outputs": {
                "alert_type": mercury_selection.alert_type,
            },
            "scores": {"confidence": mercury_selection.confidence},
            "branch_decision": "codex_priority_key",
            "details": "Codex selected the Mercury priority key and deterministic Mercury builders produced the final alert metadata.",
        })
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
            "source": "deterministic",
            "inputs": {
                "alert_id": alert_id,
                "entity_count": len(informed_entities),
            },
            "outputs": {
                "payload_keys": list(payload.keys()),
            },
            "branch_decision": "deterministic_output_builder",
            "details": "Final payload assembly reused the existing deterministic output builder with English-only model text and deterministic zh/es/TTS fallbacks.",
        })

        temporal_confidence = 0.98 if active_periods else 0.4
        confidence = global_confidence(route_conf, stop_conf, temporal_confidence, 1.0)
        compile_report["telemetry"] = {
            "backend_mode": "codex_hybrid",
            "retrieval_state": retrieval_eval.state,
            "fallback_trigger_reason": fallback_trigger_reason,
            "fallback_outcome": fallback_outcome,
            "evidence_units_used": len(evidence_units),
            "command_strip_applied": clean_rider_source != instruction,
            "high_level_context_used": any(bool(v) for v in high_level_context.values()) if isinstance(high_level_context, dict) else False,
            "codex_calls": 1 + text_call_count,
            "codex_usage": {
                "core_extraction": core_result.usage.as_dict(),
                "english_text": text_result.usage.as_dict() if text_call_count else None,
            },
        }
        compile_report["final"] = {
            "confidence": round(confidence, 3),
            "route_confidence": round(route_conf, 3),
            "stop_confidence": round(stop_conf, 3),
            "temporal_confidence": round(temporal_confidence, 3),
            "entity_count": len(informed_entities),
            "route_count": len(route_ids_for_text),
        }
        self._store_compile_report(request.request_id, compile_report)
        return payload

    def _store_compile_report(self, request_id: Optional[str], compile_report: Dict[str, Any]) -> None:
        with self._compile_report_lock:
            self.last_compile_report = compile_report
            if not request_id:
                return
            self._compile_reports_by_request[request_id] = compile_report
            self._compile_reports_by_request.move_to_end(request_id)
            while len(self._compile_reports_by_request) > self._compile_report_limit:
                self._compile_reports_by_request.popitem(last=False)

    def get_compile_report(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._compile_report_lock:
            report = self._compile_reports_by_request.get(request_id)
            if report is not None:
                self._compile_reports_by_request.move_to_end(request_id)
            return report

    def _parse_reference_time(self, reference_time: Optional[str]) -> datetime:
        raw = (reference_time or "").strip()
        if not raw:
            return datetime.now(self.tz)
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("reference_time must be an ISO-8601 datetime string.") from exc
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=self.tz)
        return parsed.astimezone(self.tz)

    def _resolve_model_name(self, provider: str, request_model: Optional[str]) -> str:
        if request_model:
            return request_model
        if provider == "openrouter":
            return self._llm_config_active.openrouter_model_name
        if provider == "gemini":
            return self._llm_config_active.gemini_model_name
        if provider == "codex_cli":
            return self._llm_config_active.codex_model_name
        return self._llm_config_active.local_model_name

    def _resolve_reasoning_effort(self, provider: str, request_reasoning: Optional[str]) -> str:
        if request_reasoning:
            return request_reasoning
        if provider == "codex_cli":
            return self._llm_config_active.codex_reasoning_effort
        return self._llm_config_active.openrouter_reasoning_effort

    @staticmethod
    def _normalize_token_list(values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        out: List[str] = []
        seen = set()
        for value in values:
            token = str(value or "").strip().upper()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    @staticmethod
    def _normalize_active_periods(values: Any) -> List[Dict[str, Any]]:
        if not isinstance(values, list):
            return []
        periods: List[Dict[str, Any]] = []
        for row in values:
            if not isinstance(row, dict):
                continue
            try:
                period = ActivePeriod(**row).model_dump(exclude_none=True)
            except Exception:
                continue
            if period.get("start"):
                periods.append(period)
        return periods

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    def _build_codex_core_prompt(
        self,
        instruction: str,
        reference_dt: datetime,
        evidence_summary: Sequence[Dict[str, Any]],
        allowed_route_ids: Sequence[str],
        stop_candidates: Sequence[Dict[str, Any]],
        high_level_context: Dict[str, Any],
    ) -> str:
        allowed_stops = [
            {
                "stop_id": str(c.get("stop_id", "")).upper(),
                "route_id": str(c.get("route_id", "")).upper(),
                "stop_name": str(c.get("stop_name", "")),
                "score": c.get("score", 0.0),
            }
            for c in stop_candidates[:20]
            if c.get("stop_id")
        ]
        return (
            "You are extracting a structured GTFS transit alert compile result.\n"
            "Return only JSON matching the provided schema.\n"
            "Rules:\n"
            "- selected_route_ids must be chosen only from allowed_route_ids.\n"
            "- selected_stop_ids must be chosen only from allowed_stop_candidates[].stop_id.\n"
            "- active_periods must use local ISO timestamps like YYYY-MM-DDTHH:MM:SS.\n"
            "- cause must be one GTFS cause enum.\n"
            "- effect must be one GTFS effect enum.\n"
            "- mercury_priority_key must be one valid Mercury priority key.\n"
            "- clean_rider_source must remove operator authoring commands while preserving rider-facing facts.\n"
            "- If a field is unknown, return [] or the UNKNOWN enum.\n\n"
            f"REFERENCE_LOCAL_TIME: {reference_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"TIMEZONE: {self.tz.key}\n"
            f"INSTRUCTION: {instruction}\n"
            f"EVIDENCE_SUMMARY: {json.dumps(list(evidence_summary), ensure_ascii=False)}\n"
            f"ALLOWED_ROUTE_IDS: {json.dumps(list(allowed_route_ids), ensure_ascii=False)}\n"
            f"ALLOWED_STOP_CANDIDATES: {json.dumps(allowed_stops, ensure_ascii=False)}\n"
            f"HIGH_LEVEL_CONTEXT: {json.dumps(high_level_context or {}, ensure_ascii=False)}\n"
            f"CAUSE_ENUMS: {json.dumps(sorted(CAUSE_ENUMS), ensure_ascii=False)}\n"
            f"EFFECT_ENUMS: {json.dumps(sorted(EFFECT_ENUMS), ensure_ascii=False)}\n"
            f"MERCURY_PRIORITY_KEYS: {json.dumps(sorted(row['priority_key'] for row in self.mercury_resolver.catalog_rows()), ensure_ascii=False)}\n"
        )

    @staticmethod
    def _codex_core_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selected_route_ids": {"type": "array", "items": {"type": "string"}},
                "selected_stop_ids": {"type": "array", "items": {"type": "string"}},
                "active_periods": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "start": {"type": "string"},
                            "end": {"type": ["string", "null"]},
                        },
                        "required": ["start", "end"],
                    },
                },
                "cause": {"type": "string"},
                "effect": {"type": "string"},
                "mercury_priority_key": {"type": "string"},
                "clean_rider_source": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": [
                "selected_route_ids",
                "selected_stop_ids",
                "active_periods",
                "cause",
                "effect",
                "mercury_priority_key",
                "clean_rider_source",
                "confidence",
            ],
        }

    def _build_codex_text_prompt(
        self,
        instruction: str,
        clean_rider_source: str,
        route_ids: Sequence[str],
        cause: str,
        effect: str,
        text_mode: str,
        header_hint: Optional[str],
        high_level_context: Dict[str, Any],
    ) -> str:
        mode_block = (
            "Rewrite the alert into a rider-facing header and description while preserving all facts."
            if text_mode == "rewrite"
            else "Preserve wording and ordering closely while splitting into rider-facing header and description."
        )
        return (
            "You are laying out English transit alert text into header_text and description_text.\n"
            "Return only JSON matching the provided schema.\n"
            "Rules:\n"
            "- Use clean_rider_source as the source of truth.\n"
            "- Keep route tokens unchanged.\n"
            "- Do not invent stops, times, or rider guidance.\n"
            "- description_text may be empty only if the header contains all rider-facing information.\n"
            f"- {mode_block}\n\n"
            f"INSTRUCTION: {instruction}\n"
            f"CLEAN_RIDER_SOURCE: {clean_rider_source}\n"
            f"ROUTE_IDS: {json.dumps(list(route_ids), ensure_ascii=False)}\n"
            f"CAUSE: {cause}\n"
            f"EFFECT: {effect}\n"
            f"HEADER_HINT: {header_hint or ''}\n"
            f"TEXT_MODE: {text_mode}\n"
            f"HIGH_LEVEL_CONTEXT: {json.dumps(high_level_context or {}, ensure_ascii=False)}\n"
        )

    @staticmethod
    def _codex_text_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "header_text": {"type": "string"},
                "description_text": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["header_text", "description_text", "confidence"],
        }

    def _ensure_llm(self) -> bool:
        if self._llm_config_active.provider == "codex_cli":
            return False
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
