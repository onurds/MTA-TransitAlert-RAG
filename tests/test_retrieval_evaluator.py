from __future__ import annotations

from pipeline.compiler.evidence import decompose_instruction
from pipeline.compiler.retrieval_evaluator import RetrievalEvaluator


def test_retrieval_evaluator_accepts_strong_route_grounding_without_stop_requirement():
    evaluator = RetrievalEvaluator()
    retrieval = {
        "location_hints": [],
        "matched_stop_count": 0,
        "fallback_needed": False,
    }

    result = evaluator.evaluate(
        retrieval=retrieval,
        evidence_units=decompose_instruction("A trains are running with delays."),
        route_confidence=0.91,
        stop_confidence=0.95,
        temporal_override=False,
    )

    assert result.state == "ACCEPT"
    assert result.trigger_reason == "route_grounding_strong_no_stop_required"


def test_retrieval_evaluator_marks_ambiguous_when_route_is_acceptable_but_stop_grounding_is_weak():
    evaluator = RetrievalEvaluator()
    retrieval = {
        "location_hints": ["Decatur St", "Wilson Ave"],
        "matched_stop_count": 1,
        "fallback_needed": False,
    }

    result = evaluator.evaluate(
        retrieval=retrieval,
        evidence_units=decompose_instruction("B20 buses are detoured at Decatur St at Wilson Ave."),
        route_confidence=0.6,
        stop_confidence=0.4,
        temporal_override=False,
    )

    assert result.state == "AMBIGUOUS"
    assert result.trigger_reason == "route_grounding_acceptable_stop_grounding_weak"


def test_retrieval_evaluator_triggers_corrective_fallback_when_stop_intent_has_no_grounded_stop():
    evaluator = RetrievalEvaluator()
    retrieval = {
        "location_hints": ["Cross Bay Blvd", "Liberty Ave"],
        "matched_stop_count": 0,
        "fallback_needed": True,
    }

    result = evaluator.evaluate(
        retrieval=retrieval,
        evidence_units=decompose_instruction(
            "Q52-SBS buses will not stop on Cross Bay Blvd at Liberty Ave."
        ),
        route_confidence=0.83,
        stop_confidence=0.2,
        temporal_override=True,
    )

    assert result.state == "CORRECTIVE_FALLBACK"
    assert result.trigger_reason == "stop_intent_without_grounded_stop"
