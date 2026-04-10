from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from pipeline.gtfs_rules import UNKNOWN_CAUSE, UNKNOWN_EFFECT

MULTI_LANG_CODES = ("zh", "es")


class InformedEntity(BaseModel):
    agency_id: str
    route_id: Optional[str] = None
    stop_id: Optional[str] = None
    mercury_entity_selector: Optional["MercuryEntitySelector"] = None

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


class MercuryEntitySelector(BaseModel):
    sort_order: str


class MercuryAlert(BaseModel):
    created_at: str
    updated_at: str
    alert_type: str
    display_before_active: str
    human_readable_active_period: TranslatedString


class CompileRequest(BaseModel):
    instruction: str = ""
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh"]] = None
    text_mode: Literal["default", "rewrite"] = "default"


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


@dataclass(frozen=True)
class CauseEffectResult:
    cause: str = UNKNOWN_CAUSE
    effect: str = UNKNOWN_EFFECT
    cause_confidence: float = 0.0
    effect_confidence: float = 0.0


EvidenceUnitType = Literal[
    "affected_service",
    "alternative_service",
    "temporal_directive",
    "operator_control",
    "rider_guidance",
    "location_evidence",
]

RetrievalState = Literal["ACCEPT", "AMBIGUOUS", "CORRECTIVE_FALLBACK"]


@dataclass(frozen=True)
class EvidenceUnit:
    unit_type: EvidenceUnitType
    text: str
    source: str = "instruction"


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    state: RetrievalState
    score: float
    route_confidence: float
    stop_confidence: float
    location_hint_quality: float
    evidence_agreement: float
    temporal_bonus: float
    trigger_reason: str
    location_hint_count: int = 0
    matched_stop_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "score": round(float(self.score), 3),
            "route_confidence": round(float(self.route_confidence), 3),
            "stop_confidence": round(float(self.stop_confidence), 3),
            "location_hint_quality": round(float(self.location_hint_quality), 3),
            "evidence_agreement": round(float(self.evidence_agreement), 3),
            "temporal_bonus": round(float(self.temporal_bonus), 3),
            "trigger_reason": self.trigger_reason,
            "location_hint_count": int(self.location_hint_count),
            "matched_stop_count": int(self.matched_stop_count),
        }
