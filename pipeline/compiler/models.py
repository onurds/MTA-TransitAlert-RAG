from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from pipeline.gtfs_rules import UNKNOWN_CAUSE, UNKNOWN_EFFECT

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


class CompileRequest(BaseModel):
    instruction: str = ""
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


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
