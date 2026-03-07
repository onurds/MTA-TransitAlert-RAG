from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .confidence import coerce_confidence


FALLBACK_PRIORITY_KEY = "SERVICE_CHANGE"
FALLBACK_PRIORITY_NUMBER = 22
DISPLAY_BEFORE_ACTIVE = "3600"

MERCURY_PRIORITY_ITEMS: Tuple[Tuple[str, int], ...] = (
    ("NO_SCHEDULED_SERVICE", 1),
    ("INFORMATION_OUTAGE", 2),
    ("STATION_NOTICE", 3),
    ("SPECIAL_NOTICE", 4),
    ("WEEKDAY_SCHEDULE", 5),
    ("WEEKEND_SCHEDULE", 6),
    ("SATURDAY_SCHEDULE", 7),
    ("SUNDAY_SCHEDULE", 8),
    ("EXTRA_SERVICE", 9),
    ("BOARDING_CHANGE", 10),
    ("SPECIAL_SCHEDULE", 11),
    ("EXPECT_DELAYS", 12),
    ("REDUCED_SERVICE", 13),
    ("PLANNED_EXPRESS_TO_LOCAL", 14),
    ("PLANNED_EXTRA_TRANSFER", 15),
    ("PLANNED_STOPS_SKIPPED", 16),
    ("PLANNED_DETOUR", 17),
    ("PLANNED_REROUTE", 18),
    ("PLANNED_SUBSTITUTE_BUSES", 19),
    ("PLANNED_PART_SUSPENDED", 20),
    ("PLANNED_SUSPENDED", 21),
    ("SERVICE_CHANGE", 22),
    ("PLANNED_WORK", 23),
    ("SOME_DELAYS", 24),
    ("EXPRESS_TO_LOCAL", 25),
    ("DELAYS", 26),
    ("CANCELLATIONS", 27),
    ("DELAYS_AND_CANCELLATIONS", 28),
    ("STOPS_SKIPPED", 29),
    ("SEVERE_DELAYS", 30),
    ("DETOUR", 31),
    ("REROUTE", 32),
    ("SUBSTITUTE_BUSES", 33),
    ("PART_SUSPENDED", 34),
    ("SUSPENDED", 35),
)

MERCURY_PRIORITY_BY_KEY: Dict[str, int] = dict(MERCURY_PRIORITY_ITEMS)
MERCURY_KEY_BY_PRIORITY: Dict[int, str] = {value: key for key, value in MERCURY_PRIORITY_ITEMS}


@dataclass(frozen=True)
class MercurySelectionResult:
    priority_key: str
    priority_number: int
    alert_type: str
    confidence: float


class MercuryResolver:
    def __init__(self, ensure_llm, llm_getter, min_confidence: float = 0.6):
        self.ensure_llm = ensure_llm
        self.llm_getter = llm_getter
        self.min_confidence = float(min_confidence)

    def resolve(
        self,
        instruction: str,
        header_text: str,
        description_text: Optional[str],
        cause: str,
        effect: str,
        active_periods: Sequence[Dict[str, Any]],
        informed_entities: Sequence[Dict[str, Any]],
    ) -> MercurySelectionResult:
        fallback = self._fallback_result()
        if not self.ensure_llm():
            return fallback

        llm = self.llm_getter()
        if llm is None:
            return fallback

        prompt = (
            "/no_think\n"
            "You are classifying an MTA transit alert into one Mercury alert priority category.\n"
            "Return strict JSON only with keys: priority_key, priority_number, confidence.\n"
            "Rules:\n"
            "- Choose exactly one category from the provided catalog.\n"
            "- You may return either a valid priority_key or a valid priority_number or both.\n"
            "- Do not invent categories.\n"
            "- If uncertain, return null values and low confidence.\n\n"
            f"INSTRUCTION: {(instruction or '').strip()}\n"
            f"HEADER_TEXT: {(header_text or '').strip()}\n"
            f"DESCRIPTION_TEXT: {(description_text or '').strip()}\n"
            f"CAUSE: {(cause or '').strip()}\n"
            f"EFFECT: {(effect or '').strip()}\n"
            f"ACTIVE_PERIODS: {json.dumps(list(active_periods), ensure_ascii=False)}\n"
            f"INFORMED_ENTITIES: {json.dumps(list(informed_entities), ensure_ascii=False)}\n"
            f"CATEGORY_CATALOG: {json.dumps(self.catalog_rows(), ensure_ascii=False)}\n"
        )

        try:
            response = llm.invoke(prompt)
            content = getattr(response, "content", "") if response is not None else ""
            if not isinstance(content, str):
                content = str(content)
            parsed = self._extract_first_json_object(content)
        except Exception:
            return fallback

        confidence = coerce_confidence(parsed.get("confidence", 0.0))
        if confidence < self.min_confidence:
            return fallback

        priority_key = self._normalize_priority_key(parsed.get("priority_key"))
        priority_number = self._normalize_priority_number(parsed.get("priority_number"))

        if priority_key and priority_key in MERCURY_PRIORITY_BY_KEY:
            return self._result_for_key(priority_key, confidence)
        if priority_number and priority_number in MERCURY_KEY_BY_PRIORITY:
            return self._result_for_key(MERCURY_KEY_BY_PRIORITY[priority_number], confidence)
        return fallback

    @staticmethod
    def catalog_rows() -> List[Dict[str, Any]]:
        return [
            {
                "priority_key": key,
                "priority_number": number,
                "alert_type": MercuryResolver.alert_type_for_key(key),
            }
            for key, number in MERCURY_PRIORITY_ITEMS
        ]

    @staticmethod
    def alert_type_for_key(priority_key: str) -> str:
        key = MercuryResolver._normalize_priority_key(priority_key)
        if not key:
            return "Service Change"
        if key.startswith("PLANNED_"):
            return f"Planned - {MercuryResolver._humanize_key(key[len('PLANNED_'):])}"
        return MercuryResolver._humanize_key(key)

    def annotate_entities(
        self,
        informed_entities: Sequence[Dict[str, Any]],
        priority_number: int,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for entity in informed_entities:
            if not isinstance(entity, dict):
                continue
            row = dict(entity)
            agency_id = str(row.get("agency_id", "")).strip()
            route_id = str(row.get("route_id", "")).strip().upper() if row.get("route_id") else ""
            if agency_id and route_id:
                row["mercury_entity_selector"] = {
                    "sort_order": f"{agency_id}:{route_id}:{int(priority_number)}"
                }
            else:
                row.pop("mercury_entity_selector", None)
            out.append(row)
        return out

    def build_mercury_alert(
        self,
        active_periods: Sequence[Dict[str, Any]],
        compiled_at_posix: int,
        selection: MercurySelectionResult,
        no_timeframe_mentioned: bool = False,
    ) -> Dict[str, Any]:
        return {
            "created_at": str(int(compiled_at_posix)),
            "updated_at": str(int(compiled_at_posix)),
            "alert_type": selection.alert_type,
            "display_before_active": DISPLAY_BEFORE_ACTIVE,
            "human_readable_active_period": {
                "translation": [
                    {
                        "text": self.human_readable_active_period(
                            active_periods,
                            no_timeframe_mentioned=no_timeframe_mentioned,
                        ),
                        "language": "en",
                    }
                ]
            },
        }

    def human_readable_active_period(
        self,
        active_periods: Sequence[Dict[str, Any]],
        no_timeframe_mentioned: bool = False,
    ) -> str:
        if no_timeframe_mentioned:
            return "Until further notice"
        periods = self._parse_periods(active_periods)
        if not periods:
            return ""
        if len(periods) == 1:
            return self._format_compact_period(periods[0][0], periods[0][1])

        recurring = self._format_weekly_recurrence(periods)
        if recurring:
            return recurring

        return "; ".join(self._format_compact_period(start, end) for start, end in periods)

    def _format_weekly_recurrence(self, periods: Sequence[Tuple[datetime, Optional[datetime]]]) -> str:
        normalized: List[Tuple[datetime, datetime]] = []
        for start, end in periods:
            if end is None:
                return ""
            normalized.append((start, end))
        if len(normalized) < 2:
            return ""

        start_weekday = normalized[0][0].weekday()
        start_time = (normalized[0][0].hour, normalized[0][0].minute)
        end_time = (normalized[0][1].hour, normalized[0][1].minute)

        for start, end in normalized:
            if start.weekday() != start_weekday:
                return ""
            if (start.hour, start.minute) != start_time:
                return ""
            if (end.hour, end.minute) != end_time:
                return ""

        sorted_starts = sorted(start for start, _ in normalized)
        week_gaps = [
            (current.date() - previous.date()).days
            for previous, current in zip(sorted_starts, sorted_starts[1:])
        ]
        if not all(gap % 7 == 0 and gap >= 7 for gap in week_gaps):
            return ""

        weekday_name = f"{sorted_starts[0].strftime('%A')}s"
        time_text = f"{self._format_clock(sorted_starts[0])} to {self._format_clock(normalized[0][1])}"
        same_month = all(start.month == sorted_starts[0].month and start.year == sorted_starts[0].year for start in sorted_starts)
        if same_month:
            return f"{weekday_name} in {sorted_starts[0].strftime('%B')} from {time_text}"

        return (
            f"{weekday_name} from {self._format_month_day(sorted_starts[0])} "
            f"to {self._format_month_day(sorted_starts[-1])}, {time_text}"
        )

    def _format_compact_period(self, start: datetime, end: Optional[datetime]) -> str:
        if end is None:
            return f"Starting {self._format_month_day(start)}, {start.strftime('%a')} {self._format_clock(start)}"
        if start.date() == end.date():
            return (
                f"{self._format_month_day(start)}, {start.strftime('%a')} "
                f"{self._format_clock(start)} to {self._format_clock(end)}"
            )
        if start.year == end.year and start.month == end.month:
            date_text = f"{start.strftime('%b')} {start.day} - {end.day}"
        elif start.year == end.year:
            date_text = f"{self._format_month_day(start)} - {self._format_month_day(end)}"
        else:
            date_text = (
                f"{start.strftime('%b')} {start.day}, {start.year} - "
                f"{end.strftime('%b')} {end.day}, {end.year}"
            )
        return (
            f"{date_text}, {start.strftime('%a')} {self._format_clock(start)} "
            f"to {end.strftime('%a')} {self._format_clock(end)}"
        )

    @staticmethod
    def _format_month_day(value: datetime) -> str:
        return f"{value.strftime('%b')} {value.day}"

    @staticmethod
    def _format_clock(value: datetime) -> str:
        if value.hour == 0 and value.minute == 0:
            return "midnight"
        if value.hour == 12 and value.minute == 0:
            return "noon"
        hour = value.hour % 12 or 12
        suffix = "AM" if value.hour < 12 else "PM"
        return f"{hour}:{value.minute:02d} {suffix}"

    @staticmethod
    def _parse_periods(
        active_periods: Sequence[Dict[str, Any]],
    ) -> List[Tuple[datetime, Optional[datetime]]]:
        parsed: List[Tuple[datetime, Optional[datetime]]] = []
        for row in active_periods:
            if not isinstance(row, dict):
                continue
            start_txt = str(row.get("start", "") or "").strip()
            end_txt = str(row.get("end", "") or "").strip()
            start = MercuryResolver._parse_iso(start_txt)
            end = MercuryResolver._parse_iso(end_txt) if end_txt else None
            if start is None:
                continue
            parsed.append((start, end))
        parsed.sort(key=lambda item: item[0])
        return parsed

    def _fallback_result(self) -> MercurySelectionResult:
        return self._result_for_key(FALLBACK_PRIORITY_KEY, 0.0)

    def _result_for_key(self, priority_key: str, confidence: float) -> MercurySelectionResult:
        key = self._normalize_priority_key(priority_key) or FALLBACK_PRIORITY_KEY
        return MercurySelectionResult(
            priority_key=key,
            priority_number=MERCURY_PRIORITY_BY_KEY[key],
            alert_type=self.alert_type_for_key(key),
            confidence=coerce_confidence(confidence),
        )

    @staticmethod
    def _normalize_priority_key(value: Any) -> str:
        if value is None:
            return ""
        key = str(value).strip().upper()
        if key.startswith("PRIORITY_"):
            key = key[len("PRIORITY_") :]
        key = re.sub(r"[^A-Z0-9]+", "_", key).strip("_")
        return key

    @staticmethod
    def _normalize_priority_number(value: Any) -> int:
        try:
            return int(str(value).strip())
        except Exception:
            return 0

    @staticmethod
    def _humanize_key(value: str) -> str:
        tokens = [token for token in str(value or "").split("_") if token]
        return " ".join(token.capitalize() for token in tokens)

    @staticmethod
    def _parse_iso(value: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(value.strip())
        except Exception:
            return None

    @staticmethod
    def _extract_first_json_object(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
