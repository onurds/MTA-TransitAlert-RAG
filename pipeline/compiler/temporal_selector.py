from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

from pipeline.temporal_resolver import ResolvedPeriod, TemporalResolver

from .confidence import coerce_confidence
from .utils import invoke_json_with_repair


class TemporalSelector:
    """
    LLM-first temporal selector:
    - LLM picks concrete date/time components using calendar context.
    - Conversion to ISO periods remains deterministic.
    """

    def __init__(self) -> None:
        self.last_resolution_report: Dict[str, Any] = {
            "source": "deterministic",
            "repair_used": False,
            "period_count": 0,
        }

    def resolve_periods(
        self,
        llm: Any,
        temporal_text: str,
        full_instruction: str,
        resolver: TemporalResolver,
        reference_dt: datetime,
    ) -> List[ResolvedPeriod]:
        if llm is None:
            return []
        text = (temporal_text or "").strip()
        if not text:
            return []

        calendar_rows = self._build_calendar_context_rows(resolver, reference_dt.date(), days_ahead=400)
        if not calendar_rows:
            return []

        prompt = (
            "/no_think\n"
            "Resolve transit alert scheduling text using the provided calendar rows.\n"
            "Return strict JSON only with keys: periods (array), confidence (number 0..1).\n"
            "Each period item must be: "
            "{\"start_date\":\"YYYY-MM-DD\",\"start_time\":\"HH:MM\",\"end_date\":\"YYYY-MM-DD\",\"end_time\":\"HH:MM\"}.\n"
            "Rules:\n"
            "- Use only dates consistent with REFERENCE_LOCAL_TIME and CALENDAR_ROWS.\n"
            "- Interpret any scheduling language, including recurrence and exceptions.\n"
            "- Expand recurrence/exception logic into explicit concrete period rows.\n"
            "- Use 24-hour HH:MM.\n"
            "- If temporal information is absent or ambiguous, return periods=[].\n"
            "- Never return prose.\n\n"
            f"REFERENCE_LOCAL_TIME: {reference_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"TIMEZONE: {resolver.tz.key}\n"
            f"TEMPORAL_TEXT: {text}\n"
            f"FULL_INSTRUCTION: {(full_instruction or '').strip()}\n"
            f"CALENDAR_ROWS: {json.dumps(calendar_rows, ensure_ascii=False)}\n"
        )

        try:
            parsed, repair_used = invoke_json_with_repair(
                llm=llm,
                prompt=prompt,
                required_keys=("periods", "confidence"),
                repair_prompt_builder=lambda bad: (
                    "/no_think\n"
                    "Repair this temporal selector output. Return strict JSON only with "
                    "periods and confidence.\n"
                    f"RAW_OUTPUT:\n{bad}"
                ),
            )
        except Exception:
            return []

        confidence = coerce_confidence(parsed.get("confidence", 0.0))
        if confidence < 0.55:
            self.last_resolution_report = {
                "source": "deterministic",
                "repair_used": False,
                "period_count": 0,
                "confidence": confidence,
            }
            return []

        periods = parsed.get("periods", [])
        if not isinstance(periods, list):
            return []

        allowed_dates = {d.isoformat() for d in resolver.available_dates}
        out: List[ResolvedPeriod] = []
        seen = set()
        for row in periods[:64]:
            if not isinstance(row, dict):
                continue

            start_date_txt = str(row.get("start_date", "") or "").strip()
            start_time_txt = str(row.get("start_time", "") or "").strip()
            end_date_txt = str(row.get("end_date", "") or "").strip() or start_date_txt
            end_time_txt = str(row.get("end_time", "") or "").strip()
            if not start_date_txt or not start_time_txt or not end_date_txt or not end_time_txt:
                continue
            if start_date_txt not in allowed_dates or end_date_txt not in allowed_dates:
                continue

            start_date_obj = self._parse_yyyy_mm_dd(start_date_txt)
            end_date_obj = self._parse_yyyy_mm_dd(end_date_txt)
            start_time_obj = resolver._parse_time_expr(start_time_txt)  # intentional shared parser
            end_time_obj = resolver._parse_time_expr(end_time_txt)  # intentional shared parser
            if not start_date_obj or not end_date_obj or not start_time_obj or not end_time_obj:
                continue

            start_dt = datetime.combine(start_date_obj, start_time_obj, tzinfo=resolver.tz)
            end_dt = datetime.combine(end_date_obj, end_time_obj, tzinfo=resolver.tz)
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)
            if end_dt <= start_dt:
                continue

            key = (start_dt.isoformat(), end_dt.isoformat())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                ResolvedPeriod(
                    start=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    end=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    source="llm_calendar_selector",
                )
            )
        self.last_resolution_report = {
            "source": "llm" if out else "deterministic",
            "repair_used": repair_used,
            "period_count": len(out),
            "confidence": confidence,
        }
        return out

    @staticmethod
    def _build_calendar_context_rows(
        resolver: TemporalResolver,
        reference_date: date,
        days_ahead: int = 120,
    ) -> List[Dict[str, str]]:
        max_date = reference_date + timedelta(days=max(1, int(days_ahead)))
        rows: List[Dict[str, str]] = []
        for row in resolver.rows:
            d = row.get("date_obj")
            if not isinstance(d, date):
                continue
            if d < reference_date - timedelta(days=2) or d > max_date:
                continue
            rows.append(
                {
                    "date": str(row.get("date", "")),
                    "day_name": str(row.get("day_name", "")),
                    "is_weekend": str(row.get("is_weekend", "")),
                    "is_holiday": str(row.get("is_holiday", "")),
                    "is_business_day": str(row.get("is_business_day", "")),
                }
            )
        return rows

    @staticmethod
    def _parse_yyyy_mm_dd(value: str) -> Optional[date]:
        try:
            return datetime.strptime((value or "").strip(), "%Y-%m-%d").date()
        except Exception:
            return None
