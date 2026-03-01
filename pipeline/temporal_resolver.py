"""
Deterministic relative-time resolver using a provided calendar CSV.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo


WEEKDAY_NAMES = {
    "monday": "Monday",
    "tuesday": "Tuesday",
    "wednesday": "Wednesday",
    "thursday": "Thursday",
    "friday": "Friday",
    "saturday": "Saturday",
    "sunday": "Sunday",
}


@dataclass(frozen=True)
class ResolvedPeriod:
    start: str
    end: str
    source: str


class TemporalResolver:
    def __init__(
        self,
        calendar_path: str = "data/2026_english_calendar.csv",
        timezone: str = "America/New_York",
    ):
        self.calendar_path = Path(calendar_path)
        self.tz = ZoneInfo(timezone)
        self.rows = self._load_calendar_rows(self.calendar_path)
        self.available_dates = [row["date_obj"] for row in self.rows]
        self.day_index = self._build_day_index(self.rows)

    @staticmethod
    def _load_calendar_rows(path: Path) -> List[Dict[str, object]]:
        if not path.exists():
            raise FileNotFoundError(f"Calendar file not found: {path}")
        rows: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        **row,
                        "date_obj": datetime.strptime(row["date"], "%Y-%m-%d").date(),
                        "day_name": row["day_name"].strip(),
                    }
                )
        rows.sort(key=lambda r: r["date_obj"])
        return rows

    @staticmethod
    def _build_day_index(rows: List[Dict[str, object]]) -> Dict[str, List[date]]:
        out: Dict[str, List[date]] = {name: [] for name in WEEKDAY_NAMES.values()}
        for row in rows:
            day_name = str(row["day_name"])
            out.setdefault(day_name, []).append(row["date_obj"])  # type: ignore[arg-type]
        return out

    def now(self) -> datetime:
        return datetime.now(self.tz)

    def _first_day_on_or_after(self, ref_date: date, day_name: str) -> Optional[date]:
        for d in self.day_index.get(day_name, []):
            if d >= ref_date:
                return d
        return None

    @staticmethod
    def _parse_time_expr(expr: str) -> Optional[time]:
        txt = expr.strip().lower().replace(".", "")
        m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", txt)
        if not m:
            return None
        hour = int(m.group(1))
        minute = int(m.group(2) or "0")
        meridiem = m.group(3)
        if meridiem:
            if hour == 12:
                hour = 0
            if meridiem == "pm":
                hour += 12
        if hour > 23 or minute > 59:
            return None
        return time(hour=hour, minute=minute)

    def _resolve_today_tomorrow_range(self, text: str, ref_dt: datetime) -> Optional[ResolvedPeriod]:
        # Pattern: "09:00 PM to 10:00 PM today"
        pattern_a = re.compile(
            r"(?:from\s+)?"
            r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*"
            r"(?:to|until|through|-)\s*"
            r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*"
            r"(today|tomorrow|tonight)",
            flags=re.IGNORECASE,
        )

        # Pattern: "today 09:00 PM to 10:00 PM"
        pattern_b = re.compile(
            r"(today|tomorrow|tonight)\s*(?:from\s+)?"
            r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*"
            r"(?:to|until|through|-)\s*"
            r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
            flags=re.IGNORECASE,
        )

        rel_day = ""
        start_txt = ""
        end_txt = ""

        match_a = pattern_a.search(text)
        if match_a:
            start_txt = match_a.group(1)
            end_txt = match_a.group(2)
            rel_day = match_a.group(3).lower()
        else:
            match_b = pattern_b.search(text)
            if not match_b:
                return None
            rel_day = match_b.group(1).lower()
            start_txt = match_b.group(2)
            end_txt = match_b.group(3)

        start_t = self._parse_time_expr(start_txt)
        end_t = self._parse_time_expr(end_txt)
        if not start_t or not end_t:
            return None

        if rel_day in {"today", "tonight"}:
            base_date = ref_dt.date()
        else:
            base_date = ref_dt.date() + timedelta(days=1)

        start_dt = datetime.combine(base_date, start_t, tzinfo=self.tz)
        end_dt = datetime.combine(base_date, end_t, tzinfo=self.tz)

        # Handle overnight windows.
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        return ResolvedPeriod(
            start=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            end=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            source="relative_day_time_range",
        )

    def _resolve_day_only(self, text: str, ref_dt: datetime) -> Optional[ResolvedPeriod]:
        lower = text.lower()
        if "today" not in lower and "tomorrow" not in lower and "tonight" not in lower:
            return None

        day = ref_dt.date()
        if "tomorrow" in lower:
            day += timedelta(days=1)

        start_dt = datetime.combine(day, time(0, 0), tzinfo=self.tz)
        end_dt = start_dt + timedelta(days=1)
        return ResolvedPeriod(
            start=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            end=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            source="relative_day_only",
        )

    def _resolve_day_range_pattern(self, text: str, ref_dt: datetime) -> Optional[ResolvedPeriod]:
        pattern = re.compile(
            r"(?:from\s+)?"
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+"
            r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)"
            r"\s*(?:to|until|through|-)\s*"
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+"
            r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
            flags=re.IGNORECASE,
        )
        m = pattern.search(text)
        if not m:
            return None

        start_day = WEEKDAY_NAMES[m.group(1).lower()]
        end_day = WEEKDAY_NAMES[m.group(3).lower()]
        start_t = self._parse_time_expr(m.group(2))
        end_t = self._parse_time_expr(m.group(4))
        if not start_t or not end_t:
            return None

        start_date = self._first_day_on_or_after(ref_dt.date(), start_day)
        if not start_date:
            return None

        # If "today at X" is already past, use next weekly occurrence in calendar.
        start_dt = datetime.combine(start_date, start_t, tzinfo=self.tz)
        if start_day == ref_dt.strftime("%A") and start_dt < ref_dt:
            start_date = self._first_day_on_or_after(start_date + timedelta(days=1), start_day)
            if not start_date:
                return None
            start_dt = datetime.combine(start_date, start_t, tzinfo=self.tz)

        end_date = self._first_day_on_or_after(start_date, end_day)
        if not end_date:
            return None
        end_dt = datetime.combine(end_date, end_t, tzinfo=self.tz)
        if end_dt <= start_dt:
            # Find the next weekly end-day occurrence.
            end_date = self._first_day_on_or_after(end_date + timedelta(days=1), end_day)
            if not end_date:
                return None
            end_dt = datetime.combine(end_date, end_t, tzinfo=self.tz)

        return ResolvedPeriod(
            start=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            end=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            source="calendar_day_range",
        )

    def resolve(self, text: str, reference_dt: Optional[datetime] = None) -> Optional[ResolvedPeriod]:
        if not text:
            return None
        ref_dt = reference_dt or self.now()
        lower = text.lower()

        # Highest-priority deterministic patterns.
        period = self._resolve_today_tomorrow_range(text, ref_dt)
        if period:
            return period

        if any(day in lower for day in WEEKDAY_NAMES):
            period = self._resolve_day_range_pattern(text, ref_dt)
            if period:
                return period

        return self._resolve_day_only(text, ref_dt)
