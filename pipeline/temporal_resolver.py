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

    def _resolve_from_today_until_weekday_time(self, text: str, ref_dt: datetime) -> Optional[ResolvedPeriod]:
        """
        Handle patterns like:
        - "from today timestamp until friday 10PM"
        - "from today until friday 10 PM"
        """
        pattern = re.compile(
            r"(?:from\s+)?today(?:\s+timestamp)?\s+"
            r"(?:to|until|through|-)\s+"
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+"
            r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
            flags=re.IGNORECASE,
        )
        m = pattern.search(text)
        if not m:
            return None

        end_day = WEEKDAY_NAMES[m.group(1).lower()]
        end_t = self._parse_time_expr(m.group(2))
        if not end_t:
            return None

        start_dt = ref_dt
        end_date = self._first_day_on_or_after(ref_dt.date(), end_day)
        if not end_date:
            return None
        end_dt = datetime.combine(end_date, end_t, tzinfo=self.tz)
        if end_dt <= start_dt:
            end_date = self._first_day_on_or_after(end_date + timedelta(days=1), end_day)
            if not end_date:
                return None
            end_dt = datetime.combine(end_date, end_t, tzinfo=self.tz)

        return ResolvedPeriod(
            start=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            end=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            source="today_until_weekday_time",
        )

    def _resolve_relative_now_duration(self, text: str, ref_dt: datetime) -> Optional[ResolvedPeriod]:
        """Handle 'now to N hours/minutes after', '2 hours from now', etc."""
        # "from right now to 2 hours after" / "now to 3 hours" / "now until 30 minutes"
        pattern = re.compile(
            r"(?:from\s+)?(?:right\s+)?now\s+(?:to|until|for(?:\s+the\s+next)?)\s+"
            r"(\d+(?:\.\d+)?)\s*(hour|hr|minute|min)s?"
            r"(?:\s+(?:after|later|from\s+now))?",
            flags=re.IGNORECASE,
        )
        m = pattern.search(text)
        if not m:
            # "2 hours from now" / "2 hours after" as a standalone end offset
            pattern2 = re.compile(
                r"(\d+(?:\.\d+)?)\s*(hour|hr|minute|min)s?\s+(?:after|later|from\s+now)",
                flags=re.IGNORECASE,
            )
            m = pattern2.search(text)
        if not m:
            return None
        amount = float(m.group(1))
        unit = m.group(2).lower()
        delta = timedelta(hours=amount) if unit.startswith("hour") or unit == "hr" else timedelta(minutes=amount)
        start_dt = ref_dt
        end_dt = start_dt + delta
        return ResolvedPeriod(
            start=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            end=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            source="relative_now_duration",
        )

    def _resolve_recurring_weekly(self, text: str, ref_dt: datetime) -> List[ResolvedPeriod]:
        """Handle 'every [weekday] for N weeks' with start/end times."""
        recur_m = re.search(
            r"(?:repeat\s+)?every\s+"
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+"
            r"for\s+(\d+)\s+weeks?",
            text,
            flags=re.IGNORECASE,
        )
        if not recur_m:
            return []

        weekday_name = WEEKDAY_NAMES[recur_m.group(1).lower()]
        num_weeks = int(recur_m.group(2))

        # Anchor date: "tomorrow" → next day; otherwise next occurrence of weekday.
        anchor = ref_dt.date()
        if re.search(r"\btomorrow\b", text, re.IGNORECASE):
            anchor = ref_dt.date() + timedelta(days=1)

        first_date = self._first_day_on_or_after(anchor, weekday_name)
        if not first_date:
            return []

        # Extract end time via explicit end marker first.
        end_t: Optional[time] = None
        m_end = re.search(
            r"(?:end(?:\s+hour)?(?:\s+is)?|until|till|through)\s*:?\s*"
            r"(\d{1,2}(?::\d{2})?\s*(?:am|pm))",
            text,
            flags=re.IGNORECASE,
        )
        if m_end:
            end_t = self._parse_time_expr(m_end.group(1))

        # Find start time: first time expression that is NOT the end-marker match.
        start_t: Optional[time] = None
        end_span = m_end.span(1) if m_end else None
        for tm in re.finditer(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)", text, re.IGNORECASE):
            if end_span and end_span[0] <= tm.start() <= end_span[1]:
                continue  # skip the end-time token
            t = self._parse_time_expr(tm.group(0))
            if t is not None:
                start_t = t
                break

        # If no explicit end marker, take last distinct time as end.
        if start_t and not end_t:
            for tm in reversed(list(re.finditer(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)", text, re.IGNORECASE))):
                t = self._parse_time_expr(tm.group(0))
                if t is not None and t != start_t:
                    end_t = t
                    break

        if not start_t or not end_t:
            return []

        periods: List[ResolvedPeriod] = []
        current = first_date
        for _ in range(num_weeks):
            s = datetime.combine(current, start_t, tzinfo=self.tz)
            e = datetime.combine(current, end_t, tzinfo=self.tz)
            if e <= s:
                e += timedelta(days=1)
            periods.append(ResolvedPeriod(
                start=s.strftime("%Y-%m-%dT%H:%M:%S"),
                end=e.strftime("%Y-%m-%dT%H:%M:%S"),
                source="recurring_weekly",
            ))
            current += timedelta(weeks=1)
        return periods

    def resolve_all(self, text: str, reference_dt: Optional[datetime] = None) -> List[ResolvedPeriod]:
        """Resolve all temporal periods from text, including recurring patterns."""
        if not text:
            return []
        ref_dt = reference_dt or self.now()
        lower = text.lower()

        # Recurring weekly — may return multiple periods.
        recurring = self._resolve_recurring_weekly(text, ref_dt)
        if recurring:
            return recurring

        # Fixed time range on today/tomorrow/tonight.
        period = self._resolve_today_tomorrow_range(text, ref_dt)
        if period:
            return [period]

        # Relative duration from now ("now to 2 hours after").
        period = self._resolve_relative_now_duration(text, ref_dt)
        if period:
            return [period]

        # "from today timestamp until friday 10PM"
        period = self._resolve_from_today_until_weekday_time(text, ref_dt)
        if period:
            return [period]

        # Weekday-to-weekday calendar range.
        if any(day in lower for day in WEEKDAY_NAMES):
            period = self._resolve_day_range_pattern(text, ref_dt)
            if period:
                return [period]

        # Day-only fallback.
        period = self._resolve_day_only(text, ref_dt)
        if period:
            return [period]

        return []

    def resolve(self, text: str, reference_dt: Optional[datetime] = None) -> Optional[ResolvedPeriod]:
        """Return the first resolved period (backward-compatible single-period interface)."""
        periods = self.resolve_all(text, reference_dt)
        return periods[0] if periods else None
