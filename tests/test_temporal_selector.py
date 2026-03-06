from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pipeline.compiler.temporal_selector import TemporalSelector
from pipeline.temporal_resolver import TemporalResolver


class _Resp:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, prompt: str):
        return _Resp(self._content)


def _resolver() -> TemporalResolver:
    return TemporalResolver(calendar_path="data/2026_english_calendar.csv", timezone="America/New_York")


def test_llm_temporal_selector_accepts_valid_calendar_period():
    selector = TemporalSelector()
    resolver = _resolver()
    llm = _FakeLLM(
        (
            '{"periods":[{"start_date":"2026-03-06","start_time":"21:00",'
            '"end_date":"2026-03-06","end_time":"22:00"}],"confidence":0.92}'
        )
    )
    reference_dt = datetime(2026, 3, 5, 12, 0, 0, tzinfo=ZoneInfo("America/New_York"))

    periods = selector.resolve_periods(
        llm=llm,
        temporal_text="tomorrow 9pm to 10pm",
        full_instruction="B20 detoured tomorrow 9pm to 10pm",
        resolver=resolver,
        reference_dt=reference_dt,
    )

    assert len(periods) == 1
    assert periods[0].start == "2026-03-06T21:00:00"
    assert periods[0].end == "2026-03-06T22:00:00"
    assert periods[0].source == "llm_calendar_selector"


def test_llm_temporal_selector_rejects_out_of_calendar_date():
    selector = TemporalSelector()
    resolver = _resolver()
    llm = _FakeLLM(
        (
            '{"periods":[{"start_date":"2027-01-03","start_time":"21:00",'
            '"end_date":"2027-01-03","end_time":"22:00"}],"confidence":0.95}'
        )
    )
    reference_dt = datetime(2026, 3, 5, 12, 0, 0, tzinfo=ZoneInfo("America/New_York"))

    periods = selector.resolve_periods(
        llm=llm,
        temporal_text="next year sunday 9pm to 10pm",
        full_instruction="next year sunday 9pm to 10pm",
        resolver=resolver,
        reference_dt=reference_dt,
    )
    assert periods == []


def test_llm_temporal_selector_rejects_low_confidence():
    selector = TemporalSelector()
    resolver = _resolver()
    llm = _FakeLLM(
        (
            '{"periods":[{"start_date":"2026-03-06","start_time":"21:00",'
            '"end_date":"2026-03-06","end_time":"22:00"}],"confidence":0.4}'
        )
    )
    reference_dt = datetime(2026, 3, 5, 12, 0, 0, tzinfo=ZoneInfo("America/New_York"))

    periods = selector.resolve_periods(
        llm=llm,
        temporal_text="tomorrow 9pm to 10pm",
        full_instruction="tomorrow 9pm to 10pm",
        resolver=resolver,
        reference_dt=reference_dt,
    )
    assert periods == []


def test_llm_temporal_selector_accepts_multi_period_recurrence_with_exception():
    selector = TemporalSelector()
    resolver = _resolver()
    llm = _FakeLLM(
        (
            '{"periods":['
            '{"start_date":"2026-03-12","start_time":"19:00","end_date":"2026-03-12","end_time":"21:00"},'
            '{"start_date":"2026-03-17","start_time":"19:00","end_date":"2026-03-17","end_time":"21:00"},'
            '{"start_date":"2026-03-26","start_time":"19:00","end_date":"2026-03-26","end_time":"21:00"}'
            '],"confidence":0.93}'
        )
    )
    reference_dt = datetime(2026, 3, 5, 12, 0, 0, tzinfo=ZoneInfo("America/New_York"))

    periods = selector.resolve_periods(
        llm=llm,
        temporal_text="every thursday 7PM-9PM for 3 weeks except second week tuesday",
        full_instruction="dates are every thursday at 7PM-9PM for 3 weeks except for the second week where it will be tuesday",
        resolver=resolver,
        reference_dt=reference_dt,
    )
    assert len(periods) == 3
    assert periods[0].start == "2026-03-12T19:00:00"
    assert periods[1].start == "2026-03-17T19:00:00"
    assert periods[2].start == "2026-03-26T19:00:00"
