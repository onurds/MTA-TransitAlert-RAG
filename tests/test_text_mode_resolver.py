from __future__ import annotations

from pipeline.compiler.models import CompileRequest
from pipeline.compiler.text_mode_resolver import TextModeResolver
from scripts.gradio_app import make_compile_request


class _Resp:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        if not self._contents:
            raise RuntimeError("No more fake responses")
        return _Resp(self._contents.pop(0))


def test_compile_request_defaults_to_default_text_mode():
    req = CompileRequest(instruction="B11 detour")
    assert req.text_mode == "default"


def test_gradio_request_helper_threads_text_mode():
    req = make_compile_request("B11 detour", "", "", "rewrite")
    assert req.text_mode == "rewrite"


def test_default_mode_preserves_full_rider_facing_content_except_commands():
    resolver = TextModeResolver()
    llm = _FakeLLM([
        (
            '{"rider_text":"Eastbound M66 buses will not stop on W 65th St at Columbus Ave. '
            'Use the stop on W 65th at Broadway instead. See a map of this stop change. '
            'What\'s happening? Building Construction Note: Bus arrival information may not be available/accurate while buses are detoured."}'
        ),
        (
            '{"header_text":"Eastbound M66 buses will not stop on W 65th St at Columbus Ave.",'
            '"description_text":"Use the stop on W 65th at Broadway instead. See a map of this stop change. '
            'What\'s happening? Building Construction Note: Bus arrival information may not be available/accurate while buses are detoured.",'
            '"confidence":0.91}'
        ),
        (
            '{"header_text":"Eastbound M66 buses will not stop on W 65th St at Columbus Ave.",'
            '"description_text":"Use the stop on W 65th at Broadway instead. See a map of this stop change. '
            'What\'s happening? Building Construction Note: Bus arrival information may not be available/accurate while buses are detoured."}'
        ),
    ])

    header, description = resolver.resolve(
        llm=llm,
        instruction=(
            "Eastbound M66 buses will not stop on W 65th St at Columbus Ave. "
            "Use the stop on W 65th at Broadway instead. See a map of this stop change. "
            "What's happening? Building Construction Note: Bus arrival information may not be available/accurate while buses are detoured. "
            "Timeframe is from today until Friday 10PM."
        ),
        route_ids=["M66"],
        cause="CONSTRUCTION",
        effect="STOP_MOVED",
        text_mode="default",
        header_hint=None,
    )

    assert header == "Eastbound M66 buses will not stop on W 65th St at Columbus Ave."
    assert description is not None
    assert "Use the stop on W 65th at Broadway instead." in description
    assert "Timeframe is" not in description
    assert "What's happening?" in description


def test_default_mode_uses_llm_cleanup_for_semantic_operator_command_leak():
    resolver = TextModeResolver()
    llm = _FakeLLM([
        (
            '{"rider_text":"These buses are delayed: B: 4, 6, 41, 42. '
            'We\'re running as much service as we can with the buses and operators we have available."}'
        ),
        (
            '{"header_text":"These buses are delayed: B: 4, 6, 41, 42.",'
            '"description_text":"We\'re running as much service as we can with the buses and operators we have available. '
            'Make the timeframe from now until 5 hours ahead.",'
            '"confidence":0.89}'
        ),
        (
            '{"header_text":"These buses are delayed: B: 4, 6, 41, 42.",'
            '"description_text":"We\'re running as much service as we can with the buses and operators we have available."}'
        ),
    ])

    header, description = resolver.resolve(
        llm=llm,
        instruction=(
            "These buses are delayed: B: 4, 6, 41, 42. "
            "We're running as much service as we can with the buses and operators we have available. "
            "Make the timeframe from now until 5 hours ahead."
        ),
        route_ids=["B4", "B6", "B41", "B42"],
        cause="UNKNOWN_CAUSE",
        effect="SIGNIFICANT_DELAYS",
        text_mode="default",
    )

    assert header == "These buses are delayed: B: 4, 6, 41, 42."
    assert description == "We're running as much service as we can with the buses and operators we have available."
    assert len(llm.prompts) == 3
    assert "Remove only operator/control commands" in llm.prompts[2]


def test_default_mode_can_skip_llm_rider_extraction_when_rider_source_is_precomputed():
    resolver = TextModeResolver()
    llm = _FakeLLM([
        (
            '{"header_text":"These buses are delayed: B: 4, 6, 41, 42.",'
            '"description_text":"We\'re running as much service as we can with the buses and operators we have available.",'
            '"confidence":0.89}'
        ),
        (
            '{"header_text":"These buses are delayed: B: 4, 6, 41, 42.",'
            '"description_text":"We\'re running as much service as we can with the buses and operators we have available."}'
        ),
    ])

    header, description = resolver.resolve(
        llm=llm,
        instruction=(
            "These buses are delayed: B: 4, 6, 41, 42. "
            "We're running as much service as we can with the buses and operators we have available. "
            "Make the timeframe from now until 5 hours ahead."
        ),
        rider_source_override=(
            "These buses are delayed: B: 4, 6, 41, 42. "
            "We're running as much service as we can with the buses and operators we have available."
        ),
        route_ids=["B4", "B6", "B41", "B42"],
        cause="UNKNOWN_CAUSE",
        effect="SIGNIFICANT_DELAYS",
        text_mode="default",
    )

    assert header == "These buses are delayed: B: 4, 6, 41, 42."
    assert description == "We're running as much service as we can with the buses and operators we have available."
    assert len(llm.prompts) == 2


def test_default_mode_retries_on_mid_sentence_split():
    resolver = TextModeResolver()
    llm = _FakeLLM([
        (
            '{"rider_text":"Eastbound M66 buses will not stop on W 65th St at Columbus Ave, '
            'use the stop on W 65th at Broadway instead. See a map of this stop change."}'
        ),
        (
            '{"header_text":"Eastbound M66 buses will not stop on W 65th St at Columbus Ave.",'
            '"description_text":"use the stop on W 65th at Broadway instead. See a map of this stop change.",'
            '"confidence":0.88}'
        ),
        (
            '{"header_text":"Eastbound M66 buses will not stop on W 65th St at Columbus Ave.",'
            '"description_text":"use the stop on W 65th at Broadway instead. See a map of this stop change."}'
        ),
        (
            '{"header_text":"Eastbound M66 buses will not stop on W 65th St at Columbus Ave, use the stop on W 65th at Broadway instead.",'
            '"description_text":"See a map of this stop change.",'
            '"confidence":0.92}'
        ),
        (
            '{"header_text":"Eastbound M66 buses will not stop on W 65th St at Columbus Ave, use the stop on W 65th at Broadway instead.",'
            '"description_text":"See a map of this stop change."}'
        ),
    ])

    header, description = resolver.resolve(
        llm=llm,
        instruction=(
            "Eastbound M66 buses will not stop on W 65th St at Columbus Ave, "
            "use the stop on W 65th at Broadway instead. See a map of this stop change."
        ),
        route_ids=["M66"],
        cause="CONSTRUCTION",
        effect="STOP_MOVED",
        text_mode="default",
    )

    assert header.endswith("Broadway instead.")
    assert description == "See a map of this stop change."
    assert len(llm.prompts) == 5


def test_default_mode_fallback_keeps_full_first_sentence_with_semicolon():
    resolver = TextModeResolver()
    llm = _FakeLLM([
        (
            '{"rider_text":"Eastbound M66 stop on W 65th St at Columbus Ave is being bypassed; '
            'use the stop on W 65th at Broadway instead. See a map of this stop change. '
            'What\'s happening? Building Construction."}'
        ),
        (
            '{"header_text":"Eastbound M66 stop on W 65th St at Columbus Ave is being bypassed",'
            '"description_text":"use the stop on W 65th at Broadway instead. See a map of this stop change.",'
            '"confidence":0.83}'
        ),
        (
            '{"header_text":"Eastbound M66 stop on W 65th St at Columbus Ave is being bypassed",'
            '"description_text":"use the stop on W 65th at Broadway instead. See a map of this stop change."}'
        ),
        (
            '{"header_text":"Eastbound M66 stop on W 65th St at Columbus Ave is being bypassed",'
            '"description_text":"use the stop on W 65th at Broadway instead. See a map of this stop change.",'
            '"confidence":0.82}'
        ),
        (
            '{"header_text":"Eastbound M66 stop on W 65th St at Columbus Ave is being bypassed",'
            '"description_text":"use the stop on W 65th at Broadway instead. See a map of this stop change."}'
        ),
    ])

    header, description = resolver.resolve(
        llm=llm,
        instruction=(
            "Eastbound M66 stop on W 65th St at Columbus Ave is being bypassed; "
            "use the stop on W 65th at Broadway instead. See a map of this stop change. "
            "What's happening? Building Construction. Timeframe is from today timestamp until Friday 10PM."
        ),
        route_ids=["M66"],
        cause="CONSTRUCTION",
        effect="STOP_MOVED",
        text_mode="default",
    )

    assert (
        header
        == "Eastbound M66 stop on W 65th St at Columbus Ave is being bypassed; use the stop on W 65th at Broadway instead."
    )
    assert description == "See a map of this stop change. What's happening? Building Construction."


def test_default_mode_fallback_keeps_full_first_sentence_with_comma():
    resolver = TextModeResolver()

    header, description = resolver.resolve(
        llm=None,
        instruction=(
            "Eastbound M66 buses will not stop on W 65th St at Columbus Ave, "
            "use the stop on W 65th at Broadway instead. See a map of this stop change."
        ),
        route_ids=["M66"],
        cause="CONSTRUCTION",
        effect="STOP_MOVED",
        text_mode="default",
    )

    assert (
        header
        == "Eastbound M66 buses will not stop on W 65th St at Columbus Ave, use the stop on W 65th at Broadway instead."
    )
    assert description == "See a map of this stop change."


def test_default_mode_ignores_partial_header_hint_in_fallback():
    resolver = TextModeResolver()

    header, description = resolver.resolve(
        llm=None,
        instruction=(
            "Eastbound M66 stop on W 65th St at Columbus Ave is being bypassed; "
            "use the stop on W 65th at Broadway instead. See a map of this stop change."
        ),
        route_ids=["M66"],
        cause="CONSTRUCTION",
        effect="STOP_MOVED",
        text_mode="default",
        header_hint="Eastbound M66 stop on W 65th St at Columbus Ave is being bypassed",
    )

    assert (
        header
        == "Eastbound M66 stop on W 65th St at Columbus Ave is being bypassed; use the stop on W 65th at Broadway instead."
    )
    assert description == "See a map of this stop change."


def test_rewrite_mode_can_restyle_while_preserving_facts():
    resolver = TextModeResolver()
    llm = _FakeLLM([
        (
            '{"rider_text":"Eastbound M66 buses will not stop on W 65th St at Columbus Ave. '
            'Use the stop on W 65th at Broadway instead. See a map of this stop change."}'
        ),
        (
            '{"header_text":"M66 stop change at W 65th St and Columbus Ave.",'
            '"description_text":"Eastbound M66 buses will not stop at W 65th St at Columbus Ave. '
            'Use the stop on W 65th at Broadway instead. See a map of this stop change.",'
            '"confidence":0.89}'
        ),
        (
            '{"header_text":"M66 stop change at W 65th St and Columbus Ave.",'
            '"description_text":"Eastbound M66 buses will not stop at W 65th St at Columbus Ave. '
            'Use the stop on W 65th at Broadway instead. See a map of this stop change."}'
        ),
    ])

    header, description = resolver.resolve(
        llm=llm,
        instruction=(
            "Eastbound M66 buses will not stop on W 65th St at Columbus Ave. "
            "Use the stop on W 65th at Broadway instead. See a map of this stop change."
        ),
        route_ids=["M66"],
        cause="CONSTRUCTION",
        effect="STOP_MOVED",
        text_mode="rewrite",
    )

    assert header == "M66 stop change at W 65th St and Columbus Ave."
    assert description is not None
    assert "Use the stop on W 65th at Broadway instead." in description


def test_default_mode_prompt_mentions_header_description_hints():
    resolver = TextModeResolver()
    llm = _FakeLLM([
        (
            '{"rider_text":"B11 buses will be detoured between 50th St at 7th Ave and 5th Ave at 49th St. '
            'For eastbound service, use the stops on 50th St at 6th Ave or 9th Ave."}'
        ),
        '{"header_text":"B11 buses will be detoured between 50th St at 7th Ave and 5th Ave at 49th St.",'
        '"description_text":"For eastbound service, use the stops on 50th St at 6th Ave or 9th Ave.",'
        '"confidence":0.9}',
        '{"header_text":"B11 buses will be detoured between 50th St at 7th Ave and 5th Ave at 49th St.",'
        '"description_text":"For eastbound service, use the stops on 50th St at 6th Ave or 9th Ave."}'
    ])

    resolver.resolve(
        llm=llm,
        instruction=(
            "header: B11 buses will be detoured between 50th St at 7th Ave and 5th Ave at 49th St. "
            "description: For eastbound service, use the stops on 50th St at 6th Ave or 9th Ave."
        ),
        route_ids=["B11"],
        cause="UNKNOWN_CAUSE",
        effect="DETOUR",
        text_mode="default",
    )

    assert llm.prompts
    assert "Treat cues like 'header:' and 'description:' as strong hints" in llm.prompts[1]


def test_default_mode_explicit_layout_cues_still_require_whole_units():
    resolver = TextModeResolver()
    llm = _FakeLLM([
        (
            '{"rider_text":"B11 buses will be detoured between 50th St at 7th Ave and 5th Ave at 49th St. '
            'For eastbound service, use the stops on 50th St at 6th Ave or 9th Ave."}'
        ),
        (
            '{"header_text":"B11 buses will be detoured between 50th St at 7th Ave and 5th Ave at 49th St.",'
            '"description_text":"For eastbound service, use the stops on 50th St at 6th Ave or 9th Ave.",'
            '"confidence":0.9}'
        ),
        (
            '{"header_text":"B11 buses will be detoured between 50th St at 7th Ave and 5th Ave at 49th St.",'
            '"description_text":"For eastbound service, use the stops on 50th St at 6th Ave or 9th Ave."}'
        ),
    ])

    header, description = resolver.resolve(
        llm=llm,
        instruction=(
            "header: B11 buses will be detoured between 50th St at 7th Ave and 5th Ave at 49th St. "
            "description: For eastbound service, use the stops on 50th St at 6th Ave or 9th Ave."
        ),
        route_ids=["B11"],
        cause="UNKNOWN_CAUSE",
        effect="DETOUR",
        text_mode="default",
    )

    assert header == "B11 buses will be detoured between 50th St at 7th Ave and 5th Ave at 49th St."
    assert description == "For eastbound service, use the stops on 50th St at 6th Ave or 9th Ave."


def test_default_mode_preserves_whats_happening_and_note_order():
    resolver = TextModeResolver()
    llm = _FakeLLM([
        (
            '{"rider_text":"Buses replace trains between Babylon and Montauk. '
            'Check bus departure times on TrainTime or at mta.info/schedules, as westbound buses leave earlier than normal train times. '
            'What\'s happening? Track maintenance. NOTE: Bus arrival information may not be accurate."}'
        ),
        (
            '{"header_text":"Buses replace trains between Babylon and Montauk.",'
            '"description_text":"Check bus departure times on TrainTime or at mta.info/schedules, as westbound buses leave earlier than normal train times. '
            'What\'s happening? Track maintenance. NOTE: Bus arrival information may not be accurate.",'
            '"confidence":0.9}'
        ),
        (
            '{"header_text":"Buses replace trains between Babylon and Montauk.",'
            '"description_text":"Check bus departure times on TrainTime or at mta.info/schedules, as westbound buses leave earlier than normal train times. '
            'What\'s happening? Track maintenance. NOTE: Bus arrival information may not be accurate."}'
        ),
    ])

    header, description = resolver.resolve(
        llm=llm,
        instruction=(
            "Buses replace trains between Babylon and Montauk. "
            "Check bus departure times on TrainTime or at mta.info/schedules, as westbound buses leave earlier than normal train times. "
            "What's happening? Track maintenance. NOTE: Bus arrival information may not be accurate."
        ),
        route_ids=[],
        cause="MAINTENANCE",
        effect="MODIFIED_SERVICE",
        text_mode="default",
    )

    assert header == "Buses replace trains between Babylon and Montauk."
    assert description is not None
    assert description.index("What's happening?") < description.index("NOTE:")
