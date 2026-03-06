from __future__ import annotations

from pipeline.description_generator import DescriptionGenerator


class _Resp:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, prompt: str):
        return _Resp(self._content)


def test_generate_or_null_allows_llm_without_legacy_markers():
    generator = DescriptionGenerator(examples_path="data/does_not_exist.json")
    llm = _FakeLLM(
        '{"description":"A trains are suspended between 59 St and 125 St due to signal maintenance.","confidence":0.84}'
    )

    out = generator.generate_or_null(
        llm=llm,
        header="A service change.",
        source_text="A trains are suspended between 59 St and 125 St due to signal maintenance.",
        route_ids=["A"],
        cause="UNKNOWN_CAUSE",
        effect="SERVICE_SUSPENDED",
        current_description=None,
    )

    assert out is not None
    assert "suspended between 59 St and 125 St" in out


def test_generate_or_null_without_llm_remains_conservative():
    generator = DescriptionGenerator(examples_path="data/does_not_exist.json")

    out = generator.generate_or_null(
        llm=None,
        header="A service change.",
        source_text="A trains are suspended between 59 St and 125 St due to signal maintenance.",
        route_ids=["A"],
        cause="UNKNOWN_CAUSE",
        effect="SERVICE_SUSPENDED",
        current_description=None,
    )

    assert out is None
