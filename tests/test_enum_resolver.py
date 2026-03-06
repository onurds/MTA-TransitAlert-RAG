from __future__ import annotations

from pipeline.compiler.enum_resolver import EnumResolver


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, prompt: str):
        return _Resp(self._content)


def test_enum_resolver_uses_llm_values_above_threshold():
    llm = _FakeLLM(
        '{"cause":"MAINTENANCE","effect":"MODIFIED_SERVICE","cause_confidence":0.91,"effect_confidence":0.82}'
    )
    resolver = EnumResolver(
        ensure_llm=lambda: True,
        llm_getter=lambda: llm,
        min_confidence=0.6,
    )
    out = resolver.resolve("signal maintenance", cause_override=None, effect_override=None)
    assert out.cause == "MAINTENANCE"
    assert out.effect == "MODIFIED_SERVICE"


def test_enum_resolver_falls_back_to_unknown_only_when_empty_llm():
    llm = _FakeLLM("{}")
    resolver = EnumResolver(
        ensure_llm=lambda: True,
        llm_getter=lambda: llm,
        min_confidence=0.6,
    )
    out = resolver.resolve("signal maintenance", cause_override=None, effect_override=None)
    assert out.cause == "UNKNOWN_CAUSE"
    assert out.effect == "UNKNOWN_EFFECT"


def test_enum_resolver_does_not_lock_unknown_from_invalid_override_if_llm_has_value():
    llm = _FakeLLM(
        '{"cause":"MAINTENANCE","effect":"MODIFIED_SERVICE","cause_confidence":0.95,"effect_confidence":0.9}'
    )
    resolver = EnumResolver(
        ensure_llm=lambda: True,
        llm_getter=lambda: llm,
        min_confidence=0.6,
    )
    out = resolver.resolve(
        "cause: signal maintenance",
        cause_override="SIGNAL MAINTENANCE",
        effect_override=None,
    )
    assert out.cause == "MAINTENANCE"
    assert out.effect == "MODIFIED_SERVICE"


def test_enum_resolver_maps_invalid_non_empty_llm_labels_to_other_enums():
    llm = _FakeLLM(
        '{"cause":"SIGNAL_MAINT","effect":"LOCAL_TO_EXPRESS","cause_confidence":0.95,"effect_confidence":0.91}'
    )
    resolver = EnumResolver(
        ensure_llm=lambda: True,
        llm_getter=lambda: llm,
        min_confidence=0.6,
    )
    out = resolver.resolve("free form", cause_override=None, effect_override=None)
    assert out.cause == "OTHER_CAUSE"
    assert out.effect == "OTHER_EFFECT"
