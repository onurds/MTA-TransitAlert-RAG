from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from evaluation.dataset import build_request_body
from pipeline.codex_cli_runner import CodexCliInvocationError, CodexCliRunner, CodexCliTaskResult, CodexCliUsage
from pipeline.compiler import AlertCompiler, CompileRequest
from pipeline.llm_config import load_llm_config


def test_load_llm_config_accepts_codex_cli_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    auth = tmp_path / "auth.json"
    auth.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("LLM_PROVIDER", "codex_cli")
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(auth))
    monkeypatch.setenv("CODEX_CLI_HOME_ROOT", str(tmp_path / "homes"))

    config = load_llm_config()

    assert config.provider == "codex_cli"
    assert config.codex_model_name == "gpt-5.4-mini"
    assert config.codex_reasoning_effort == "low"
    assert config.codex_auth_json_path == str(auth)


def test_build_request_body_forwards_llm_overrides():
    row = {
        "inputs": {"instruction": "A trains delayed."},
        "meta": {"reference_time": "2026-05-01T12:00:00-04:00"},
    }

    body = build_request_body(
        row,
        "default",
        request_id="req-123",
        llm_provider="codex_cli",
        llm_model="gpt-5.4",
        llm_reasoning_effort="low",
    )

    assert body["request_id"] == "req-123"
    assert body["llm_provider"] == "codex_cli"
    assert body["llm_model"] == "gpt-5.4"
    assert body["llm_reasoning_effort"] == "low"


def test_codex_cli_runner_builds_isolated_home_and_parses_usage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    auth = tmp_path / "auth.json"
    auth.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("LLM_PROVIDER", "codex_cli")
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(auth))
    monkeypatch.setenv("CODEX_CLI_HOME_ROOT", str(tmp_path / "homes"))
    config = load_llm_config()
    runner = CodexCliRunner(config=config, cwd=tmp_path)

    def fake_run(cmd, cwd, env, capture_output, text, timeout):
        home = Path(env["HOME"])
        codex_home = home / ".codex"
        assert (codex_home / "config.toml").is_file()
        assert (codex_home / "auth.json").is_symlink()
        assert (codex_home / "auth.json").resolve() == auth.resolve()
        config_text = (codex_home / "config.toml").read_text(encoding="utf-8")
        assert 'model = "gpt-5.4-mini"' in config_text
        assert 'model_reasoning_effort = "low"' in config_text

        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        output_path.write_text('{"status":"OK"}\n', encoding="utf-8")
        stdout = "\n".join([
            '{"type":"thread.started","thread_id":"t1"}',
            '{"type":"turn.started"}',
            '{"type":"turn.completed","usage":{"input_tokens":11,"cached_input_tokens":7,"output_tokens":5}}',
        ])
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.run_json_task(
        task_name="setup_check",
        prompt="Return status.",
        schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        },
    )

    assert result.payload == {"status": "OK"}
    assert result.usage == CodexCliUsage(input_tokens=11, cached_input_tokens=7, output_tokens=5)


def test_codex_cli_runner_surfaces_unsupported_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    auth = tmp_path / "auth.json"
    auth.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("LLM_PROVIDER", "codex_cli")
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(auth))
    monkeypatch.setenv("CODEX_CLI_HOME_ROOT", str(tmp_path / "homes"))
    config = load_llm_config()
    runner = CodexCliRunner(config=config, cwd=tmp_path)

    def fake_run(cmd, cwd, env, capture_output, text, timeout):
        stdout = "\n".join([
            '{"type":"thread.started","thread_id":"t1"}',
            '{"type":"turn.started"}',
            '{"type":"turn.failed","error":{"message":"The \'gpt-5\' model is not supported when using Codex with a ChatGPT account."}}',
        ])
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(CodexCliInvocationError) as excinfo:
        runner.run_json_task(
            task_name="setup_check",
            prompt="Return status.",
            schema={"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]},
            model="gpt-5",
        )
    assert excinfo.value.category == "unsupported_model"


def test_compiler_routes_codex_cli_branch(monkeypatch: pytest.MonkeyPatch):
    compiler = AlertCompiler(
        graph_path="data/mta_knowledge_graph.gpickle",
        calendar_path="data/2026_english_calendar.csv",
        timezone="America/New_York",
        confidence_threshold=0.85,
    )

    called = {}

    def fake_compile_codex_cli(request, instruction, compile_report, model, reasoning_effort):
        called["provider"] = compile_report["provider"]
        called["model"] = model
        called["reasoning_effort"] = reasoning_effort
        return {"id": "fake"}

    monkeypatch.setattr(compiler, "_compile_codex_cli", fake_compile_codex_cli)

    result = compiler.compile(
        CompileRequest(
            instruction="A trains are delayed.",
            request_id="req-branch",
            llm_provider="codex_cli",
            llm_model="gpt-5.4-mini",
            llm_reasoning_effort="low",
        )
    )

    assert result == {"id": "fake"}
    assert called == {
        "provider": "codex_cli",
        "model": "gpt-5.4-mini",
        "reasoning_effort": "low",
    }


def test_compiler_codex_cli_produces_payload_with_deterministic_variants(monkeypatch: pytest.MonkeyPatch):
    compiler = AlertCompiler(
        graph_path="data/mta_knowledge_graph.gpickle",
        calendar_path="data/2026_english_calendar.csv",
        timezone="America/New_York",
        confidence_threshold=0.85,
    )

    calls: list[str] = []

    def fake_run_json_task(self, task_name, prompt, schema, model=None, reasoning_effort=None):
        calls.append(task_name)
        if task_name == "core_extraction":
            return CodexCliTaskResult(
                    payload={
                        "selected_route_ids": ["A"],
                        "selected_stop_ids": [],
                        "active_periods": [{"start": "2026-04-25T18:00:00", "end": "2026-04-25T21:00:00"}],
                        "cause": "TECHNICAL_PROBLEM",
                        "effect": "SIGNIFICANT_DELAYS",
                        "mercury_priority_key": "DELAYS",
                        "clean_rider_source": "Northbound A trains are delayed from 6 PM to 9 PM.",
                        "confidence": 0.92,
                    },
                usage=CodexCliUsage(input_tokens=10, cached_input_tokens=5, output_tokens=3),
                events=[],
            )
        return CodexCliTaskResult(
            payload={
                "header_text": "Northbound A trains are delayed",
                "description_text": "Expect delays from 6 PM to 9 PM.",
                "confidence": 0.88,
            },
            usage=CodexCliUsage(input_tokens=12, cached_input_tokens=6, output_tokens=4),
            events=[],
        )

    monkeypatch.setattr(CodexCliRunner, "run_json_task", fake_run_json_task)

    compiled = compiler.compile(
        CompileRequest(
            instruction="Northbound A trains are delayed from today 6 PM to 9 PM.",
            request_id="req-codex",
            llm_provider="codex_cli",
            llm_model="gpt-5.4-mini",
            llm_reasoning_effort="low",
            reference_time="2026-04-25T12:00:00-04:00",
        )
    )

    assert calls == ["core_extraction", "english_text"]
    assert compiled["cause"] == "TECHNICAL_PROBLEM"
    assert compiled["effect"] == "SIGNIFICANT_DELAYS"
    assert compiled["tts_header_text"]["translation"][0]["text"]
    langs = {row["language"] for row in compiled["header_text"]["translation"]}
    assert {"en", "en-html", "zh", "es"} <= langs
    assert compiler.last_compile_report["telemetry"]["backend_mode"] == "codex_hybrid"
    assert compiler.last_compile_report["telemetry"]["codex_calls"] == 2
    assert compiler.get_compile_report("req-codex") == compiler.last_compile_report
