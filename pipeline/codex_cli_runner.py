from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pipeline.llm_config import RuntimeLLMConfig


@dataclass(frozen=True)
class CodexCliUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "input_tokens": int(self.input_tokens),
            "cached_input_tokens": int(self.cached_input_tokens),
            "output_tokens": int(self.output_tokens),
        }


@dataclass(frozen=True)
class CodexCliTaskResult:
    payload: Dict[str, Any]
    usage: CodexCliUsage
    events: Sequence[Dict[str, Any]]


class CodexCliInvocationError(RuntimeError):
    def __init__(self, category: str, message: str, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.category = category
        self.stdout = stdout
        self.stderr = stderr


class CodexCliRunner:
    def __init__(self, config: RuntimeLLMConfig, cwd: str | Path):
        self.config = config
        self.cwd = str(Path(cwd).resolve())

    def run_json_task(
        self,
        task_name: str,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> CodexCliTaskResult:
        codex_bin = shutil.which(self.config.codex_cli_bin) or self.config.codex_cli_bin
        auth_path = Path(self.config.codex_auth_json_path).expanduser()
        if not auth_path.is_file():
            raise CodexCliInvocationError(
                "missing_auth",
                f"Codex auth file not found: {auth_path}",
            )

        home_root = Path(self.config.codex_cli_home_root).expanduser()
        home_root.mkdir(parents=True, exist_ok=True)
        temp_home = Path(tempfile.mkdtemp(prefix="codex-", dir=home_root))
        codex_home = temp_home / ".codex"
        codex_home.mkdir(parents=True, exist_ok=True)
        schema_path = temp_home / f"{task_name}_schema.json"
        output_path = temp_home / f"{task_name}_output.json"
        config_path = codex_home / "config.toml"
        auth_link = codex_home / "auth.json"

        selected_model = (model or self.config.codex_model_name).strip()
        selected_effort = (reasoning_effort or self.config.codex_reasoning_effort).strip().lower()

        schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        config_path.write_text(
            f'model = "{selected_model}"\nmodel_reasoning_effort = "{selected_effort}"\n',
            encoding="utf-8",
        )
        auth_link.symlink_to(auth_path)

        cmd = [
            codex_bin,
            "exec",
            "--json",
            "--output-last-message",
            str(output_path),
            "--output-schema",
            str(schema_path),
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "-C",
            self.cwd,
            "-m",
            selected_model,
            "-c",
            f'model_reasoning_effort="{selected_effort}"',
            prompt,
        ]

        env = os.environ.copy()
        env["HOME"] = str(temp_home)

        try:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self.cwd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.config.llm_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise CodexCliInvocationError("timeout", f"Codex CLI timed out during {task_name}.") from exc

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            events = self._parse_events(stdout)
            if self._has_turn_failed(events):
                raise CodexCliInvocationError(
                    self._categorize_failure(stdout, stderr, proc.returncode),
                    self._extract_error_message(stdout, stderr, proc.returncode),
                    stdout=stdout,
                    stderr=stderr,
                )
            if proc.returncode != 0:
                raise CodexCliInvocationError(
                    self._categorize_failure(stdout, stderr, proc.returncode),
                    self._extract_error_message(stdout, stderr, proc.returncode),
                    stdout=stdout,
                    stderr=stderr,
                )

            if not output_path.is_file():
                raise CodexCliInvocationError(
                    "missing_output",
                    f"Codex CLI did not produce an output file for {task_name}.",
                    stdout=stdout,
                    stderr=stderr,
                )

            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise CodexCliInvocationError(
                    "invalid_json",
                    f"Codex CLI returned invalid JSON for {task_name}: {exc}",
                    stdout=stdout,
                    stderr=stderr,
                ) from exc

            if not isinstance(payload, dict):
                raise CodexCliInvocationError(
                    "invalid_json",
                    f"Codex CLI returned a non-object payload for {task_name}.",
                    stdout=stdout,
                    stderr=stderr,
                )

            usage = self._extract_usage(events)
            return CodexCliTaskResult(payload=payload, usage=usage, events=events)
        finally:
            shutil.rmtree(temp_home, ignore_errors=True)

    @staticmethod
    def _parse_events(stdout: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for line in (stdout or "").splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                events.append(item)
        return events

    @staticmethod
    def _extract_usage(events: Sequence[Dict[str, Any]]) -> CodexCliUsage:
        for event in reversed(list(events)):
            if event.get("type") != "turn.completed":
                continue
            usage = event.get("usage", {})
            if not isinstance(usage, dict):
                continue
            return CodexCliUsage(
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                cached_input_tokens=int(usage.get("cached_input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
            )
        return CodexCliUsage()

    @staticmethod
    def _has_turn_failed(events: Sequence[Dict[str, Any]]) -> bool:
        for event in events:
            if event.get("type") == "turn.failed":
                return True
        return False

    @staticmethod
    def _categorize_failure(stdout: str, stderr: str, returncode: int) -> str:
        text = "\n".join([stdout or "", stderr or ""]).lower()
        if "not supported when using codex with a chatgpt account" in text:
            return "unsupported_model"
        if "missing bearer or basic authentication" in text or "401 unauthorized" in text:
            return "missing_auth"
        if "invalid json" in text:
            return "invalid_json"
        return "subprocess_failed"

    @staticmethod
    def _extract_error_message(stdout: str, stderr: str, returncode: int) -> str:
        events = CodexCliRunner._parse_events(stdout)
        for event in reversed(events):
            if event.get("type") == "turn.failed":
                err = event.get("error", {})
                if isinstance(err, dict) and err.get("message"):
                    return str(err["message"])
            if event.get("type") == "error" and event.get("message"):
                return str(event["message"])
        stderr_text = (stderr or "").strip()
        if stderr_text:
            return stderr_text.splitlines()[-1]
        return f"Codex CLI exited with code {returncode}."
