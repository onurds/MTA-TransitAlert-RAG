"""
Shared LLM runtime configuration for compiler and graph retrieval.

Supported providers:
- gemini
- openrouter (OpenAI-compatible endpoint)
- local (OpenAI-compatible local endpoint, e.g. Ollama, vLLM, mlx-lm)

Default runtime: OpenRouter with model x-ai/grok-4.1-fast and reasoning effort none
Local default: Ollama at http://localhost:11434/v1 with model qwen3.5:9b
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import SecretStr


@dataclass(frozen=True)
class RuntimeLLMConfig:
    provider: str
    gemini_model_name: str
    gemini_api_key_file: str
    openrouter_base_url: str
    openrouter_model_name: str
    openrouter_api_key_file: str
    openrouter_reasoning_effort: str
    local_base_url: str
    local_model_name: str
    local_disable_thinking: bool
    llm_timeout_seconds: float


def _normalize_gemini_model_name(model_name: str) -> str:
    """
    Normalize common Gemini aliases to callable model IDs for the current API.
    """
    name = model_name.strip()
    if name == "gemini-3-flash":
        return "gemini-3-flash-preview"
    return name


def load_llm_config() -> RuntimeLLMConfig:
    """Load runtime configuration from environment variables."""
    return RuntimeLLMConfig(
        provider=os.environ.get("LLM_PROVIDER", "openrouter").strip().lower(),
        gemini_model_name=_normalize_gemini_model_name(
            os.environ.get("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
        ),
        gemini_api_key_file=os.environ.get("GEMINI_API_KEY_FILE", ".gemini_api").strip(),
        openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip(),
        openrouter_model_name=os.environ.get("OPENROUTER_MODEL_NAME", "x-ai/grok-4.1-fast").strip(),
        openrouter_api_key_file=os.environ.get("OPENROUTER_API_KEY_FILE", ".vscode/.openrouter_api").strip(),
        openrouter_reasoning_effort=os.environ.get("OPENROUTER_REASONING_EFFORT", "none").strip().lower(),
        local_base_url=os.environ.get("LOCAL_BASE_URL", "http://localhost:11434/v1").strip(),
        local_model_name=os.environ.get("LOCAL_MODEL_NAME", "qwen3-30b-ctx8k").strip(),
        local_disable_thinking=os.environ.get("LOCAL_DISABLE_THINKING", "1").strip() not in ("0", "false", "no"),
        llm_timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "180")),
    )


def _read_secret_from_file(path: Path) -> Optional[str]:
    try:
        if path.is_file():
            secret = path.read_text(encoding="utf-8").strip()
            return secret or None
    except OSError:
        return None
    return None


def resolve_gemini_api_key(config: Optional[RuntimeLLMConfig] = None) -> Optional[str]:
    """Resolve Gemini API key from env var first, then .gemini_api files."""
    if not config:
        config = load_llm_config()

    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    candidate_paths = []
    configured = Path(config.gemini_api_key_file).expanduser()
    if configured.is_absolute():
        candidate_paths.append(configured)
    else:
        candidate_paths.append(Path.cwd() / configured)
        candidate_paths.append(Path.home() / configured)
    candidate_paths.append(Path.cwd() / ".gemini_api")
    candidate_paths.append(Path.home() / ".gemini_api")
    candidate_paths.append(Path.cwd() / ".vscode" / ".gemini_api")

    seen = set()
    for path in candidate_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        key = _read_secret_from_file(path)
        if key:
            return key

    return None


def resolve_openrouter_api_key(config: Optional[RuntimeLLMConfig] = None) -> Optional[str]:
    """Resolve OpenRouter API key from env var first, then configured key files."""
    if not config:
        config = load_llm_config()

    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key

    candidate_paths = []
    configured = Path(config.openrouter_api_key_file).expanduser()
    if configured.is_absolute():
        candidate_paths.append(configured)
    else:
        candidate_paths.append(Path.cwd() / configured)
        candidate_paths.append(Path.home() / configured)
    candidate_paths.append(Path.cwd() / ".vscode" / ".openrouter_api")
    candidate_paths.append(Path.home() / ".vscode" / ".openrouter_api")
    candidate_paths.append(Path.cwd() / ".openrouter_api")
    candidate_paths.append(Path.home() / ".openrouter_api")

    seen = set()
    for path in candidate_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        key = _read_secret_from_file(path)
        if key:
            return key

    return None


def with_overrides(
    base: RuntimeLLMConfig,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> RuntimeLLMConfig:
    """Return a config copy with optional provider/model overrides."""
    provider_override = (provider or "").strip().lower()
    model_override = (model or "").strip()
    reasoning_override = (reasoning_effort or "").strip().lower()
    selected_provider = provider_override or base.provider

    gemini_model = base.gemini_model_name
    openrouter_model = base.openrouter_model_name
    openrouter_reasoning_effort = base.openrouter_reasoning_effort
    local_model = base.local_model_name

    if model_override:
        if selected_provider == "gemini":
            gemini_model = _normalize_gemini_model_name(model_override)
        elif selected_provider == "openrouter":
            openrouter_model = model_override
        elif selected_provider == "local":
            local_model = model_override

    if reasoning_override and selected_provider == "openrouter":
        openrouter_reasoning_effort = reasoning_override

    return RuntimeLLMConfig(
        provider=selected_provider,
        gemini_model_name=gemini_model,
        gemini_api_key_file=base.gemini_api_key_file,
        openrouter_base_url=base.openrouter_base_url,
        openrouter_model_name=openrouter_model,
        openrouter_api_key_file=base.openrouter_api_key_file,
        openrouter_reasoning_effort=openrouter_reasoning_effort,
        local_base_url=base.local_base_url,
        local_model_name=local_model,
        local_disable_thinking=base.local_disable_thinking,
        llm_timeout_seconds=base.llm_timeout_seconds,
    )


def current_model_label(config: Optional[RuntimeLLMConfig] = None) -> str:
    if not config:
        config = load_llm_config()
    if config.provider == "gemini":
        return f"gemini:{config.gemini_model_name}"
    if config.provider == "openrouter":
        return f"openrouter:{config.openrouter_model_name}@{config.openrouter_base_url}"
    if config.provider == "local":
        return f"local:{config.local_model_name}@{config.local_base_url}"
    return f"{config.provider}:<unsupported>"


def build_langchain_chat_model(
    config: Optional[RuntimeLLMConfig] = None,
    temperature: float = 0.0,
):
    """Build a LangChain chat model instance from runtime config."""
    if not config:
        config = load_llm_config()

    if config.provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'langchain-google-genai'. Install requirements first."
            ) from exc

        api_key = resolve_gemini_api_key(config)
        if not api_key:
            raise RuntimeError(
                "Gemini API key not found. Set GEMINI_API_KEY or create .gemini_api."
            )

        return ChatGoogleGenerativeAI(
            model=config.gemini_model_name,
            google_api_key=api_key,
            temperature=temperature,
            timeout=config.llm_timeout_seconds,
        )

    if config.provider == "openrouter":
        from langchain_openai import ChatOpenAI  # local import avoids heavy init when unused

        api_key = resolve_openrouter_api_key(config)
        if not api_key:
            raise RuntimeError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY or create .vscode/.openrouter_api."
            )

        return ChatOpenAI(
            model=config.openrouter_model_name,
            base_url=config.openrouter_base_url,
            api_key=SecretStr(api_key),
            temperature=temperature,
            timeout=config.llm_timeout_seconds,
            extra_body={"reasoning": {"effort": config.openrouter_reasoning_effort}},
            default_headers={
                "HTTP-Referer": "https://github.com/google-gemini/gemini-cli",
                "X-Title": "Gemini CLI Transit Compiler",
            },
        )

    if config.provider == "local":
        from langchain_openai import ChatOpenAI  # local import avoids heavy init when unused

        extra: dict = {}
        if config.local_disable_thinking:
            extra["extra_body"] = {"think": False}

        # Force short outputs to discourage deep reasoning loops on simple JSON extraction tasks
        # Qwen3.5 9B reasoning models will jump straight to the answer if they know their token budget is small.
        return ChatOpenAI(
            model=config.local_model_name,
            base_url=config.local_base_url,
            api_key=SecretStr("not-needed"),
            temperature=temperature,
            timeout=config.llm_timeout_seconds,
            model_kwargs={"max_tokens": 600},
            **extra,
        )

    raise ValueError("Unsupported LLM_PROVIDER. Use 'gemini', 'openrouter', or 'local'.")
