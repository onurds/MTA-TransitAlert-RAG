"""
Shared LLM runtime configuration for compiler and graph retrieval.

Supported providers:
- gemini (default)
- xai (OpenAI-compatible endpoint)
- vllm (OpenAI-compatible local endpoint, for final Qwen comparison)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class RuntimeLLMConfig:
    provider: str
    gemini_model_name: str
    gemini_api_key_file: str
    xai_base_url: str
    xai_model_name: str
    xai_api_key_file: str
    vllm_base_url: str
    vllm_model_name: str
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
        provider=os.environ.get("LLM_PROVIDER", "gemini").strip().lower(),
        gemini_model_name=_normalize_gemini_model_name(
            os.environ.get("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
        ),
        gemini_api_key_file=os.environ.get("GEMINI_API_KEY_FILE", ".gemini_api").strip(),
        xai_base_url=os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1").strip(),
        xai_model_name=os.environ.get("XAI_MODEL_NAME", "grok-4-1-fast-reasoning").strip(),
        xai_api_key_file=os.environ.get("XAI_API_KEY_FILE", ".vscode/.xai_api").strip(),
        vllm_base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1").strip(),
        vllm_model_name=os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen3.5-35B-A3B").strip(),
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


def resolve_xai_api_key(config: Optional[RuntimeLLMConfig] = None) -> Optional[str]:
    """Resolve xAI API key from env var first, then configured key files."""
    if not config:
        config = load_llm_config()

    env_key = os.environ.get("XAI_API_KEY", "").strip()
    if env_key:
        return env_key

    candidate_paths = []
    configured = Path(config.xai_api_key_file).expanduser()
    if configured.is_absolute():
        candidate_paths.append(configured)
    else:
        candidate_paths.append(Path.cwd() / configured)
        candidate_paths.append(Path.home() / configured)
    candidate_paths.append(Path.cwd() / ".vscode" / ".xai_api")
    candidate_paths.append(Path.home() / ".vscode" / ".xai_api")
    candidate_paths.append(Path.cwd() / ".xai_api")
    candidate_paths.append(Path.home() / ".xai_api")

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
) -> RuntimeLLMConfig:
    """Return a config copy with optional provider/model overrides."""
    provider_override = (provider or "").strip().lower()
    model_override = (model or "").strip()
    selected_provider = provider_override or base.provider

    kwargs = {
        "provider": selected_provider,
        "gemini_model_name": base.gemini_model_name,
        "gemini_api_key_file": base.gemini_api_key_file,
        "xai_base_url": base.xai_base_url,
        "xai_model_name": base.xai_model_name,
        "xai_api_key_file": base.xai_api_key_file,
        "vllm_base_url": base.vllm_base_url,
        "vllm_model_name": base.vllm_model_name,
        "llm_timeout_seconds": base.llm_timeout_seconds,
    }

    if model_override:
        if selected_provider == "gemini":
            kwargs["gemini_model_name"] = _normalize_gemini_model_name(model_override)
        elif selected_provider == "xai":
            kwargs["xai_model_name"] = model_override
        elif selected_provider == "vllm":
            kwargs["vllm_model_name"] = model_override

    return RuntimeLLMConfig(**kwargs)


def current_model_label(config: Optional[RuntimeLLMConfig] = None) -> str:
    if not config:
        config = load_llm_config()
    if config.provider == "gemini":
        return f"gemini:{config.gemini_model_name}"
    if config.provider == "xai":
        return f"xai:{config.xai_model_name}@{config.xai_base_url}"
    if config.provider == "vllm":
        return f"vllm:{config.vllm_model_name}@{config.vllm_base_url}"
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

    if config.provider == "xai":
        from langchain_openai import ChatOpenAI  # local import avoids heavy init when unused

        api_key = resolve_xai_api_key(config)
        if not api_key:
            raise RuntimeError(
                "xAI API key not found. Set XAI_API_KEY or create .vscode/.xai_api."
            )

        return ChatOpenAI(
            model=config.xai_model_name,
            base_url=config.xai_base_url,
            api_key=api_key,
            temperature=temperature,
            request_timeout=config.llm_timeout_seconds,
        )

    if config.provider == "vllm":
        from langchain_openai import ChatOpenAI  # local import avoids heavy init when unused

        return ChatOpenAI(
            model=config.vllm_model_name,
            base_url=config.vllm_base_url,
            api_key="not-needed",
            temperature=temperature,
            request_timeout=config.llm_timeout_seconds,
        )

    raise ValueError("Unsupported LLM_PROVIDER. Use 'gemini', 'xai', or 'vllm'.")
