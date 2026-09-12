from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_key: str | None
    base_url: str | None
    model: str
    temperature: float
    reasoning_effort: str | None
    thinking: str | None
    json_mode: bool


def get_llm_settings() -> LLMSettings:
    """Resolve a generic OpenAI-compatible model config with DeepSeek defaults."""
    provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower() or "deepseek"

    if provider == "deepseek":
        api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        base_url = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com"
        )
        model = (
            os.getenv("LLM_MODEL")
            or os.getenv("DEEPSEEK_MODEL")
            or "deepseek-v4-flash"
        )
        temperature = float(
            os.getenv("LLM_TEMPERATURE")
            or os.getenv("DEEPSEEK_TEMPERATURE")
            or "0.2"
        )
        reasoning_effort = (
            os.getenv("LLM_REASONING_EFFORT")
            or os.getenv("DEEPSEEK_REASONING_EFFORT")
            or "high"
        )
        thinking = (
            os.getenv("LLM_THINKING")
            or os.getenv("DEEPSEEK_THINKING")
            or "enabled"
        )
    else:
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL") or None
        model = os.getenv("LLM_MODEL", "").strip()
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        reasoning_effort = os.getenv("LLM_REASONING_EFFORT") or None
        thinking = os.getenv("LLM_THINKING") or None

    json_mode = os.getenv("LLM_JSON_MODE", "enabled").lower() not in {
        "0",
        "false",
        "disabled",
        "off",
    }

    return LLMSettings(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        thinking=thinking,
        json_mode=json_mode,
    )


def llm_enabled() -> bool:
    settings = get_llm_settings()
    return bool(settings.api_key and settings.model)


def deepseek_enabled() -> bool:
    """Backward-compatible alias; new code should use ``llm_enabled``."""
    return llm_enabled()


def llm_metadata() -> dict[str, Any]:
    """Return non-secret model metadata suitable for experiment result logging."""
    settings = get_llm_settings()
    metadata = asdict(settings)
    metadata.pop("api_key", None)
    return metadata


def _client(settings: LLMSettings | None = None) -> OpenAI:
    settings = settings or get_llm_settings()
    if not settings.api_key:
        raise RuntimeError(
            "No LLM API key is configured. Set DEEPSEEK_API_KEY for the default DeepSeek "
            "provider, or set LLM_API_KEY for another OpenAI-compatible provider."
        )
    if not settings.model:
        raise RuntimeError("LLM model name is not configured.")

    kwargs: dict[str, Any] = {"api_key": settings.api_key}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    return OpenAI(**kwargs)


def _completion_kwargs(settings: LLMSettings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "temperature": settings.temperature,
    }
    if settings.json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if settings.reasoning_effort:
        kwargs["reasoning_effort"] = settings.reasoning_effort
    if settings.provider == "deepseek" and settings.thinking:
        kwargs["extra_body"] = {"thinking": {"type": settings.thinking}}
    return kwargs


def chat_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    settings = get_llm_settings()
    client = _client(settings)
    kwargs = _completion_kwargs(settings)

    response = client.chat.completions.create(
        **kwargs,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"{settings.provider} returned empty content.")
    return json.loads(content)


def smoke_test() -> str:
    settings = get_llm_settings()
    client = _client(settings)
    kwargs: dict[str, Any] = {"model": settings.model, "max_tokens": 32}
    if settings.provider == "deepseek":
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    response = client.chat.completions.create(
        **kwargs,
        messages=[
            {"role": "system", "content": "Reply with exactly: LLM_OK"},
            {"role": "user", "content": "API connectivity test."},
        ],
    )
    return (response.choices[0].message.content or "").strip()
