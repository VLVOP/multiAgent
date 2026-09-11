from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def deepseek_enabled() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def _client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in.")

    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def chat_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    client = _client()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    thinking = os.getenv("DEEPSEEK_THINKING", "enabled")
    reasoning_effort = os.getenv("DEEPSEEK_REASONING_EFFORT", "high")
    temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.2"))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        extra_body={"thinking": {"type": thinking}},
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("DeepSeek returned empty content.")
    return json.loads(content)


def smoke_test() -> str:
    client = _client()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Reply with exactly: DEEPSEEK_OK"},
            {"role": "user", "content": "API connectivity test."},
        ],
        max_tokens=32,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return (response.choices[0].message.content or "").strip()
