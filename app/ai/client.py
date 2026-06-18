from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from app.core.config import Settings


class AIServiceError(RuntimeError):
    """Raised when an AI provider result cannot be trusted."""


def build_openai_client(settings: Settings) -> AsyncOpenAI:
    if settings.openai_api_key is None:
        raise AIServiceError("OpenAI API key is not configured")
    return AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())


def parse_strict_json(raw_content: str | None) -> dict[str, Any]:
    if not raw_content:
        raise AIServiceError("AI provider returned an empty response")
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise AIServiceError("AI provider returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise AIServiceError("AI provider returned a non-object JSON payload")
    return parsed


def clamp_float(value: Any, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(maximum, number))


def normalize_str_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())
