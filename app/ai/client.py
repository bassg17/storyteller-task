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


def coerce_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return default


def format_ai_exception(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    if not message:
        return exc.__class__.__name__
    return f"{exc.__class__.__name__}: {_redact_secrets(message[:500])}"


def _redact_secrets(message: str) -> str:
    words = message.split()
    redacted_words = [
        "[redacted]" if word.startswith(("sk-", "AIza")) else word for word in words
    ]
    return " ".join(redacted_words)
