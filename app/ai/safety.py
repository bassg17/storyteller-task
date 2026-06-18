from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from app.core.config import Settings
from app.models.internal import SafetyResult


class SafetyModerationService:
    def __init__(self, client: AsyncOpenAI, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    async def moderate(self, image_url: str, label: str) -> SafetyResult:
        response = await self.client.moderations.create(
            model=self.settings.openai_moderation_model,
            input=[
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            ],
            timeout=self.settings.ai_timeout_seconds,
        )
        result = response.results[0]
        categories = _true_categories(result.categories)
        category_scores = _category_scores(result.category_scores)
        reasons = _safety_reasons(
            label=label,
            provider_flagged=bool(result.flagged),
            categories=categories,
            category_scores=category_scores,
            settings=self.settings,
        )

        return SafetyResult(
            flagged=bool(result.flagged) or bool(reasons),
            categories=categories,
            category_scores=category_scores,
            confidence=_safety_confidence(category_scores),
            reasons=reasons,
        )


def _true_categories(categories: Any) -> tuple[str, ...]:
    data = _to_dict(categories)
    return tuple(sorted(_normalize_category_key(key) for key, value in data.items() if bool(value)))


def _category_scores(scores: Any) -> dict[str, float]:
    data = _to_dict(scores)
    parsed_scores: dict[str, float] = {}
    for key, value in data.items():
        try:
            parsed_scores[_normalize_category_key(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed_scores


def _safety_reasons(
    label: str,
    provider_flagged: bool,
    categories: tuple[str, ...],
    category_scores: dict[str, float],
    settings: Settings,
) -> list[str]:
    reasons: list[str] = []
    sexual_score = category_scores.get("sexual", 0.0)
    sexual_minors_score = category_scores.get("sexual/minors", 0.0)

    if provider_flagged:
        category_text = ", ".join(categories) if categories else "provider flagged"
        reasons.append(f"Explicit or unsafe content detected in {label}: {category_text}")
    if sexual_score >= settings.sexual_content_threshold:
        reasons.append(
            f"Sexual content score {sexual_score:.2f} in {label} exceeds "
            f"{settings.sexual_content_threshold:.2f}"
        )
    if sexual_minors_score >= settings.explicit_content_threshold:
        reasons.append(
            f"Sexual minors content score {sexual_minors_score:.2f} in {label} exceeds "
            f"{settings.explicit_content_threshold:.2f}"
        )

    return reasons


def _safety_confidence(category_scores: dict[str, float]) -> float:
    if not category_scores:
        return 1.0
    return max(0.0, min(1.0, max(category_scores.values())))


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }


def _normalize_category_key(key: str) -> str:
    return key.replace("_", "/")
