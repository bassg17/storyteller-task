from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from app.ai.client import clamp_float, normalize_str_list, parse_strict_json
from app.core.config import Settings
from app.models.internal import NormalizedPage, NormalizedStory, SemanticResult


class SemanticValidationService:
    def __init__(self, client: AsyncOpenAI, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    async def validate(
        self,
        story: NormalizedStory,
        page: NormalizedPage,
        ocr_text: str,
        vision_tags: tuple[str, ...],
        vision_description: str,
    ) -> SemanticResult:
        action = page.action
        response = await self.client.chat.completions.create(
            model=self.settings.openai_semantic_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict automated content validation engine. Return only "
                        "strict JSON with keys `semantic_score`, `confidence`, `risk_flags`, "
                        "and `reasoning`. Scores must be 0 to 1. Flag misleading CTAs, title "
                        "mismatches, unsafe claims, and URL intent mismatches."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                        "story_title": story.story_title,
                        "cta": action.cta if action else "",
                        "url": action.url if action else "",
                        "ocr_text": ocr_text,
                        "vision_description": vision_description,
                        "vision_tags": list(vision_tags),
                        "tenant_context": {
                            "tenant_id": story.tenant_id,
                            "tenant_name": story.tenant_name,
                            "categories": list(story.categories),
                            "publish_date": story.publish_date,
                        },
                        },
                        sort_keys=True,
                    ),
                },
            ],
            timeout=self.settings.ai_timeout_seconds,
        )
        payload = parse_strict_json(_message_content(response))
        return SemanticResult(
            semantic_score=clamp_float(payload.get("semantic_score")),
            confidence=clamp_float(payload.get("confidence")),
            risk_flags=normalize_str_list(payload.get("risk_flags")),
            reasoning=str(payload.get("reasoning", "")).strip(),
        )


def _message_content(response: Any) -> str | None:
    return response.choices[0].message.content
