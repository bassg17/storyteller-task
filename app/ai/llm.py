from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from app.ai.client import coerce_bool, clamp_float, normalize_str_list, parse_strict_json
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
        media_context: str,
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
                        "`reasoning`, `media_matches_title`, `media_matches_categories`, "
                        "`cta_matches_url`, `cta_matches_title`, and `media_reasoning`. "
                        "Scores must be 0 to 1. The boolean match fields must be true only "
                        "when the relationship is semantically consistent. Use only these "
                        "risk flags when applicable: `media_title_mismatch`, "
                        "`media_category_mismatch`, `cta_url_mismatch`, `cta_title_mismatch`, "
                        "`unsafe_content`, `low_visual_confidence`, `ocr_text_mismatch`, "
                        "`misleading_cta`, `title_mismatch`, and `url_intent_mismatch`. "
                        "The input includes `media_type` and `media_context`. If media_type "
                        "is `video`, the visual description summarizes sampled frames from "
                        "the video; judge whether those sampled frames match the story title "
                        "and categories. "
                        "If the visual description or tags do not match the story title or "
                        "categories, set the relevant media match boolean to false and explain "
                        "the mismatch in `media_reasoning`."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                        "story_title": story.story_title,
                        "media_type": page.type,
                        "media_context": media_context,
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
            media_matches_title=coerce_bool(payload.get("media_matches_title")),
            media_matches_categories=coerce_bool(payload.get("media_matches_categories")),
            cta_matches_url=coerce_bool(payload.get("cta_matches_url")),
            cta_matches_title=coerce_bool(payload.get("cta_matches_title")),
            media_reasoning=str(payload.get("media_reasoning", "")).strip(),
        )


def _message_content(response: Any) -> str | None:
    return response.choices[0].message.content
