from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from app.ai.client import clamp_float, normalize_str_list, parse_strict_json
from app.core.config import Settings
from app.models.internal import VisionResult


class VisionService:
    def __init__(self, client: AsyncOpenAI, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    async def understand(self, image_url: str) -> VisionResult:
        response = await self.client.chat.completions.create(
            model=self.settings.openai_vision_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify visual content for publication safety and semantic "
                        "matching. Return only strict JSON with keys `description`, `tags`, "
                        "`confidence`, and `nsfw_detected`."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe the scene, list concise content tags, and flag "
                                "NSFW or unsafe content."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            timeout=self.settings.ai_timeout_seconds,
        )
        payload = parse_strict_json(_message_content(response))
        return VisionResult(
            description=str(payload.get("description", "")).strip(),
            tags=normalize_str_list(payload.get("tags")),
            confidence=clamp_float(payload.get("confidence")),
            nsfw_detected=bool(payload.get("nsfw_detected", False)),
        )


def _message_content(response: Any) -> str | None:
    return response.choices[0].message.content
