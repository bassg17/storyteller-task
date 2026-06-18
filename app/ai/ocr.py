from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from app.ai.client import clamp_float, parse_strict_json
from app.core.config import Settings
from app.models.internal import OcrResult


class OcrService:
    def __init__(self, client: AsyncOpenAI, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    async def extract(self, image_url: str) -> OcrResult:
        response = await self.client.chat.completions.create(
            model=self.settings.openai_ocr_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an OCR engine. Return only strict JSON with keys "
                        "`text` and `confidence`. Confidence must be 0 to 1 and "
                        "should reflect certainty that the extracted text is complete."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all visible text from this image."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            timeout=self.settings.ai_timeout_seconds,
        )
        payload = parse_strict_json(_message_content(response))
        return OcrResult(
            text=str(payload.get("text", "")).strip(),
            confidence=clamp_float(payload.get("confidence")),
        )


def _message_content(response: Any) -> str | None:
    return response.choices[0].message.content
