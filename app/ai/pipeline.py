from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.ai.client import AIServiceError, build_openai_client
from app.ai.llm import SemanticValidationService
from app.ai.ocr import OcrService
from app.ai.video import VideoFrameSampler
from app.ai.vision import VisionService
from app.core.config import Settings
from app.models.internal import (
    AIResult,
    NormalizedPage,
    NormalizedPayload,
    NormalizedStory,
    OcrResult,
    SemanticResult,
    VisionResult,
)


@dataclass(frozen=True)
class PageAIResult:
    ocr_text: str
    ocr_confidence: float
    vision_description: str
    vision_tags: tuple[str, ...]
    vision_confidence: float
    semantic: SemanticResult
    nsfw_detected: bool


class AIPipeline:
    def __init__(
        self,
        ocr: OcrService,
        vision: VisionService,
        semantic: SemanticValidationService,
        video_sampler: VideoFrameSampler,
    ) -> None:
        self.ocr = ocr
        self.vision = vision
        self.semantic = semantic
        self.video_sampler = video_sampler

    @classmethod
    def from_settings(cls, settings: Settings) -> AIPipeline:
        client = build_openai_client(settings)
        return cls(
            ocr=OcrService(client, settings),
            vision=VisionService(client, settings),
            semantic=SemanticValidationService(client, settings),
            video_sampler=VideoFrameSampler(settings),
        )

    async def validate(self, payload: NormalizedPayload) -> dict[str, AIResult]:
        results = await asyncio.gather(
            *(self._validate_story(story) for story in payload.stories),
            return_exceptions=False,
        )
        return dict(results)

    async def _validate_story(self, story: NormalizedStory) -> tuple[str, AIResult]:
        try:
            page_results = await asyncio.gather(
                *(self._validate_page(story, page) for page in story.pages)
            )
        except Exception as exc:
            return story.story_id, AIResult(
                confidence_score=0.0,
                reasons=[f"AI validation failed closed: {exc.__class__.__name__}"],
            )

        semantic_scores = [page.semantic.semantic_score for page in page_results]
        semantic_confidences = [page.semantic.confidence for page in page_results]
        vision_confidences = [page.vision_confidence for page in page_results]
        ocr_confidences = [page.ocr_confidence for page in page_results]

        semantic_score = _average(semantic_scores)
        semantic_confidence = min(semantic_confidences, default=0.0)
        vision_confidence = min(vision_confidences, default=0.0)
        ocr_confidence = min(ocr_confidences, default=0.0)
        confidence_score = min(semantic_confidence, vision_confidence, ocr_confidence)

        risk_flags = sorted(
            {
                flag
                for page in page_results
                for flag in page.semantic.risk_flags
                if flag.strip()
            }
        )
        reasons = [
            page.semantic.reasoning
            for page in page_results
            if page.semantic.reasoning and page.semantic.semantic_score < 1.0
        ]
        if risk_flags:
            reasons.append(f"AI risk flags: {', '.join(risk_flags)}")
        if any(page.nsfw_detected for page in page_results):
            reasons.append("NSFW or unsafe visual content detected")

        return story.story_id, AIResult(
            semantic_score=semantic_score,
            semantic_confidence=semantic_confidence,
            vision_confidence=vision_confidence,
            ocr_confidence=ocr_confidence,
            confidence_score=confidence_score,
            risk_flags=risk_flags,
            reasons=reasons,
            nsfw_detected=any(page.nsfw_detected for page in page_results),
        )

    async def _validate_page(self, story: NormalizedStory, page: NormalizedPage) -> PageAIResult:
        image_sources = await self._image_sources(page)

        ocr_results, vision_results = await asyncio.gather(
            asyncio.gather(*(self.ocr.extract(source) for source in image_sources)),
            asyncio.gather(*(self.vision.understand(source) for source in image_sources)),
        )

        ocr_text = "\n".join(result.text for result in ocr_results if result.text)
        ocr_confidence = min((result.confidence for result in ocr_results), default=0.0)
        vision_description = "\n".join(
            result.description for result in vision_results if result.description
        )
        vision_tags = tuple(
            sorted({tag for result in vision_results for tag in result.tags if tag.strip()})
        )
        vision_confidence = min((result.confidence for result in vision_results), default=0.0)

        semantic = await self.semantic.validate(
            story=story,
            page=page,
            ocr_text=ocr_text,
            vision_tags=vision_tags,
            vision_description=vision_description,
        )

        return PageAIResult(
            ocr_text=ocr_text,
            ocr_confidence=ocr_confidence,
            vision_description=vision_description,
            vision_tags=vision_tags,
            vision_confidence=vision_confidence,
            semantic=semantic,
            nsfw_detected=any(result.nsfw_detected for result in vision_results),
        )

    async def _image_sources(self, page: NormalizedPage) -> tuple[str, ...]:
        if page.type == "image":
            return (page.asset_url,)

        frames = await self.video_sampler.sample(page.asset_url)
        if not frames:
            raise AIServiceError(f"{page.internal_id}: video frame sampling failed")
        return frames


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
