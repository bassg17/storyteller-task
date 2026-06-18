from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.ai.client import AIServiceError, build_openai_client, format_ai_exception
from app.ai.llm import SemanticValidationService
from app.ai.ocr import OcrService
from app.ai.safety import SafetyModerationService
from app.ai.video import VideoFrame, VideoFrameSampler
from app.ai.vision import VisionService
from app.core.config import Settings
from app.models.internal import (
    AIResult,
    NormalizedPage,
    NormalizedPayload,
    NormalizedStory,
    OcrResult,
    SafetyResult,
    SemanticResult,
    VisionResult,
)


@dataclass(frozen=True)
class MediaSource:
    value: str
    label: str
    media_type: str
    frame: VideoFrame | None = None


@dataclass(frozen=True)
class PageAIResult:
    page: NormalizedPage
    ocr_text: str
    ocr_confidence: float
    vision_description: str
    vision_tags: tuple[str, ...]
    vision_confidence: float
    media_context: str
    semantic: SemanticResult
    nsfw_detected: bool
    safety_results: tuple[SafetyResult, ...]


class AIPipeline:
    def __init__(
        self,
        ocr: OcrService,
        vision: VisionService,
        semantic: SemanticValidationService,
        safety: SafetyModerationService,
        video_sampler: VideoFrameSampler,
    ) -> None:
        self.ocr = ocr
        self.vision = vision
        self.semantic = semantic
        self.safety = safety
        self.video_sampler = video_sampler

    @classmethod
    def from_settings(cls, settings: Settings) -> AIPipeline:
        client = build_openai_client(settings)
        return cls(
            ocr=OcrService(client, settings),
            vision=VisionService(client, settings),
            semantic=SemanticValidationService(client, settings),
            safety=SafetyModerationService(client, settings),
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
                reasons=[f"AI validation failed closed: {format_ai_exception(exc)}"],
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

        risk_flags = sorted({flag for page in page_results for flag in _page_risk_flags(page)})
        reasons = [
            page.semantic.reasoning
            for page in page_results
            if page.semantic.reasoning and page.semantic.semantic_score < 1.0
        ]
        reasons.extend(
            reason
            for page in page_results
            for reason in _page_consistency_reasons(page)
        )
        reasons.extend(
            reason
            for page in page_results
            for result in page.safety_results
            for reason in result.reasons
        )
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
            safety_results=[
                result
                for page in page_results
                for result in page.safety_results
            ],
        )

    async def _validate_page(self, story: NormalizedStory, page: NormalizedPage) -> PageAIResult:
        media_sources = await self._media_sources(page)
        media_context = _media_context(page, media_sources)
        safety_results = await asyncio.gather(
            *(self.safety.moderate(source.value, source.label) for source in media_sources)
        )
        if any(result.flagged for result in safety_results):
            return PageAIResult(
                page=page,
                ocr_text="",
                ocr_confidence=1.0,
                vision_description="",
                vision_tags=("unsafe_content",),
                vision_confidence=_min_safety_confidence(safety_results),
                media_context=media_context,
                semantic=SemanticResult(
                    semantic_score=0.0,
                    confidence=1.0,
                    risk_flags=("unsafe_content",),
                    reasoning="Explicit content safety gate rejected this media.",
                ),
                nsfw_detected=True,
                safety_results=tuple(safety_results),
            )

        ocr_results, vision_results = await asyncio.gather(
            asyncio.gather(*(self.ocr.extract(source.value) for source in media_sources)),
            asyncio.gather(*(self.vision.understand(source.value) for source in media_sources)),
        )

        ocr_text = "\n".join(result.text for result in ocr_results if result.text)
        ocr_confidence = min((result.confidence for result in ocr_results), default=0.0)
        vision_description = "\n".join(
            _label_vision_description(source, result.description)
            for source, result in zip(media_sources, vision_results, strict=True)
            if result.description
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
            media_context=media_context,
        )

        return PageAIResult(
            page=page,
            ocr_text=ocr_text,
            ocr_confidence=ocr_confidence,
            vision_description=vision_description,
            vision_tags=vision_tags,
            vision_confidence=vision_confidence,
            media_context=media_context,
            semantic=semantic,
            nsfw_detected=any(result.nsfw_detected for result in vision_results)
            or any(result.flagged for result in safety_results),
            safety_results=tuple(safety_results),
        )

    async def _media_sources(self, page: NormalizedPage) -> tuple[MediaSource, ...]:
        if page.type == "image":
            return (
                MediaSource(
                    value=page.asset_url,
                    label="image asset",
                    media_type="image",
                ),
            )

        frames = await self.video_sampler.sample(page.asset_url)
        if not frames:
            raise AIServiceError(f"{page.internal_id}: video frame sampling failed")
        return tuple(
            MediaSource(
                value=frame.data_url,
                label=f"sampled video frame {frame.frame_index} at source frame {frame.source_position}",
                media_type="video_frame",
                frame=frame,
            )
            for frame in frames
        )


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _min_safety_confidence(safety_results: tuple[SafetyResult, ...] | list[SafetyResult]) -> float:
    if not safety_results:
        return 1.0
    return min((result.confidence for result in safety_results), default=1.0)


def _page_risk_flags(page: PageAIResult) -> tuple[str, ...]:
    flags = {flag for flag in page.semantic.risk_flags if flag.strip()}
    if not page.semantic.media_matches_title:
        flags.add("media_title_mismatch")
    if not page.semantic.media_matches_categories:
        flags.add("media_category_mismatch")
    if not page.semantic.cta_matches_url:
        flags.add("cta_url_mismatch")
    if not page.semantic.cta_matches_title:
        flags.add("cta_title_mismatch")
    if page.nsfw_detected:
        flags.add("unsafe_content")
    if any(result.flagged for result in page.safety_results):
        flags.add("unsafe_content")
    if page.vision_confidence < 0.9:
        flags.add("low_visual_confidence")
    return tuple(sorted(flags))


def _page_consistency_reasons(page: PageAIResult) -> tuple[str, ...]:
    reasons: list[str] = []
    visual_context = _visual_context(page)

    if not page.semantic.media_matches_title:
        mismatch_label = "video-title mismatch" if page.page.type == "video" else "media-title mismatch"
        fallback = (
            "sampled video frames do not match page story title"
            if page.page.type == "video"
            else "visual content does not match page story title"
        )
        reason = _with_visual_context(
            page.semantic.media_reasoning or fallback,
            visual_context,
        )
        reasons.append(f"{page.page.internal_id}: {mismatch_label}: {reason}")

    if not page.semantic.media_matches_categories:
        mismatch_label = (
            "video-category mismatch" if page.page.type == "video" else "media-category mismatch"
        )
        fallback = (
            "sampled video frames do not match story categories"
            if page.page.type == "video"
            else "visual content does not match story categories"
        )
        reason = _with_visual_context(
            page.semantic.media_reasoning or fallback,
            visual_context,
        )
        reasons.append(f"{page.page.internal_id}: {mismatch_label}: {reason}")

    if not page.semantic.cta_matches_url:
        reasons.append(f"{page.page.internal_id}: CTA does not match destination URL intent")

    if not page.semantic.cta_matches_title:
        reasons.append(f"{page.page.internal_id}: CTA does not match story title intent")

    return tuple(reasons)


def _visual_context(page: PageAIResult) -> str:
    parts: list[str] = []
    if page.media_context:
        parts.append(f"media context: {page.media_context}")
    if page.vision_description:
        parts.append(f"vision summary: {page.vision_description}")
    if page.vision_tags:
        parts.append(f"vision tags: {', '.join(page.vision_tags)}")
    return "; ".join(parts) or "no vision summary was available"


def _with_visual_context(reason: str, visual_context: str) -> str:
    if visual_context in reason:
        return reason
    return f"{reason} ({visual_context})"


def _label_vision_description(source: MediaSource, description: str) -> str:
    if source.media_type == "video_frame":
        return f"{source.label}: {description}"
    return description


def _media_context(page: NormalizedPage, media_sources: tuple[MediaSource, ...]) -> str:
    if page.type == "image":
        return "image asset"
    labels = ", ".join(source.label for source in media_sources)
    return f"video asset sampled into {len(media_sources)} frame(s): {labels}"
