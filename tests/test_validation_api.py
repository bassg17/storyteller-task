from __future__ import annotations

from copy import deepcopy

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.normalizer import normalize_payload
from app.core.validator import ContentValidator
from app.models.schemas import StoryPayload
from app.main import create_app
from app.ai.video import VideoFrame
from app.models.internal import AIResult, OcrResult, SafetyResult, SemanticResult, VisionResult


class PassingAIPipeline:
    async def validate(self, normalized):
        return {
            story.story_id: AIResult(
                semantic_score=1.0,
                semantic_confidence=0.95,
                vision_confidence=0.95,
                ocr_confidence=0.95,
                confidence_score=0.95,
            )
            for story in normalized.stories
        }


class FailingAIPipeline:
    async def validate(self, normalized):
        raise RuntimeError("provider unavailable")


class StaticOcrService:
    async def extract(self, image_source):
        return OcrResult(text="", confidence=0.96)


class PersonVisionService:
    async def understand(self, image_source):
        return VisionResult(
            description="A person sitting on a chair indoors.",
            tags=("person", "chair", "indoors"),
            confidence=0.96,
        )


class MediaMismatchSemanticService:
    async def validate(
        self,
        story,
        page,
        ocr_text,
        vision_tags,
        vision_description,
        media_context,
    ):
        return SemanticResult(
            semantic_score=0.4,
            confidence=0.93,
            risk_flags=(),
            reasoning="The visual content does not support the football story.",
            media_matches_title=False,
            media_matches_categories=False,
            cta_matches_url=True,
            cta_matches_title=False,
            media_reasoning=(
                "The image shows a person sitting, not football teams or match history."
            ),
        )


class NoopVideoSampler:
    async def sample(self, video_url):
        return ()


class SafeSafetyService:
    async def moderate(self, image_url, label):
        return SafetyResult(flagged=False, confidence=0.99)


class ExplicitSafetyService:
    async def moderate(self, image_url, label):
        return SafetyResult(
            flagged=True,
            categories=("sexual",),
            category_scores={"sexual": 0.96},
            confidence=0.96,
            reasons=[f"Explicit sexual content detected in {label}"],
        )


class FailingSafetyService:
    async def moderate(self, image_url, label):
        raise RuntimeError("moderation unavailable")


class StaticVideoSampler:
    async def sample(self, video_url):
        return (
            VideoFrame(
                frame_index=0,
                source_position=12,
                data_url="data:image/jpeg;base64,ZmFrZS1mcmFtZQ==",
            ),
            VideoFrame(
                frame_index=1,
                source_position=48,
                data_url="data:image/jpeg;base64,ZmFrZS1mcmFtZS0y",
            ),
        )


def test_validate_approves_when_rules_and_ai_pass() -> None:
    app = create_app()
    app.state.validator_factory = lambda settings: ContentValidator(
        settings,
        ai_pipeline=PassingAIPipeline(),
    )

    with respx.mock:
        respx.get("https://cdn.example.com/story/page.jpg").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "image/jpeg"},
                content=b"unique-image",
            )
        )
        response = TestClient(app).post("/validate", json=_payload())

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["decision"] == "APPROVE"
    assert result["quality_score"] >= 85
    assert result["confidence_score"] == 0.95


def test_validate_allows_duplicate_assets_within_tenant_batch() -> None:
    payload = _payload()
    duplicate_page = deepcopy(payload["stories"][0]["pages"][0])
    duplicate_page["page_id"] = "page_2"
    payload["stories"][0]["pages"].append(duplicate_page)

    second_story = deepcopy(payload["stories"][0])
    second_story["story_id"] = "story_2"
    second_story["pages"][0]["page_id"] = "page_3"
    payload["stories"].append(second_story)

    app = create_app()
    app.state.validator_factory = lambda settings: ContentValidator(
        settings,
        ai_pipeline=PassingAIPipeline(),
    )

    with respx.mock:
        respx.get("https://cdn.example.com/story/page.jpg").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "image/jpeg"},
                content=b"same-image",
            )
        )
        response = TestClient(app).post("/validate", json=payload)

    assert response.status_code == 200
    results = response.json()["results"]
    assert {result["decision"] for result in results} == {"APPROVE"}
    assert all("duplicate asset detected" not in " ".join(result["reasons"]) for result in results)


def test_validate_fails_closed_when_ai_pipeline_errors() -> None:
    app = create_app()
    app.state.validator_factory = lambda settings: ContentValidator(
        settings,
        ai_pipeline=FailingAIPipeline(),
    )

    with respx.mock:
        respx.get("https://cdn.example.com/story/page.jpg").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "image/jpeg"},
                content=b"unique-image",
            )
        )
        response = TestClient(app).post("/validate", json=_payload())

    result = response.json()["results"][0]
    assert result["decision"] == "REJECT"
    assert "AI confidence 0.00 is below 0.90" in result["reasons"]
    assert any("AI validation failed closed" in reason for reason in result["reasons"])


def test_invalid_schema_returns_400() -> None:
    response = TestClient(create_app()).post("/validate", json={"tenant_id": "tenant_1"})

    assert response.status_code == 400


def test_private_asset_url_is_rejected_without_fetching() -> None:
    payload = _payload()
    payload["stories"][0]["pages"][0]["asset_url"] = "http://127.0.0.1/private.jpg"

    app = create_app()
    app.state.validator_factory = lambda settings: ContentValidator(
        settings,
        ai_pipeline=PassingAIPipeline(),
    )
    response = TestClient(app).post("/validate", json=payload)

    result = response.json()["results"][0]
    assert result["decision"] == "REJECT"
    assert any("asset URL is invalid or unsafe" in reason for reason in result["reasons"])


def test_openai_provider_requires_api_key() -> None:
    from app.ai.pipeline import AIPipeline

    with pytest.raises(RuntimeError, match="OpenAI API key is not configured"):
        AIPipeline.from_settings(Settings(_env_file=None))


async def test_ai_pipeline_surfaces_media_mismatch_reasons() -> None:
    from app.ai.pipeline import AIPipeline

    payload = _payload()
    payload["stories"][0]["story_title"] = "Last 5 meetings: Penguin FC vs Seals United"
    payload["stories"][0]["context"]["categories"] = [
        "Penguin FC",
        "Seals United",
        "Matchday",
    ]
    normalized = normalize_payload(
        StoryPayload.model_validate(payload),
        Settings(openai_api_key="test", _env_file=None),
    )
    pipeline = AIPipeline(
        ocr=StaticOcrService(),
        vision=PersonVisionService(),
        semantic=MediaMismatchSemanticService(),
        safety=SafeSafetyService(),
        video_sampler=NoopVideoSampler(),
    )

    results = await pipeline.validate(normalized)
    ai_result = results["story_1"]

    assert "media_title_mismatch" in ai_result.risk_flags
    assert "media_category_mismatch" in ai_result.risk_flags
    assert "cta_title_mismatch" in ai_result.risk_flags
    assert any("media-title mismatch" in reason for reason in ai_result.reasons)
    assert any("person sitting" in reason for reason in ai_result.reasons)
    assert any("vision summary" in reason for reason in ai_result.reasons)


async def test_ai_pipeline_surfaces_video_frame_mismatch_reasons() -> None:
    from app.ai.pipeline import AIPipeline

    payload = _payload()
    payload["stories"][0]["story_title"] = "Last 5 meetings: Penguin FC vs Seals United"
    payload["stories"][0]["pages"][0]["type"] = "video"
    payload["stories"][0]["pages"][0]["asset_url"] = "https://cdn.example.com/story/page.mp4"
    payload["stories"][0]["context"]["categories"] = [
        "Penguin FC",
        "Seals United",
        "Matchday",
    ]
    normalized = normalize_payload(
        StoryPayload.model_validate(payload),
        Settings(openai_api_key="test", _env_file=None),
    )
    pipeline = AIPipeline(
        ocr=StaticOcrService(),
        vision=PersonVisionService(),
        semantic=MediaMismatchSemanticService(),
        safety=SafeSafetyService(),
        video_sampler=StaticVideoSampler(),
    )

    results = await pipeline.validate(normalized)
    ai_result = results["story_1"]

    assert "media_title_mismatch" in ai_result.risk_flags
    assert "media_category_mismatch" in ai_result.risk_flags
    assert any("video-title mismatch" in reason for reason in ai_result.reasons)
    assert any("sampled video frame 0" in reason for reason in ai_result.reasons)
    assert any("video asset sampled into 2 frame(s)" in reason for reason in ai_result.reasons)


async def test_ai_pipeline_fails_closed_when_video_sampling_fails() -> None:
    from app.ai.pipeline import AIPipeline

    payload = _payload()
    payload["stories"][0]["pages"][0]["type"] = "video"
    payload["stories"][0]["pages"][0]["asset_url"] = "https://cdn.example.com/story/page.mp4"
    normalized = normalize_payload(
        StoryPayload.model_validate(payload),
        Settings(openai_api_key="test", _env_file=None),
    )
    pipeline = AIPipeline(
        ocr=StaticOcrService(),
        vision=PersonVisionService(),
        semantic=MediaMismatchSemanticService(),
        safety=SafeSafetyService(),
        video_sampler=NoopVideoSampler(),
    )

    results = await pipeline.validate(normalized)
    ai_result = results["story_1"]

    assert ai_result.confidence_score == 0.0
    assert any("video frame sampling failed" in reason for reason in ai_result.reasons)


async def test_ai_pipeline_rejects_explicit_image_even_when_semantically_matching() -> None:
    from app.ai.pipeline import AIPipeline

    normalized = normalize_payload(
        StoryPayload.model_validate(_payload()),
        Settings(openai_api_key="test", _env_file=None),
    )
    pipeline = AIPipeline(
        ocr=StaticOcrService(),
        vision=PersonVisionService(),
        semantic=MediaMismatchSemanticService(),
        safety=ExplicitSafetyService(),
        video_sampler=NoopVideoSampler(),
    )

    results = await pipeline.validate(normalized)
    ai_result = results["story_1"]

    assert ai_result.nsfw_detected is True
    assert "unsafe_content" in ai_result.risk_flags
    assert any("Explicit sexual content detected in image asset" in reason for reason in ai_result.reasons)


async def test_ai_pipeline_rejects_explicit_video_frame() -> None:
    from app.ai.pipeline import AIPipeline

    payload = _payload()
    payload["stories"][0]["pages"][0]["type"] = "video"
    payload["stories"][0]["pages"][0]["asset_url"] = "https://cdn.example.com/story/page.mp4"
    normalized = normalize_payload(
        StoryPayload.model_validate(payload),
        Settings(openai_api_key="test", _env_file=None),
    )
    pipeline = AIPipeline(
        ocr=StaticOcrService(),
        vision=PersonVisionService(),
        semantic=MediaMismatchSemanticService(),
        safety=ExplicitSafetyService(),
        video_sampler=StaticVideoSampler(),
    )

    results = await pipeline.validate(normalized)
    ai_result = results["story_1"]

    assert ai_result.nsfw_detected is True
    assert "unsafe_content" in ai_result.risk_flags
    assert any("Explicit sexual content detected in sampled video frame" in reason for reason in ai_result.reasons)


async def test_ai_pipeline_fails_closed_when_moderation_fails() -> None:
    from app.ai.pipeline import AIPipeline

    normalized = normalize_payload(
        StoryPayload.model_validate(_payload()),
        Settings(openai_api_key="test", _env_file=None),
    )
    pipeline = AIPipeline(
        ocr=StaticOcrService(),
        vision=PersonVisionService(),
        semantic=MediaMismatchSemanticService(),
        safety=FailingSafetyService(),
        video_sampler=NoopVideoSampler(),
    )

    results = await pipeline.validate(normalized)
    ai_result = results["story_1"]

    assert ai_result.confidence_score == 0.0
    assert any("AI validation failed closed" in reason for reason in ai_result.reasons)


@pytest.mark.parametrize(
    ("semantic_score", "confidence", "expected_decision"),
    [
        (1.0, 0.95, "APPROVE"),
        (1.0, 0.89, "REJECT"),
        (0.4, 0.95, "REJECT"),
    ],
)
async def test_policy_thresholds(semantic_score, confidence, expected_decision) -> None:
    from app.core.decision_engine import apply_policy
    from app.core.scorer import compute_score
    from app.models.internal import RuleResult

    rule_result = RuleResult(rule_score=100.0)
    ai_result = AIResult(
        semantic_score=semantic_score,
        semantic_confidence=confidence,
        vision_confidence=confidence,
        ocr_confidence=confidence,
        confidence_score=confidence,
    )
    score = compute_score(rule_result, ai_result)
    decision, _ = apply_policy(rule_result, ai_result, score, Settings(openai_api_key="test"))

    assert decision == expected_decision


def _payload() -> dict:
    return {
        "tenant_id": "tenant_1",
        "tenant_name": "Test Tenant",
        "last_synced_at": "2026-02-14T10:05:00Z",
        "stories": [
            {
                "story_id": "story_1",
                "story_title": "Championship highlights",
                "pages": [
                    {
                        "page_id": "page_1",
                        "type": "image",
                        "asset_url": "https://cdn.example.com/story/page.jpg",
                        "action": {
                            "cta": "Watch highlights",
                            "url": "https://example.com/highlights",
                        },
                    }
                ],
                "context": {
                    "categories": ["sports", "highlights"],
                    "tenant": "Test Tenant",
                    "publish_date": "2026-02-14",
                },
            }
        ],
    }
