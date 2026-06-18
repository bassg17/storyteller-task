from __future__ import annotations

from copy import deepcopy

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.validator import ContentValidator
from app.main import create_app
from app.models.internal import AIResult


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


def test_validate_rejects_duplicate_assets_within_tenant_batch() -> None:
    payload = _payload()
    second_story = deepcopy(payload["stories"][0])
    second_story["story_id"] = "story_2"
    second_story["pages"][0]["page_id"] = "page_2"
    second_story["pages"][0]["asset_url"] = "https://cdn.example.com/story/page-2.jpg"
    payload["stories"].append(second_story)

    app = create_app()
    app.state.validator_factory = lambda settings: ContentValidator(
        settings,
        ai_pipeline=PassingAIPipeline(),
    )

    with respx.mock:
        for url in [
            "https://cdn.example.com/story/page.jpg",
            "https://cdn.example.com/story/page-2.jpg",
        ]:
            respx.get(url).mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "image/jpeg"},
                    content=b"same-image",
                )
            )
        response = TestClient(app).post("/validate", json=payload)

    assert response.status_code == 200
    results = response.json()["results"]
    assert {result["decision"] for result in results} == {"REJECT"}
    assert all("duplicate asset detected" in " ".join(result["reasons"]) for result in results)


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
