from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from uuid import uuid4

from app.ai.client import format_ai_exception
from app.ai.pipeline import AIPipeline
from app.core.config import Settings
from app.core.decision_engine import apply_policy
from app.core.normalizer import normalize_payload
from app.core.scorer import compute_score
from app.models.internal import AIResult
from app.models.schemas import StoryPayload, StoryValidationResult, ValidationResponse
from app.rules.rule_engine import RuleEngine

logger = logging.getLogger(__name__)


class ContentValidator:
    def __init__(
        self,
        settings: Settings,
        rule_engine: RuleEngine | None = None,
        ai_pipeline: AIPipeline | None = None,
    ) -> None:
        self.settings = settings
        self.rule_engine = rule_engine or RuleEngine(settings)
        self.ai_pipeline = ai_pipeline

    async def validate(self, payload: StoryPayload) -> ValidationResponse:
        request_id = uuid4()
        started_at = perf_counter()
        normalized_started_at = perf_counter()
        normalized = normalize_payload(payload, self.settings)
        normalization_ms = _elapsed_ms(normalized_started_at)

        rules_started_at = perf_counter()
        rule_results = await self.rule_engine.validate(normalized)
        rules_ms = _elapsed_ms(rules_started_at)

        ai_started_at = perf_counter()
        ai_results = await self._run_ai(normalized)
        ai_ms = _elapsed_ms(ai_started_at)

        response_results: list[StoryValidationResult] = []
        for story in normalized.stories:
            rule_result = rule_results[story.story_id]
            ai_result = ai_results[story.story_id]
            score = compute_score(rule_result, ai_result)
            decision, reasons = apply_policy(rule_result, ai_result, score, self.settings)
            response_results.append(
                StoryValidationResult(
                    story_id=story.story_id,
                    decision=decision,
                    quality_score=round(score.quality_score),
                    confidence_score=round(score.confidence_score, 4),
                    rule_score=round(score.rule_score),
                    ai_score=round(score.ai_score),
                    reasons=reasons,
                )
            )

        logger.info(
            "content_validation_completed",
            extra={
                "request_id": str(request_id),
                "tenant_id": payload.tenant_id,
                "decisions": {
                    result.story_id: result.decision for result in response_results
                },
                "rule_failures": {
                    story_id: result.critical_failures
                    for story_id, result in rule_results.items()
                    if result.critical_failures
                },
                "latency_ms": {
                    "normalization": normalization_ms,
                    "rules": rules_ms,
                    "ai": ai_ms,
                    "total": _elapsed_ms(started_at),
                },
            },
        )

        return ValidationResponse(request_id=request_id, results=response_results)

    async def _run_ai(self, normalized) -> dict[str, AIResult]:
        try:
            pipeline = self.ai_pipeline or AIPipeline.from_settings(self.settings)
            return await asyncio.wait_for(
                pipeline.validate(normalized),
                timeout=self.settings.request_timeout_seconds,
            )
        except Exception as exc:
            reason = f"AI validation failed closed: {format_ai_exception(exc)}"
            return {
                story.story_id: AIResult(confidence_score=0.0, reasons=[reason])
                for story in normalized.stories
            }


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000.0, 2)
