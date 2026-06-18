from app.core.config import Settings
from app.models.internal import AIResult, RuleResult, ScoreResult
from app.models.schemas import Decision


def apply_policy(
    rule_result: RuleResult,
    ai_result: AIResult,
    score: ScoreResult,
    settings: Settings,
) -> tuple[Decision, list[str]]:
    rejection_reasons: list[str] = []

    if rule_result.critical_failures:
        rejection_reasons.extend(rule_result.critical_failures)

    if ai_result.nsfw_detected:
        rejection_reasons.append("NSFW detected")

    if ai_result.confidence_score < settings.ai_confidence_threshold:
        rejection_reasons.append(
            f"AI confidence {ai_result.confidence_score:.2f} is below "
            f"{settings.ai_confidence_threshold:.2f}"
        )

    if score.quality_score < settings.final_score_threshold:
        rejection_reasons.append(
            f"quality score {score.quality_score:.0f} is below "
            f"{settings.final_score_threshold:.0f}"
        )

    if rejection_reasons:
        return "REJECT", _dedupe(rejection_reasons + ai_result.reasons + rule_result.warnings)
    return "APPROVE", _dedupe(rule_result.warnings)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        clean_value = value.strip()
        if clean_value and clean_value not in seen:
            deduped.append(clean_value)
            seen.add(clean_value)
    return deduped
