from app.models.internal import AIResult, RuleResult, ScoreResult


def compute_score(rule_result: RuleResult, ai_result: AIResult) -> ScoreResult:
    ai_score = (
        ai_result.semantic_score * 70.0
        + ai_result.vision_confidence * 20.0
        + ai_result.ocr_confidence * 10.0
    )
    ai_score = max(0.0, min(100.0, ai_score))
    quality_score = rule_result.rule_score * 0.5 + ai_score * 0.5

    return ScoreResult(
        quality_score=max(0.0, min(100.0, quality_score)),
        rule_score=max(0.0, min(100.0, rule_result.rule_score)),
        ai_score=ai_score,
        confidence_score=max(0.0, min(1.0, ai_result.confidence_score)),
    )
