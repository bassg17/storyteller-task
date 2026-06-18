from __future__ import annotations

from dataclasses import dataclass, field

from app.models.schemas import MediaType, StoryPayload


@dataclass(frozen=True)
class NormalizedAction:
    cta: str
    url: str


@dataclass(frozen=True)
class NormalizedPage:
    story_id: str
    page_id: str
    page_index: int
    type: MediaType
    asset_url: str
    action: NormalizedAction | None

    @property
    def internal_id(self) -> str:
        return f"{self.story_id}:{self.page_index}:{self.page_id}"


@dataclass(frozen=True)
class NormalizedStory:
    tenant_id: str
    tenant_name: str
    story_id: str
    story_title: str
    categories: tuple[str, ...]
    publish_date: str
    pages: tuple[NormalizedPage, ...]


@dataclass(frozen=True)
class NormalizedPayload:
    source: StoryPayload
    stories: tuple[NormalizedStory, ...]


@dataclass
class RuleResult:
    rule_score: float = 100.0
    critical_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OcrResult:
    text: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class VisionResult:
    description: str = ""
    tags: tuple[str, ...] = ()
    confidence: float = 0.0
    nsfw_detected: bool = False


@dataclass(frozen=True)
class SemanticResult:
    semantic_score: float = 0.0
    confidence: float = 0.0
    risk_flags: tuple[str, ...] = ()
    reasoning: str = ""
    media_matches_title: bool = True
    media_matches_categories: bool = True
    cta_matches_url: bool = True
    cta_matches_title: bool = True
    media_reasoning: str = ""


@dataclass
class SafetyResult:
    flagged: bool = False
    categories: tuple[str, ...] = ()
    category_scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class AIResult:
    semantic_score: float = 0.0
    semantic_confidence: float = 0.0
    vision_confidence: float = 0.0
    ocr_confidence: float = 0.0
    confidence_score: float = 0.0
    risk_flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    nsfw_detected: bool = False
    safety_results: list[SafetyResult] = field(default_factory=list)


@dataclass(frozen=True)
class ScoreResult:
    quality_score: float
    rule_score: float
    ai_score: float
    confidence_score: float
