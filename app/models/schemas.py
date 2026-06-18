from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


Decision = Literal["APPROVE", "REJECT"]
MediaType = Literal["image", "video"]

NonEmptyStr = Annotated[str, Field(min_length=1)]


class Action(BaseModel):
    cta: NonEmptyStr
    url: NonEmptyStr


class StoryContext(BaseModel):
    categories: list[NonEmptyStr] = Field(min_length=1)
    publish_date: NonEmptyStr
    tenant: str | None = None

    model_config = ConfigDict(extra="allow")


class Page(BaseModel):
    page_id: NonEmptyStr
    type: MediaType
    asset_url: NonEmptyStr
    action: Action | None = None


class Story(BaseModel):
    story_id: NonEmptyStr
    story_title: NonEmptyStr
    pages: list[Page] = Field(min_length=1)
    context: StoryContext


class StoryPayload(BaseModel):
    tenant_id: NonEmptyStr
    tenant_name: NonEmptyStr
    last_synced_at: datetime
    stories: list[Story] = Field(min_length=1)

    @field_validator("stories")
    @classmethod
    def story_ids_must_be_unique(cls, stories: list[Story]) -> list[Story]:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for story in stories:
            if story.story_id in seen:
                duplicates.add(story.story_id)
            seen.add(story.story_id)
        if duplicates:
            raise ValueError(f"duplicate story_id values: {', '.join(sorted(duplicates))}")
        return stories


class StoryValidationResult(BaseModel):
    story_id: str
    decision: Decision
    quality_score: int = Field(ge=0, le=100)
    confidence_score: float = Field(ge=0.0, le=1.0)
    rule_score: int = Field(ge=0, le=100)
    ai_score: int = Field(ge=0, le=100)
    reasons: list[str]


class ValidationResponse(BaseModel):
    request_id: UUID
    results: list[StoryValidationResult]


class ApiError(BaseModel):
    detail: Any
