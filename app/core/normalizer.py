from app.core.config import Settings
from app.models.internal import (
    NormalizedAction,
    NormalizedPage,
    NormalizedPayload,
    NormalizedStory,
)
from app.models.schemas import StoryPayload


class PayloadLimitError(ValueError):
    """Raised when a structurally valid request exceeds service limits."""


def normalize_payload(payload: StoryPayload, settings: Settings) -> NormalizedPayload:
    if len(payload.stories) > settings.max_stories_per_request:
        raise PayloadLimitError(
            f"request contains {len(payload.stories)} stories; "
            f"maximum is {settings.max_stories_per_request}"
        )

    normalized_stories: list[NormalizedStory] = []
    for story in payload.stories:
        if len(story.pages) > settings.max_pages_per_story:
            raise PayloadLimitError(
                f"story {story.story_id} contains {len(story.pages)} pages; "
                f"maximum is {settings.max_pages_per_story}"
            )

        pages = tuple(
            NormalizedPage(
                story_id=story.story_id,
                page_id=page.page_id,
                page_index=index,
                type=page.type,
                asset_url=page.asset_url.strip(),
                action=(
                    NormalizedAction(cta=page.action.cta.strip(), url=page.action.url.strip())
                    if page.action
                    else None
                ),
            )
            for index, page in enumerate(story.pages)
        )
        normalized_stories.append(
            NormalizedStory(
                tenant_id=payload.tenant_id,
                tenant_name=payload.tenant_name,
                story_id=story.story_id,
                story_title=story.story_title.strip(),
                categories=tuple(category.strip() for category in story.context.categories),
                publish_date=story.context.publish_date.strip(),
                pages=pages,
            )
        )

    return NormalizedPayload(source=payload, stories=tuple(normalized_stories))
