from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Content Quality Gateway"
    api_version: str = "0.1.0"

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("CONTENT_GATEWAY_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_ocr_model: str = "gpt-4o-mini"
    openai_vision_model: str = "gpt-4o-mini"
    openai_semantic_model: str = "gpt-4o-mini"

    asset_timeout_seconds: float = 2.0
    asset_max_download_bytes: int = 5_000_000
    ai_timeout_seconds: float = 4.0
    request_timeout_seconds: float = 5.0

    max_stories_per_request: int = Field(default=50, ge=1)
    max_pages_per_story: int = Field(default=20, ge=1)
    max_video_frames: int = Field(default=2, ge=1, le=5)

    ai_confidence_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    final_score_threshold: float = Field(default=85.0, ge=0.0, le=100.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CONTENT_GATEWAY_",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
