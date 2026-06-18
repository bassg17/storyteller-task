from app.core.config import Settings


def test_request_limit_defaults_when_env_vars_are_absent(monkeypatch) -> None:
    monkeypatch.delenv("CONTENT_GATEWAY_MAX_STORIES_PER_REQUEST", raising=False)
    monkeypatch.delenv("CONTENT_GATEWAY_MAX_PAGES_PER_STORY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.max_stories_per_request == 50
    assert settings.max_pages_per_story == 20


def test_request_limits_can_be_configured_from_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("CONTENT_GATEWAY_MAX_STORIES_PER_REQUEST", "10")
    monkeypatch.setenv("CONTENT_GATEWAY_MAX_PAGES_PER_STORY", "5")

    settings = Settings(_env_file=None)

    assert settings.max_stories_per_request == 10
    assert settings.max_pages_per_story == 5
