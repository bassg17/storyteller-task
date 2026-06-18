from __future__ import annotations

import asyncio
import hashlib
import ipaddress
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from app.core.config import Settings
from app.models.internal import NormalizedPage, NormalizedPayload, NormalizedStory, RuleResult


@dataclass(frozen=True)
class AssetCheck:
    page: NormalizedPage
    status_code: int | None
    content_type: str
    sha256: str | None
    error: str | None = None


class RuleEngine:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client

    async def validate(self, payload: NormalizedPayload) -> dict[str, RuleResult]:
        results = {
            story.story_id: self._validate_metadata_and_actions(story)
            for story in payload.stories
        }

        pages = [page for story in payload.stories for page in story.pages]
        asset_checks = await self._check_assets(pages)
        self._apply_asset_results(results, asset_checks)
        self._apply_duplicate_results(results, asset_checks)

        for result in results.values():
            result.rule_score = self._score(result)

        return results

    def _validate_metadata_and_actions(self, story: NormalizedStory) -> RuleResult:
        result = RuleResult()

        if not story.story_title:
            result.critical_failures.append("missing story title")
        if not story.categories:
            result.critical_failures.append("missing categories")
        if not story.publish_date:
            result.warnings.append("missing publish date")

        for page in story.pages:
            if _is_malicious_or_invalid_http_url(page.asset_url):
                result.critical_failures.append(
                    f"{page.internal_id}: asset URL is invalid or unsafe"
                )

            if page.action is None:
                result.warnings.append(f"{page.internal_id}: missing action")
                continue

            if not page.action.cta:
                result.critical_failures.append(f"{page.internal_id}: missing CTA")
            if _is_malicious_or_invalid_http_url(page.action.url):
                result.critical_failures.append(
                    f"{page.internal_id}: action URL is invalid or unsafe"
                )

        return result

    async def _check_assets(self, pages: list[NormalizedPage]) -> list[AssetCheck]:
        owned_client = self.client is None
        client = self.client or httpx.AsyncClient(
            follow_redirects=True,
            timeout=self.settings.asset_timeout_seconds,
        )
        try:
            return await asyncio.gather(*(self._check_asset(client, page) for page in pages))
        finally:
            if owned_client:
                await client.aclose()

    async def _check_asset(self, client: httpx.AsyncClient, page: NormalizedPage) -> AssetCheck:
        if _is_malicious_or_invalid_http_url(page.asset_url):
            return AssetCheck(page=page, status_code=None, content_type="", sha256=None)

        try:
            response = await client.get(page.asset_url)
        except httpx.HTTPError as exc:
            return AssetCheck(
                page=page,
                status_code=None,
                content_type="",
                sha256=None,
                error=f"asset fetch failed: {exc.__class__.__name__}",
            )

        content_type = response.headers.get("content-type", "").split(";")[0].lower()
        content = response.content
        if len(content) > self.settings.asset_max_download_bytes:
            return AssetCheck(
                page=page,
                status_code=response.status_code,
                content_type=content_type,
                sha256=None,
                error="asset exceeds maximum validation download size",
            )

        digest = hashlib.sha256(content).hexdigest() if response.status_code == 200 else None
        return AssetCheck(
            page=page,
            status_code=response.status_code,
            content_type=content_type,
            sha256=digest,
        )

    def _apply_asset_results(
        self, results: dict[str, RuleResult], asset_checks: list[AssetCheck]
    ) -> None:
        for check in asset_checks:
            result = results[check.page.story_id]
            if check.status_code != 200:
                detail = check.error or f"asset returned HTTP {check.status_code}"
                result.critical_failures.append(f"{check.page.internal_id}: {detail}")
                continue

            expected_prefix = f"{check.page.type}/"
            if not check.content_type.startswith(expected_prefix):
                result.critical_failures.append(
                    f"{check.page.internal_id}: expected {check.page.type} MIME type, "
                    f"got {check.content_type or 'unknown'}"
                )

            if check.error:
                result.critical_failures.append(f"{check.page.internal_id}: {check.error}")

    def _apply_duplicate_results(
        self, results: dict[str, RuleResult], asset_checks: list[AssetCheck]
    ) -> None:
        hashes: dict[str, list[NormalizedPage]] = {}
        for check in asset_checks:
            if check.sha256:
                hashes.setdefault(check.sha256, []).append(check.page)

        for duplicate_pages in hashes.values():
            if len(duplicate_pages) < 2:
                continue
            page_refs = ", ".join(page.internal_id for page in duplicate_pages)
            for page in duplicate_pages:
                results[page.story_id].critical_failures.append(
                    f"{page.internal_id}: duplicate asset detected in tenant batch ({page_refs})"
                )

    def _score(self, result: RuleResult) -> float:
        score = 100.0
        score -= 25.0 * len(result.critical_failures)
        score -= 5.0 * len(result.warnings)
        return max(0.0, min(100.0, score))


def _is_malicious_or_invalid_http_url(raw_url: str) -> bool:
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        return True
    if any(char in raw_url for char in ("\n", "\r", "\t", "\x00")):
        return True
    if parsed.username or parsed.password:
        return True

    hostname = parsed.hostname.lower()
    if hostname in {"localhost", "metadata.google.internal"}:
        return True

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return False

    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
