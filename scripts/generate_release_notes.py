"""Generate the XWhy release-notes page from published GitHub Releases."""

from __future__ import annotations

import html
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_NOTES_PAGE = PROJECT_ROOT / "docs" / "release-notes.md"

RELEASES_START = "<!-- AUTO-RELEASE-NOTES:START -->"
RELEASES_END = "<!-- AUTO-RELEASE-NOTES:END -->"


def _replace_block(text: str, replacement: str) -> str:
    """Replace the generated block while preserving the page introduction."""
    if text.count(RELEASES_START) != 1 or text.count(RELEASES_END) != 1:
        raise RuntimeError(
            f"Expected exactly one marker pair: {RELEASES_START} / {RELEASES_END}"
        )

    prefix, remainder = text.split(RELEASES_START, 1)
    _, suffix = remainder.split(RELEASES_END, 1)
    return f"{prefix}{RELEASES_START}\n{replacement.rstrip()}\n{RELEASES_END}{suffix}"


def _request_json(url: str, token: str | None) -> object:
    """Request JSON from GitHub with an optional workflow token."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "xwhy-documentation-build",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _fetch_releases(repository: str, token: str | None) -> list[dict[str, Any]]:
    """Fetch every published GitHub Release, newest first."""
    releases: list[dict[str, Any]] = []
    page = 1

    while True:
        url = (
            f"https://api.github.com/repos/{repository}/releases"
            f"?per_page=100&page={page}"
        )
        payload = _request_json(url, token)
        if not isinstance(payload, list):
            raise RuntimeError("GitHub Releases API returned an unexpected response.")

        for item in payload:
            if not isinstance(item, dict) or item.get("draft"):
                continue
            if not item.get("tag_name") or not item.get("html_url"):
                continue
            releases.append(item)

        if len(payload) < 100:
            break
        page += 1

    releases.sort(
        key=lambda release: str(
            release.get("published_at") or release.get("created_at") or ""
        ),
        reverse=True,
    )
    return releases


def _release_date(release: dict[str, Any]) -> str:
    """Format a GitHub timestamp as an unambiguous human-readable date."""
    value = str(release.get("published_at") or release.get("created_at") or "")
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "Unknown"
    return f"{date.day} {date.strftime('%B %Y')}"


def _release_anchor(tag: str) -> str:
    """Return a stable HTML anchor for a release tag."""
    slug = re.sub(r"[^a-z0-9]+", "-", tag.casefold()).strip("-")
    return f"release-{slug}"


def _table_text(value: str) -> str:
    """Escape dynamic text for a Markdown table cell."""
    return html.escape(value, quote=False).replace("|", "\\|")


def _demote_headings(body: str) -> str:
    """Keep release-body headings below each version heading."""
    body = body.replace("\r\n", "\n").replace("\r", "\n")

    # Some early releases contain the same GitHub-generated block twice.
    # Remove the second copy only when both blocks are line-for-line identical.
    starts = [
        match.start()
        for match in re.finditer(r"^## What's Changed\s*$", body, re.MULTILINE)
    ]
    if len(starts) == 2:
        first = body[starts[0] : starts[1]].strip().splitlines()
        second = body[starts[1] :].strip().splitlines()
        if first == second:
            body = body[: starts[1]].rstrip()

    def replace_heading(match: re.Match[str]) -> str:
        level = min(6, max(3, len(match.group(1)) + 1))
        return f"{'#' * level}{match.group(2)}"

    return re.sub(r"^(#{1,6})(\s+)", replace_heading, body, flags=re.MULTILINE)


def _render_releases(releases: list[dict[str, Any]], repository: str) -> str:
    """Render the release summary table and full release notes."""
    if not releases:
        return (
            "No published releases are available yet. "
            f"[View the repository]"
            f"(https://github.com/{repository}/releases)."
        )

    rows = [
        "## Release history",
        "",
        "| Release | Released | Type | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for release in releases:
        tag = str(release["tag_name"])
        release_url = str(release["html_url"])
        release_type = "Pre-release" if release.get("prerelease") else "Stable"
        rows.append(
            f"| [{_table_text(tag)}]({release_url}) "
            f"| {_release_date(release)} | {release_type} "
            f"| [Read notes](#{_release_anchor(tag)}) |"
        )

    latest = releases[0]
    latest_tag = str(latest["tag_name"])
    rows.extend(
        [
            "",
            (
                "For work merged since the latest release, see "
                f"[`{latest_tag}...main`]"
                f"(https://github.com/{repository}/compare/{latest_tag}...main)."
            ),
            "",
            "The complete notes for each release follow.",
        ]
    )

    for release in releases:
        tag = str(release["tag_name"])
        name = str(release.get("name") or tag).strip()
        release_url = str(release["html_url"])
        body = str(release.get("body") or "").strip()
        release_type = "Pre-release" if release.get("prerelease") else "Stable release"

        heading = tag if name == tag else f"{tag} — {name}"
        rows.extend(
            [
                "",
                f"## {_table_text(heading)} {{#{_release_anchor(tag)}}}",
                "",
                (
                    f"Released {_release_date(release)} · {release_type} · "
                    f"[GitHub release]({release_url}) · "
                    f"[PyPI](https://pypi.org/project/xwhy/)"
                ),
                "",
                _demote_headings(body)
                if body
                else "No release description was provided.",
            ]
        )

    return "\n".join(rows)


def main() -> int:
    """Refresh the generated release-notes block in place."""
    repository = os.getenv(
        "GITHUB_REPOSITORY", "Dependable-Intelligent-Systems-Lab/xwhy"
    )
    token = os.getenv("GITHUB_TOKEN")
    page = RELEASE_NOTES_PAGE.read_text(encoding="utf-8")

    try:
        release_section = _render_releases(
            _fetch_releases(repository, token), repository
        )
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
        print(
            f"Warning: could not refresh GitHub releases: {error}. "
            "Keeping the existing release-notes block.",
            file=sys.stderr,
        )
        release_section = (
            page.split(RELEASES_START, 1)[1].split(RELEASES_END, 1)[0].strip()
        )

    RELEASE_NOTES_PAGE.write_text(
        _replace_block(page, release_section), encoding="utf-8"
    )
    print(f"Updated {RELEASE_NOTES_PAGE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
