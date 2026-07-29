"""Generate the dynamic sections of the XWhy contributors page."""

from __future__ import annotations

import html
import json
import os
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRIBUTORS_PAGE = PROJECT_ROOT / "docs" / "contributors.md"
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"

CONTRIBUTORS_START = "<!-- AUTO-CONTRIBUTORS:START -->"
CONTRIBUTORS_END = "<!-- AUTO-CONTRIBUTORS:END -->"
AUTHORS_START = "<!-- AUTO-AUTHORS:START -->"
AUTHORS_END = "<!-- AUTO-AUTHORS:END -->"


def _replace_block(text: str, start: str, end: str, replacement: str) -> str:
    """Replace the content between two marker lines."""
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(f"Expected exactly one marker pair: {start} / {end}")

    prefix, remainder = text.split(start, 1)
    _, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n{replacement.rstrip()}\n{end}{suffix}"


def _request_json(url: str, token: str | None) -> tuple[Any, dict[str, str]]:
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
        payload = json.load(response)
        response_headers = {
            key.lower(): value for key, value in response.headers.items()
        }
    return payload, response_headers


def _fetch_contributors(
    repository: str, token: str | None
) -> list[dict[str, Any]]:
    """Fetch all non-bot contributors from the GitHub contributors API."""
    contributors: list[dict[str, Any]] = []
    page = 1

    while True:
        url = (
            f"https://api.github.com/repos/{repository}/contributors"
            f"?per_page=100&page={page}&anon=0"
        )
        payload, _ = _request_json(url, token)
        if not isinstance(payload, list):
            raise RuntimeError(
                "GitHub contributors API returned an unexpected response."
            )

        for item in payload:
            if not isinstance(item, dict):
                continue
            login = str(item.get("login") or "")
            account_type = str(item.get("type") or "")
            if (
                not login
                or account_type.casefold() == "bot"
                or login.casefold().endswith("[bot]")
            ):
                continue
            if not item.get("avatar_url") or not item.get("html_url"):
                continue
            contributors.append(item)

        if len(payload) < 100:
            break
        page += 1

    return contributors


def _render_contributors(contributors: list[dict[str, Any]]) -> str:
    """Render contributor cards for the documentation page."""
    if not contributors:
        return (
            "[View the live GitHub contributor graph]"
            "(https://github.com/Dependable-Intelligent-Systems-Lab/xwhy/graphs/contributors)."
        )

    cards: list[str] = []
    for contributor in contributors:
        login = html.escape(str(contributor["login"]))
        profile_url = html.escape(str(contributor["html_url"]), quote=True)
        avatar_url = str(contributor["avatar_url"])
        separator = "&" if "?" in avatar_url else "?"
        avatar_url = html.escape(f"{avatar_url}{separator}s=160", quote=True)
        commits = int(contributor.get("contributions") or 0)
        commit_label = "commit" if commits == 1 else "commits"

        cards.append(
            "\n".join(
                [
                    f'<a href="{profile_url}" target="_blank" '
                    'rel="noopener noreferrer" '
                    'style="display:flex;flex-direction:column;align-items:center;'
                    'width:7rem;padding:0.65rem;border:1px solid '
                    'var(--md-default-fg-color--lightest);border-radius:0.5rem;'
                    'text-decoration:none;text-align:center">',
                    f'  <img src="{avatar_url}" '
                    f'alt="GitHub avatar for {login}" loading="lazy" '
                    'style="width:4rem;height:4rem;border-radius:50%;'
                    'object-fit:cover">',
                    '  <span style="display:block;margin-top:0.45rem;'
                    f'font-weight:700;overflow-wrap:anywhere">@{login}</span>',
                    '  <span style="display:block;font-size:0.68rem;'
                    'color:var(--md-default-fg-color--light)">'
                    f"{commits} {commit_label}</span>",
                    "</a>",
                ]
            )
        )

    return "\n".join(
        [
            "## Repository contributors",
            "",
            (
                "This section is generated from GitHub during each documentation "
                "build. Bot accounts are excluded, and the commit counts reflect "
                "GitHub's contributor API."
            ),
            "",
            '<div style="display:flex;flex-wrap:wrap;gap:0.7rem;'
            'align-items:stretch">',
            *cards,
            "</div>",
            "",
            (
                "[View the full contributor graph on GitHub]"
                "(https://github.com/Dependable-Intelligent-Systems-Lab/"
                "xwhy/graphs/contributors)."
            ),
        ]
    )


def _load_people() -> list[tuple[str, str]]:
    """Read authors and maintainers from project metadata and merge their roles."""
    with PYPROJECT_FILE.open("rb") as file:
        project = tomllib.load(file)["project"]

    ordered_names: list[str] = []
    roles: dict[str, set[str]] = {}

    for role, field in (("Author", "authors"), ("Maintainer", "maintainers")):
        for person in project.get(field, []):
            name = str(person.get("name") or "").strip()
            if not name:
                continue
            if name not in roles:
                roles[name] = set()
                ordered_names.append(name)
            roles[name].add(role)

    people: list[tuple[str, str]] = []
    for name in ordered_names:
        person_roles = roles[name]
        if person_roles == {"Author", "Maintainer"}:
            role_text = "Author and maintainer"
        elif "Maintainer" in person_roles:
            role_text = "Maintainer"
        else:
            role_text = "Author"
        people.append((name, role_text))

    return people


def _render_people(people: list[tuple[str, str]]) -> str:
    """Render the package authors and maintainers table."""
    rows = [
        "## Authors and maintainers",
        "",
        "This table is generated from the current `[project]` metadata in "
        "`pyproject.toml`.",
        "",
        "| Contributor | Role recorded in the package metadata |",
        "| --- | --- |",
    ]
    rows.extend(f"| {name} | {role} |" for name, role in people)
    return "\n".join(rows)


def main() -> int:
    """Update dynamic contributor sections in place."""
    repository = os.getenv(
        "GITHUB_REPOSITORY", "Dependable-Intelligent-Systems-Lab/xwhy"
    )
    token = os.getenv("GITHUB_TOKEN")

    page = CONTRIBUTORS_PAGE.read_text(encoding="utf-8")

    try:
        contributors = _fetch_contributors(repository, token)
        contributor_section = _render_contributors(contributors)
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
        print(
            f"Warning: could not refresh GitHub contributors: {error}. "
            "Keeping the existing contributor block.",
            file=sys.stderr,
        )
        contributor_section = page.split(CONTRIBUTORS_START, 1)[1].split(
            CONTRIBUTORS_END, 1
        )[0].strip()

    page = _replace_block(
        page,
        CONTRIBUTORS_START,
        CONTRIBUTORS_END,
        contributor_section,
    )
    page = _replace_block(
        page,
        AUTHORS_START,
        AUTHORS_END,
        _render_people(_load_people()),
    )
    CONTRIBUTORS_PAGE.write_text(page, encoding="utf-8")

    print(f"Updated {CONTRIBUTORS_PAGE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
