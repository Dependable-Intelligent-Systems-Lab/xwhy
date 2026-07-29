"""Inject the package version into documentation source files before building."""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"
INDEX_FILE = PROJECT_ROOT / "docs" / "index.md"
PLACEHOLDER = "{{ XWHY_VERSION }}"


def main() -> int:
    """Replace the documentation version placeholder from project metadata."""
    with PYPROJECT_FILE.open("rb") as file:
        version = str(tomllib.load(file)["project"]["version"])

    index_text = INDEX_FILE.read_text(encoding="utf-8")
    if PLACEHOLDER not in index_text:
        raise RuntimeError(
            f"Expected version placeholder {PLACEHOLDER!r} in {INDEX_FILE}."
        )

    INDEX_FILE.write_text(
        index_text.replace(PLACEHOLDER, version),
        encoding="utf-8",
    )
    print(f"Injected XWhy v{version} into {INDEX_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
