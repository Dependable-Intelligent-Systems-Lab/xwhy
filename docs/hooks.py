"""Build-time variables for the XWhy documentation."""

from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"


def _read_project_version() -> str:
    """Read the authoritative package version from pyproject.toml."""
    with PYPROJECT_PATH.open("rb") as file:
        project_data = tomllib.load(file)

    try:
        return str(project_data["project"]["version"])
    except KeyError as error:
        raise RuntimeError(
            "The project version is missing from pyproject.toml."
        ) from error


PROJECT_VERSION = _read_project_version()


def on_page_markdown(markdown: str, **kwargs: object) -> str:
    """Replace documentation version placeholders during the build."""
    return markdown.replace("{{ XWHY_VERSION }}", PROJECT_VERSION)
