"""Generate the public API index, object pages, and internal module reference."""

from __future__ import annotations

import sys
from pathlib import Path

import mkdocs_gen_files

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.api_reference import discover_public_api, render_api_index  # noqa: E402


def _module_name(path: Path) -> str:
    """Return a Python module name for a source path."""
    relative = path.relative_to(SRC_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


objects = discover_public_api()

with mkdocs_gen_files.open("reference/index.md", "w") as index_file:
    index_file.write(render_api_index(objects))
mkdocs_gen_files.set_edit_path("reference/index.md", "scripts/api_reference.py")

for item in objects:
    object_path = Path("reference", "generated", *item.public_path.split("."))
    object_path = object_path.with_suffix(".md")
    with mkdocs_gen_files.open(object_path, "w") as object_file:
        object_file.write(
            "\n".join(
                [
                    f"# `{item.public_path}`",
                    "",
                    f"::: {item.public_path}",
                    "    options:",
                    "      show_root_heading: false",
                    "      show_root_full_path: true",
                    "      members_order: source",
                    "",
                ]
            )
        )
    mkdocs_gen_files.set_edit_path(
        object_path, item.source_path.relative_to(PROJECT_ROOT)
    )

for source_path in sorted((SRC_ROOT / "xwhy").rglob("*.py")):
    if source_path.name == "__main__.py":
        continue
    module = _module_name(source_path)
    # Keep the historical module URL layout because guides deep-link to it.
    module_path = Path("reference", *module.split("."))
    if source_path.name == "__init__.py":
        module_path = module_path / "index.md"
    else:
        module_path = module_path.with_suffix(".md")

    with mkdocs_gen_files.open(module_path, "w") as module_file:
        module_file.write(
            "\n".join(
                [
                    f"# `{module}`",
                    "",
                    f"::: {module}",
                    "    options:",
                    "      show_root_heading: false",
                    "      show_root_full_path: true",
                    "      members_order: source",
                    "",
                ]
            )
        )
    mkdocs_gen_files.set_edit_path(module_path, source_path.relative_to(PROJECT_ROOT))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.write("* [API Reference](index.md)\n")
