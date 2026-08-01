"""Discover and validate the public XWhy API without importing the package."""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "xwhy"

CATEGORY_TITLES = {
    "config": "Configuration",
    "core": "Core abstractions and results",
    "distance": "Distances",
    "explainers": "Explainers",
    "metrics": "Metrics",
    "models": "Models",
    "perturbation": "Perturbation",
    "plots": "Plots",
    "providers": "Providers",
    "surrogate": "Surrogate models",
    "utils": "Utilities",
    "xwhy": "General",
}

CATEGORY_ORDER = (
    "explainers",
    "core",
    "plots",
    "distance",
    "models",
    "providers",
    "perturbation",
    "surrogate",
    "metrics",
    "config",
    "utils",
    "xwhy",
)


@dataclass(frozen=True)
class PublicObject:
    """A public object exposed by an XWhy package."""

    public_path: str
    origin: str
    category: str
    summary: str
    source_path: Path


def _module_name(path: Path) -> str:
    """Return a Python module name for a source file."""
    relative = path.relative_to(SRC_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _parse(path: Path) -> ast.Module:
    """Parse a source file and include its path in syntax errors."""
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        raise RuntimeError(
            f"Could not parse {path.relative_to(PROJECT_ROOT)}"
        ) from error


def _literal_all(tree: ast.Module, path: Path) -> list[str]:
    """Read a literal ``__all__`` declaration from a package initializer."""
    for node in tree.body:
        value: ast.expr | None = None
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            value = node.value

        if value is None:
            continue

        try:
            exports = ast.literal_eval(value)
        except (ValueError, TypeError) as error:
            raise RuntimeError(
                f"{path.relative_to(PROJECT_ROOT)} must declare __all__ as a "
                "literal list or tuple."
            ) from error

        if not isinstance(exports, (list, tuple)) or not all(
            isinstance(name, str) for name in exports
        ):
            raise RuntimeError(
                f"{path.relative_to(PROJECT_ROOT)} contains an invalid __all__."
            )
        if len(exports) != len(set(exports)):
            raise RuntimeError(
                f"{path.relative_to(PROJECT_ROOT)} contains duplicate __all__ entries."
            )
        return list(exports)

    return []


def _resolve_import_module(package: str, node: ast.ImportFrom) -> str:
    """Resolve an absolute module name for an ImportFrom node."""
    if node.level == 0:
        return node.module or ""

    package_parts = package.split(".")
    keep = len(package_parts) - node.level + 1
    if keep < 0:
        raise RuntimeError(f"Invalid relative import in package {package}")
    prefix = package_parts[:keep]
    if node.module:
        prefix.extend(node.module.split("."))
    return ".".join(prefix)


def _bindings(tree: ast.Module, package: str) -> dict[str, str]:
    """Map names bound by a package initializer to their source objects."""
    bindings: dict[str, str] = {}

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = _resolve_import_module(package, node)
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound_name = alias.asname or alias.name
                bindings[bound_name] = f"{module}.{alias.name}".strip(".")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".")[0]
                bindings[bound_name] = alias.name
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            bindings[node.name] = f"{package}.{node.name}"
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = f"{package}.{target.id}"

    return bindings


def _source_for_origin(origin: str) -> tuple[Path, ast.AST | None]:
    """Find the source file and definition node for an imported object."""
    parts = origin.split(".")

    for module_length in range(len(parts) - 1, 0, -1):
        module_parts = parts[:module_length]
        symbol_parts = parts[module_length:]
        module_file = SRC_ROOT.joinpath(*module_parts).with_suffix(".py")
        package_file = SRC_ROOT.joinpath(*module_parts, "__init__.py")
        path = module_file if module_file.is_file() else package_file
        if not path.is_file():
            continue

        tree = _parse(path)
        symbol_name = symbol_parts[0] if symbol_parts else ""
        for node in tree.body:
            if (
                isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == symbol_name
            ):
                return path, node
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if any(
                    isinstance(target, ast.Name) and target.id == symbol_name
                    for target in targets
                ):
                    return path, node
        return path, None

    raise RuntimeError(f"Could not locate source for public API object {origin}")


def _summary(node: ast.AST | None, public_path: str) -> str:
    """Return a one-sentence summary suitable for the API index table."""
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        docstring = ast.get_docstring(node, clean=True)
        if docstring:
            paragraph = docstring.split("\n\n", 1)[0]
            compact = " ".join(paragraph.split())
            sentence = re.split(r"(?<=[.!?])\s+", compact, maxsplit=1)[0]
            return sentence.replace("|", "\\|")

    name = public_path.rsplit(".", 1)[-1]
    return f"Public XWhy API object `{name}`."


def _category(origin: str) -> str:
    """Group an object by its top-level XWhy source package."""
    parts = origin.split(".")
    return parts[1] if len(parts) > 2 else "xwhy"


def _is_exported_subpackage(origin: str) -> bool:
    """Return whether an export is a package represented by its own objects."""
    path = SRC_ROOT.joinpath(*origin.split("."), "__init__.py")
    return path.is_file() and bool(_literal_all(_parse(path), path))


def discover_public_api() -> list[PublicObject]:
    """Discover the public API from root and first-level package ``__all__`` lists."""
    root_init = PACKAGE_ROOT / "__init__.py"
    package_inits = [root_init, *sorted(PACKAGE_ROOT.glob("*/__init__.py"))]
    objects: list[PublicObject] = []
    seen_origins: set[str] = set()

    for init_path in package_inits:
        package = _module_name(init_path)
        tree = _parse(init_path)
        exports = _literal_all(tree, init_path)
        bindings = _bindings(tree, package)

        for name in exports:
            origin = bindings.get(name)
            if origin is None:
                raise RuntimeError(
                    f"{init_path.relative_to(PROJECT_ROOT)} exports {name!r}, "
                    "but that name is not defined or imported."
                )
            if origin in seen_origins or _is_exported_subpackage(origin):
                continue

            source_path, node = _source_for_origin(origin)
            public_path = f"{package}.{name}"
            objects.append(
                PublicObject(
                    public_path=public_path,
                    origin=origin,
                    category=_category(origin),
                    summary=_summary(node, public_path),
                    source_path=source_path,
                )
            )
            seen_origins.add(origin)

    order = {name: index for index, name in enumerate(CATEGORY_ORDER)}
    return sorted(
        objects,
        key=lambda item: (
            order.get(item.category, len(order)),
            item.category,
            item.public_path.casefold(),
        ),
    )


def render_api_index(objects: list[PublicObject]) -> str:
    """Render a SHAP-style summary page grouped by API area."""
    lines = [
        "# API Reference",
        "",
        (
            "This page is generated from the public objects declared in "
            "`src/xwhy/**/__init__.py`. Adding or removing an object from a "
            "package's `__all__` updates this index during the next documentation "
            "build."
        ),
        "",
        (
            "Select an object for its full signature, parameters, return values, "
            "and documented members."
        ),
    ]

    categories: dict[str, list[PublicObject]] = {}
    for item in objects:
        categories.setdefault(item.category, []).append(item)

    order = {name: index for index, name in enumerate(CATEGORY_ORDER)}
    for category in sorted(
        categories,
        key=lambda name: (order.get(name, len(order)), name),
    ):
        title = CATEGORY_TITLES.get(category, category.replace("_", " ").title())
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Object | Description |",
                "| --- | --- |",
            ]
        )
        for item in categories[category]:
            target = "generated/" + "/".join(item.public_path.split(".")) + ".md"
            lines.append(f"| [`{item.public_path}`]({target}) | {item.summary} |")

    lines.extend(
        [
            "",
            "---",
            "",
            (
                "Module-level documentation is also generated under "
                "`reference/xwhy/` for maintainers and existing deep links, but "
                "it is intentionally "
                "excluded from the public API index."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate public exports and print the discovered API inventory.",
    )
    return parser.parse_args()


def main() -> int:
    """Validate and report the public API inventory."""
    args = _parse_args()
    if not args.check:
        raise SystemExit("Use --check to validate the public API reference contract.")

    objects = discover_public_api()
    counts: dict[str, int] = {}
    for item in objects:
        counts[item.category] = counts.get(item.category, 0) + 1

    print(f"Validated {len(objects)} public API objects from {PACKAGE_ROOT}")
    for category, count in sorted(counts.items()):
        title = CATEGORY_TITLES.get(category, category.title())
        print(f"- {title}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
