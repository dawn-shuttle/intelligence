#!/usr/bin/env python3
"""Project initialization script for dawn_shuttle template.

Usage::

    python init_project.py <sub_package_name> [--dirs DIR [DIR ...]] [-i]

Examples::

    python init_project.py web
    python init_project.py web --dirs api models services
    python init_project.py web -i
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_name(name: str) -> bool:
    """Return True if *name* is a valid Python identifier component."""
    return bool(_NAME_RE.match(name))


def _update_pyproject(sub_pkg_name: str, root: Path) -> None:
    """Rename the project in *pyproject.toml* to ``dawn_shuttle_<sub_pkg_name>``."""
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.exists():
        print("Warning: pyproject.toml not found - skipping rename.", file=sys.stderr)
        return

    new_name = f"dawn_shuttle_{sub_pkg_name}"
    content = pyproject_path.read_text(encoding="utf-8")

    updated = re.sub(
        r'^(name\s*=\s*")[^"]*(")',
        rf"\g<1>{new_name}\g<2>",
        content,
        count=1,
        flags=re.MULTILINE,
    )

    if updated == content:
        print(
            "Warning: could not locate the 'name' field in pyproject.toml.",
            file=sys.stderr,
        )
        return

    pyproject_path.write_text(updated, encoding="utf-8")
    print(f"✓ pyproject.toml renamed to '{new_name}'")


def _create_package(sub_pkg_name: str, extra_dirs: list[str], root: Path) -> None:
    """Create the package directory and optional sub-directories."""
    package_name = f"dawn_shuttle_{sub_pkg_name}"
    pkg_dir = root / "dawn_shuttle" / package_name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    init_file = pkg_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text(
            f'"""dawn_shuttle.{package_name} package."""\n', encoding="utf-8"
        )

    print(f"✓ Package directory created: dawn_shuttle/{package_name}/")

    for rel in extra_dirs:
        sub_dir = pkg_dir / rel
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub_init = sub_dir / "__init__.py"
        if not sub_init.exists():
            sub_init.touch()
        print(f"  ✓ Sub-directory created: dawn_shuttle/{package_name}/{rel}/")


def _collect_interactive_dirs() -> list[str]:
    """Prompt the user to enter directory names one by one."""
    dirs: list[str] = []
    print(
        "Enter sub-directory names to create inside the package (empty line to finish):"
    )
    while True:
        try:
            entry = input("  directory: ").strip()
        except EOFError:
            break
        if not entry:
            break
        dirs.append(entry)
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize a dawn_shuttle sub-project from this template. "
            "Renames pyproject.toml and creates the package directory."
        ),
    )
    parser.add_argument(
        "sub_package",
        help="Sub-package name, e.g. 'web' → project becomes 'dawn_shuttle_web'",
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=[],
        metavar="DIR",
        help="Sub-directories to create inside the new package",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactively enter sub-directory names",
    )

    args = parser.parse_args()

    if not _validate_name(args.sub_package):
        print(
            f"Error: '{args.sub_package}' is not a valid package name component. "
            "Use lowercase letters, digits, and underscores only, "
            "starting with a letter.",
            file=sys.stderr,
        )
        sys.exit(1)

    extra_dirs: list[str] = list(args.dirs or [])
    if args.interactive:
        extra_dirs.extend(_collect_interactive_dirs())

    root = Path(__file__).parent.resolve()

    _update_pyproject(args.sub_package, root)
    _create_package(args.sub_package, extra_dirs, root)

    print(f"\nProject 'dawn_shuttle_{args.sub_package}' initialized successfully.")


if __name__ == "__main__":
    main()
