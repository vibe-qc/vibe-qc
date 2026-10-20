"""Select the naming engine's source for hatchling (#536).

``vibeqc_naming`` lives at ``python/vibeqc_naming`` in the vibe-qc
repository, where vibe-qc's own wheel bundles it. This distribution
packages the same directory without moving or copying it in Git:

* from a checkout, the wheel takes the files from
  ``../python/vibeqc_naming`` and the sdist carries them as
  ``src/vibeqc_naming``;
* from an sdist, ``src/vibeqc_naming`` exists and the static wheel
  selection in ``pyproject.toml`` applies, so this hook does nothing;
* an editable install adds ``../python`` to ``sys.path`` through
  ``dev-mode-dirs`` and copies nothing.

The in-package test scripts (``_test_*.py``, ``_smoke_test.py``) and
byte-code caches never ship.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PACKAGE = "vibeqc_naming"


def is_test_script(name: str) -> bool:
    """True for the ad-hoc test scripts kept beside the package modules."""
    return name == "_smoke_test.py" or (name.startswith("_test_") and name.endswith(".py"))


def shipped_files(source: Path) -> list[Path]:
    """The package files that ship, relative to ``source``, sorted."""
    files: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if is_test_script(relative.name):
            continue
        files.append(relative)
    return files


class NamingSourceHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        root = Path(self.root)
        if (root / "src" / PACKAGE).is_dir():
            return  # sdist layout: pyproject.toml selects src/vibeqc_naming
        if version == "editable":
            return  # dev-mode-dirs puts ../python on sys.path
        source = (root.parent / "python" / PACKAGE).resolve()
        if not source.is_dir():
            raise FileNotFoundError(
                f"{PACKAGE} source not found at {source}: build from a vibe-qc "
                "checkout or from an sdist"
            )
        target = Path("src", PACKAGE) if self.target_name == "sdist" else Path(PACKAGE)
        force_include = build_data.setdefault("force_include", {})
        for relative in shipped_files(source):
            force_include[str(source / relative)] = (target / relative).as_posix()
