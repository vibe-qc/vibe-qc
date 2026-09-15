"""Guard: every source and vibeqc header a CMakeLists names must exist.

Ordinary ``main`` pushes run no CI (CLAUDE.md § 2), so a build-graph
reference to a file that was never ``git add``ed is invisible until some
chat happens to do a fresh CMake configure. On 2026-08-06 that gap cost
~30 minutes of unbuildable ``main``: ``58d980d656`` added
``src/bipole_dispatch.cpp`` to the ``vibeqc_core`` source list and an
``#include "vibeqc/bipole_dispatch.hpp"`` to ``bindings.cpp`` while both
files stayed untracked in the authoring clone, and nine further commits
landed on top before ``8a07eda80`` supplied them.

That break had two independent halves, which is why this guard checks
both: deleting the CMakeLists line alone would not have fixed it, since
``bindings.cpp`` is itself in the source list and would still have failed
on the missing header.

The check is filesystem-only — no configure, no compiler, milliseconds.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]

# Target-declaring commands that take a source list.
_TARGET_CMD = re.compile(
    r"\b(?:add_library|add_executable|pybind11_add_module)\s*\(",
)

# Tokens that are target properties rather than sources.
_KEYWORDS = frozenset(
    {
        "STATIC",
        "SHARED",
        "MODULE",
        "OBJECT",
        "INTERFACE",
        "IMPORTED",
        "ALIAS",
        "GLOBAL",
        "EXCLUDE_FROM_ALL",
        "WIN32",
        "MACOSX_BUNDLE",
    }
)

_SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx", ".c")

# Headers that no checkout contains because CMake writes them into the
# build tree. ``build_config.hpp`` is configured into
# ``${CMAKE_CURRENT_BINARY_DIR}/generated/`` and is correct by design.
_GENERATED_HEADERS = frozenset({"vibeqc/build_config.hpp"})

# Known-dangling references, kept visible rather than silently skipped.
# Each entry needs a reason and, where relevant, why it is inert.
_ALLOWED_MISSING_SOURCES = {
    # Points at ``cpp/tests/``; the real file is at repo-root ``tests/``.
    # Doubly inert: gated behind VIBEQC_USE_SEQUANT AND BUILD_TESTING
    # (both default OFF), and ``cpp/src/sequant_codegen/`` is never
    # pulled in — no add_subdirectory references it anywhere in the tree.
    "cpp/src/tests/test_sequant_codegen_ccsd.cpp",
}


def _strip_comments(text: str) -> str:
    """Drop CMake ``#`` comments.

    Not cosmetic: ``cpp/CMakeLists.txt`` discusses libecpint's own
    ``src/lib/api.cpp`` in a comment, which a naive scan reads as a
    source of ours.
    """
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _declared_sources(cmakelists: Path) -> list[tuple[str, Path]]:
    """Return ``(raw_token, resolved_path)`` for each source named."""
    text = _strip_comments(cmakelists.read_text(encoding="utf-8"))
    here = cmakelists.parent
    found: list[tuple[str, Path]] = []

    for match in _TARGET_CMD.finditer(text):
        open_idx = match.end() - 1
        close_idx = _matching_paren(text, open_idx)
        if close_idx < 0:
            continue
        body = text[open_idx + 1 : close_idx]

        tokens = body.split()
        # First token is the target name, never a source.
        for token in tokens[1:]:
            if token in _KEYWORDS:
                continue
            # Generator expressions and unresolvable variables.
            if "$<" in token:
                continue
            resolved = token.replace("${CMAKE_CURRENT_SOURCE_DIR}/", "")
            if "${" in resolved:
                continue
            if not resolved.endswith(_SOURCE_SUFFIXES):
                continue
            found.append((token, (here / resolved).resolve()))

    return found


def _cmakelists_files() -> list[Path]:
    # The guard polices the TRACKED build graph (see module docstring), so
    # discovery must be scoped to git-tracked files: a shared clone can carry
    # other agents' registered worktrees (.worktrees/, .codex-worktrees/,
    # .claude/worktrees/) whose older snapshots a filesystem glob would scan
    # as if they were this tree's build graph, failing on files that only
    # exist in the current tree.
    import subprocess

    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", "CMakeLists.txt", "*/CMakeLists.txt"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        paths = [_REPO_ROOT / p for p in out.split("\0") if p]
        if paths:
            return sorted(paths)
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(_REPO_ROOT.glob("**/CMakeLists.txt"))


def test_every_cmake_source_exists() -> None:
    """A source named in a target's list must exist on disk."""
    missing: list[str] = []

    for cmakelists in _cmakelists_files():
        if "third_party" in cmakelists.parts:
            continue
        for token, path in _declared_sources(cmakelists):
            if path.exists():
                continue
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel in _ALLOWED_MISSING_SOURCES:
                continue
            where = cmakelists.relative_to(_REPO_ROOT).as_posix()
            missing.append(f"{where} names {token!r} -> {rel} (absent)")

    assert not missing, (
        "CMakeLists references source files that do not exist. CMake will "
        "fail at the generate step for every fresh configure:\n  "
        + "\n  ".join(missing)
        + "\n\nDid a commit add the build-graph entry but not `git add` the "
        "file itself?"
    )


def test_every_vibeqc_include_resolves() -> None:
    """``#include "vibeqc/foo.hpp"`` must resolve under ``cpp/include``."""
    include_root = _REPO_ROOT / "cpp" / "include"
    pattern = re.compile(r'#\s*include\s*"(vibeqc/[^"]+)"')
    missing: list[str] = []

    sources = [
        p
        for suffix in (*_SOURCE_SUFFIXES, ".hpp", ".h")
        for p in (_REPO_ROOT / "cpp").glob(f"**/*{suffix}")
        if "third_party" not in p.parts
    ]

    for source in sorted(sources):
        for header in pattern.findall(
            source.read_text(encoding="utf-8", errors="ignore")
        ):
            if header in _GENERATED_HEADERS:
                continue
            if (include_root / header).exists():
                continue
            where = source.relative_to(_REPO_ROOT).as_posix()
            missing.append(f"{where} includes {header!r} (absent)")

    assert not missing, (
        "C++ sources include vibeqc headers that do not exist:\n  "
        + "\n  ".join(missing)
        + "\n\nDid a commit add the #include but not `git add` the header?"
    )


@pytest.mark.parametrize(
    "cmakelists",
    [p.relative_to(_REPO_ROOT).as_posix() for p in _cmakelists_files()],
)
def test_cmakelists_is_parseable(cmakelists: str) -> None:
    """The parser must actually reach each file's target declarations.

    Without this, a parser that silently matched nothing would make the
    guards above pass vacuously.
    """
    path = _REPO_ROOT / cmakelists
    text = _strip_comments(path.read_text(encoding="utf-8"))
    for match in _TARGET_CMD.finditer(text):
        assert _matching_paren(text, match.end() - 1) > 0, (
            f"{cmakelists}: unbalanced parentheses in a target declaration; "
            "the source-existence guard cannot read this file."
        )
