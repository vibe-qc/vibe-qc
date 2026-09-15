"""No maintainer home paths in committed files.

CONTRIBUTING.md, "Personal information", forbids author home paths (``/Users/<name>/…``,
``/home/<name>/…``) and the day-job employer string in any committed
file. Placeholders (``/home/USER/``, ``<vibe-qc-checkout>``, ``~/``)
are the correct alternative.

The pre-commit hook at `.githooks/pre-commit` is the first line of
defence, but pre-commit is opt-in per clone — a contributor who
skipped ``git config --local core.hooksPath .githooks`` can still
push a leaking file. This test re-runs the hook's pattern set in
CI so the same policy gates the merge gate regardless of per-clone
hook activation.

Three scopes:

* ``test_no_maintainer_paths_in_generated_basis_dir`` — the
  runtime-shipped basis directory. Catches leaks in ``.g94``
  provenance headers, which are easy to miss in a large basis
  diff.
* ``test_no_maintainer_paths_in_custom_basis_sources`` — the
  ``custom/`` source the build pipeline promotes into ``basis/``;
  a leak here would be copied into the generated dir on the next
  ``setup_basis_library.sh`` run.
* ``test_no_maintainer_paths_in_repo_tree`` — the rest of the
  repository (every committed file under the source extensions the
  hook inspects). Catches doc, example, and test leaks that the
  basis-only scopes miss. Excludes build artefacts and the
  policy-meta files the hook itself excludes (``CLAUDE.md``,
  ``AGENTS.md``, ``.mailmap``, ``.githooks/``, this file).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Inspect this checkout, without importing a possibly unrelated installed core.
BUNDLED_BASIS_DIR = _REPO_ROOT / "python" / "vibeqc" / "basis_library"

# Same broadened pattern set as ``.githooks/pre-commit``. The hook
# enforces the policy on staged additions per clone; this test
# enforces it on the committed basis library in CI. Update both
# together if the policy changes.
_HOME_PATH_RE = re.compile(
    # /Users/<name>/ or /home/<name>/, with <name> starting with a
    # letter; exact placeholders and system roles are checked separately.
    r"/(Users|home)/([A-Za-z][A-Za-z0-9_.-]*)/?"
)
# Private names are configured externally; never publish the denylist itself.
import runpy
_PRIVATE_TERMS = runpy.run_path(str(_REPO_ROOT / ".githooks/private_terms.py"))["load_terms"](_REPO_ROOT)


def _private_match(line: str) -> bool:
    return any(term in line.casefold() for term in _PRIVATE_TERMS)


# Placeholders and system usernames that are legitimate references in committed
# content. Keep in sync with ``.githooks/pre-commit``'s ALLOWED_USERS.
# Add sparingly; each entry weakens the gate.
# - runner, root: CI / system accounts in GitHub-Actions-style docs
# - user: documented placeholder convention (see vibe-queue/tests/
#   test_config.py and docs/user_guide/queue.md — the "/home/user/..."
#   form predates the all-caps "/home/USER/..." form and remains in
#   wide use)
_ALLOWED_USERS = frozenset({"USER", "Shared", "runner", "root", "user"})


def _scan(root: Path) -> list[tuple[Path, int, str, str]]:
    """Return (path, lineno, leak_token, full_line) tuples."""
    hits: list[tuple[Path, int, str, str]] = []
    if not root.is_dir():
        return hits
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            for m in _HOME_PATH_RE.finditer(line):
                if m.group(2) in _ALLOWED_USERS:
                    continue
                hits.append((path, lineno, m.group(0), line.strip()))
            if _private_match(line):
                hits.append((path, lineno, "<private-term>", line.strip()))
    return hits


def test_no_maintainer_paths_in_generated_basis_dir():
    """The committed, runtime-shipped basis directory must be free of
    author home paths and the day-job employer string."""
    hits = _scan(BUNDLED_BASIS_DIR / "basis")
    assert not hits, _format_hits(hits, "basis/")


def test_no_maintainer_paths_in_custom_basis_sources():
    """``custom/`` is the source the build pipeline promotes into
    ``basis/``; a leak here will be copied into the generated dir on
    the next re-run of ``setup_basis_library.sh``."""
    custom = BUNDLED_BASIS_DIR / "custom"
    if not custom.is_dir():
        pytest.skip(f"{custom} not present in this checkout")
    hits = _scan(custom)
    assert not hits, _format_hits(hits, "custom/")


# Repo root = two levels up from this file (tests/<file> → repo/).
_REPO_ROOT = Path(__file__).resolve().parents[1]

# File extensions to scan. Mirrors the file types the pre-commit hook
# inspects via ``git diff --cached`` (no extension filter there — it
# sees every staged file). Constrained here to source-tree text files
# so the test stays fast on a large checkout. ``.g94`` and ``.ecp``
# bundled basis data are covered by the dedicated basis-dir scopes
# above.
_REPO_SCAN_SUFFIXES = frozenset({
    ".md", ".py", ".sh", ".toml", ".json", ".yaml", ".yml",
    ".cff", ".cmake", ".cpp", ".h", ".hpp", ".rst", ".txt",
})

# Directories to skip wholesale. Build artefacts, virtualenvs, vendored
# third-party sources, and generated output. These are also the same
# excludes used by the audit grep documented in CONTRIBUTING.md, Personal information.
_REPO_SCAN_SKIP_DIRS = frozenset({
    ".git", ".venv", ".venv-rebase", "venv", ".pytest_cache", ".mypy_cache",
    "__pycache__", "node_modules",
    # `build_dev` is the build-dir name `HANDOVER_PERIODIC_BASIS_GRADIENT.md`
    # tells developers to use, and `.gitignore` covers it -- but CMake stamps
    # absolute paths into CMakeCache.txt, so leaving it out of this list made
    # the scan fail for anyone who followed that handover.
    "build", "build-local", "build_dev", "_build", "_skbuild", "_static",
    "dist",
    "third_party",
    ".release-status",  # gitignored local-only chat coordination files
    ".claude",  # gitignored local-only agent worktrees (full repo copies)
})

# Files the hook itself excludes (.githooks/pre-commit pathspec list).
# Optional agent-instruction filenames are reserved, .mailmap re-maps
# historical author emails, and this file mirrors the hook patterns for CI.
# The public policy is in CONTRIBUTING.md; no agent file is required.
_REPO_SCAN_SKIP_FILES = frozenset({
    "CLAUDE.md", "AGENTS.md", ".mailmap",
    str(Path("tests") / "test_basis_no_maintainer_paths.py"),
})

# Entire directory trees the hook excludes. ``.githooks/`` holds the
# hook itself, which documents the patterns it blocks.
_REPO_SCAN_SKIP_REL_DIRS = frozenset({".githooks"})


def _scan_repo_tree(root: Path) -> list[tuple[Path, int, str, str]]:
    """Walk ``root`` and return hits in source-tree text files,
    excluding the same surfaces the pre-commit hook excludes."""
    hits: list[tuple[Path, int, str, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Directory-based skips (any ancestor name in the skip set). The
        # ``.venv`` prefix match also excludes audit/CI virtualenvs created
        # in-repo under a suffixed name (e.g. ``.venv-audit-<sha>``), matching
        # this skip list's documented "virtualenvs" intent. Virtualenvs are
        # never committed, so this cannot hide a leak in tracked content.
        rel_parts = path.relative_to(root).parts
        if any(
            part in _REPO_SCAN_SKIP_DIRS or part.startswith(".venv")
            for part in rel_parts[:-1]
        ):
            continue
        if rel_parts and rel_parts[0] in _REPO_SCAN_SKIP_REL_DIRS:
            continue
        rel_str = str(path.relative_to(root))
        if rel_str in _REPO_SCAN_SKIP_FILES:
            continue
        if path.name in _REPO_SCAN_SKIP_FILES:
            continue
        if path.suffix not in _REPO_SCAN_SUFFIXES:
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for m in _HOME_PATH_RE.finditer(line):
                if m.group(2) in _ALLOWED_USERS:
                    continue
                hits.append((path, lineno, m.group(0), line.strip()))
            if _private_match(line):
                hits.append((path, lineno, "<private-term>", line.strip()))
    return hits


def test_no_maintainer_paths_in_repo_tree():
    """The committed source tree (docs, examples, scripts, tests, all
    file types the hook inspects) must be free of author home paths
    and the day-job employer string. Mirrors CONTRIBUTING.md / the
    .githooks/pre-commit denylist across the whole repo, not just the
    basis library."""
    hits = _scan_repo_tree(_REPO_ROOT)
    assert not hits, _format_hits(hits, "<repo>/")


def test_repo_scan_ignores_local_build_trees(tmp_path: Path):
    for directory in ("build-local", "_skbuild"):
        metadata = tmp_path / directory / "cmake" / "metadata.json"
        metadata.parent.mkdir(parents=True)
        metadata.write_text('"path": "/Users/private-maintainer/build"')

    assert _scan_repo_tree(tmp_path) == []


def _format_hits(
    hits: list[tuple[Path, int, str, str]], label: str
) -> str:
    lines = [
        f"Personal-info leak in basis_library/{label} "
        f"(CONTRIBUTING.md, Personal information; same patterns as .githooks/pre-commit):",
    ]
    for path, lineno, _token, _content in hits:
        try:
            relative = path.relative_to(_REPO_ROOT)
        except ValueError:
            relative = Path("<external-fixture>") / path.name
        lines.append(f"  {relative}:{lineno}: personal-info match [redacted]")
    lines.append(
        "\nUse a placeholder instead: '~/', '/home/USER/', "
        "'<vibe-qc-checkout>', or similar."
    )
    return "\n".join(lines)


def test_privacy_hook_regressions():
    """Run isolated staged-diff regressions without any runtime dependencies."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / ".githooks" / "test_privacy_hook.py"), "-q"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
