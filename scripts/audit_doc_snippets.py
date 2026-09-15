#!/usr/bin/env python3
r"""Audit Python code snippets in the docs for "runs as-is when copy-pasted".

Walks ``docs/`` for Markdown files, extracts every fenced
``\`\`\`python ... \`\`\``` block, and runs each one through
``pyflakes`` to flag undefined names (i.e. missing imports). Reports
which blocks fail and where, so we can fix them.

Skips blocks tagged with the comment ``# doc-audit: skip`` on the
first line — used for narrative fragments that intentionally aren't
standalone (e.g. mid-conversation REPL excerpts).

Usage::

    .venv/bin/python scripts/audit_doc_snippets.py
    .venv/bin/python scripts/audit_doc_snippets.py docs/quickstart.md  # one file
    .venv/bin/python scripts/audit_doc_snippets.py --strict             # also flag unused imports

Exit status is the number of failing blocks; 0 means everything
parses + imports cleanly.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
)
DOCS_DIR = REPO / "docs"

# Match ```python ... ``` fenced blocks. Captures the body.
FENCE_RE = re.compile(
    r"^```python\s*\n(.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)


@dataclass
class Snippet:
    file: Path
    line_start: int       # 1-indexed line of the opening fence
    body: str
    skipped: bool = False
    skip_reason: str = ""
    # Filled in after pyflakes runs:
    pyflakes_output: str = ""
    has_problem: bool = False


def find_snippets(md_path: Path) -> list[Snippet]:
    r"""Return all python code blocks in *md_path*.

    Block line numbers are 1-indexed and point at the opening
    ``\`\`\`python`` fence.
    """
    text = md_path.read_text()
    snippets: list[Snippet] = []
    for m in FENCE_RE.finditer(text):
        body = m.group(1)
        # Compute the line number of the opening fence.
        before = text[: m.start()]
        line_start = before.count("\n") + 1
        snippets.append(Snippet(file=md_path, line_start=line_start, body=body))
    return snippets


SKIP_DIRECTIVE_RE = re.compile(r"^\s*#\s*doc-audit:\s*skip\b", re.IGNORECASE)


def annotate_skips(snippets: list[Snippet]) -> None:
    """Mark snippets opted out via ``# doc-audit: skip`` first-line comment."""
    for s in snippets:
        first = s.body.lstrip().splitlines()[:1]
        if first and SKIP_DIRECTIVE_RE.match(first[0]):
            s.skipped = True
            s.skip_reason = "marked # doc-audit: skip"


def run_pyflakes(snippet: Snippet, strict: bool) -> None:
    """Run pyflakes on a snippet body. Records output + has_problem."""
    if snippet.skipped:
        return
    proc = subprocess.run(
        [sys.executable, "-m", "pyflakes", "/dev/stdin"],
        input=snippet.body,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout + proc.stderr).strip()
    if not out:
        return

    # pyflakes flags fall into a few categories:
    #   * "undefined name 'X'"           — missing import / typo  (BLOCKING)
    #   * "imported but unused"          — cosmetic               (only --strict)
    #   * "redefinition of unused 'X'"   — cosmetic               (only --strict)
    #   * "local variable 'X' assigned but never used"            (cosmetic)
    #   * "may be undefined, or defined from star imports"        (BLOCKING)
    blocking_substrings = (
        "undefined name",
        "may be undefined",
        "invalid syntax",
        "expected an indented block",
    )
    cosmetic_substrings = (
        "imported but unused",
        "redefinition of unused",
        "assigned but never used",
        "imported but unused",
        "local variable",
    )

    keep_lines: list[str] = []
    for line in out.splitlines():
        stripped = line.replace("/dev/stdin", "<snippet>")
        if any(b in stripped for b in blocking_substrings):
            keep_lines.append(stripped)
            snippet.has_problem = True
        elif strict and any(c in stripped for c in cosmetic_substrings):
            keep_lines.append(stripped)
            snippet.has_problem = True

    snippet.pyflakes_output = "\n".join(keep_lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Specific .md files to audit (default: all under docs/)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="also flag cosmetic issues (unused imports, unused locals)",
    )
    args = ap.parse_args()

    if args.paths:
        md_files = [p.resolve() for p in args.paths]
    else:
        md_files = sorted(
            p for p in DOCS_DIR.rglob("*.md")
            if "_build" not in p.parts and "_static" not in p.parts
        )

    all_snippets: list[Snippet] = []
    for md in md_files:
        snippets = find_snippets(md)
        annotate_skips(snippets)
        for s in snippets:
            run_pyflakes(s, args.strict)
        all_snippets.extend(snippets)

    # Report.
    failing = [s for s in all_snippets if s.has_problem]
    skipped = [s for s in all_snippets if s.skipped]

    print(
        f"Scanned {len(md_files)} file(s), {len(all_snippets)} python "
        f"block(s) ({len(skipped)} skipped via directive)."
    )

    if not failing:
        print("All blocks pass static analysis. ✓")
        return 0

    print(f"\n{len(failing)} block(s) with problems:\n")
    last_file: Path | None = None
    for s in failing:
        if s.file != last_file:
            try:
                rel = s.file.relative_to(REPO)
            except ValueError:
                rel = s.file
            print(f"  {rel}")
            last_file = s.file
        for line in s.pyflakes_output.splitlines():
            # pyflakes lines look like "<snippet>:N: COL: MESSAGE"
            print(f"    L{s.line_start} {line}")
        print()

    return len(failing)


if __name__ == "__main__":
    sys.exit(main())
