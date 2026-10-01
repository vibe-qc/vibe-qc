#!/usr/bin/env python3
"""Fold `changelog.d/` fragments into CHANGELOG.md's [Unreleased] section.

Why fragments exist
-------------------
Every branch that appends to `[Unreleased]` inserts at the same anchor, so two
branches conflict as an add/add that no content edit resolves -- only moving the
merge base does. With one merge queue and a serialised `native-build-test`
resource group, draining N ready merge requests costs N(N+1)/2 pipeline runs
instead of N, because each merge re-conflicts everything behind it.

A fragment is one file per change, so two branches never touch the same path.

What this does NOT touch
------------------------
Released `## [vX.Y.Z]` sections and `scripts/test_gate/changelog_pins.toml`.
`changelog_guard.py` pins released bodies only -- `[Unreleased]` is never
pinned -- so assembling here cannot invalidate a pin. Run this when cutting a
release, before the promotion commit that turns `[Unreleased]` into a version.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

UNRELEASED = "## [Unreleased]"


def fragments(d: Path) -> list[tuple[str, str]]:
    """Every fragment as (name, text), sorted by filename for a stable order."""
    out = []
    for p in sorted(d.glob("*.md")):
        if p.name == "README.md":
            continue
        text = p.read_text().strip("\n")
        if text:
            out.append((p.name, text))
    return out


def unreleased_bounds(lines: list[str]) -> tuple[int, int]:
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith(UNRELEASED))
    except StopIteration:
        raise SystemExit("CHANGELOG.md has no %s section" % UNRELEASED)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## [")), len(lines))
    return start, end


def assemble(changelog: Path, d: Path) -> str:
    frags = fragments(d)
    lines = changelog.read_text().splitlines()
    start, end = unreleased_bounds(lines)
    body = lines[start + 1 : end]
    while body and not body[-1].strip():
        body.pop()
    for _, text in frags:
        body += [""] + text.splitlines()
    return "\n".join(lines[: start + 1] + body + [""] + lines[end:]) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    ap.add_argument("--fragments", type=Path, default=Path("changelog.d"))
    ap.add_argument("--check", action="store_true",
                    help="validate fragments and report, writing nothing")
    ap.add_argument("--clean", action="store_true",
                    help="delete the fragments after a successful write")
    a = ap.parse_args(argv)

    if not a.fragments.is_dir():
        print("no %s directory; nothing to assemble" % a.fragments)
        return 0
    frags = fragments(a.fragments)
    if a.check:
        for name, text in frags:
            first = text.splitlines()[0]
            if not (first.startswith("### ") or first.startswith("- ")):
                print("%s: must start with a '### ' heading or a '- ' bullet, got %r"
                      % (name, first[:40]), file=sys.stderr)
                return 1
        print("%d fragment(s) OK" % len(frags))
        return 0
    if not frags:
        print("no fragments to assemble")
        return 0
    a.changelog.write_text(assemble(a.changelog, a.fragments))
    print("folded %d fragment(s) into %s" % (len(frags), a.changelog))
    if a.clean:
        for p in sorted(a.fragments.glob("*.md")):
            if p.name != "README.md":
                p.unlink()
        print("removed %d fragment file(s)" % len(frags))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
