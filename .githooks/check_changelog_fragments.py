#!/usr/bin/env python3
"""Reject staged changelog.d fragments that `assemble_changelog.py --check`
would reject: each must open with a '### ' heading or a '- ' bullet.

The rule lives in scripts/assemble_changelog.py and is not restated here.
This hook checks out the staged (index) version of every changelog.d file
the commit adds, copies, modifies or renames into a temporary directory and
runs the assembler's own `--check` on that directory, so the message is the
assembler's and the assembler decides what counts as a fragment (README.md
and subdirectories do not).

Only fragments the commit touches are checked: a malformed fragment already
on main does not block an unrelated commit. Agents push to main with
[skip ci], so without this hook nothing ran the check before the fragment
was on main.
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLER = ROOT / "scripts" / "assemble_changelog.py"
FRAGMENTS = "changelog.d"


def main():
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR",
         "--", FRAGMENTS],
        cwd=ROOT, capture_output=True, check=True).stdout
    paths = [p for p in staged.split(b"\0") if p]
    if not paths:
        return 0
    if not ASSEMBLER.is_file():
        sys.stderr.write(
            "ERROR: cannot check staged %s fragments: "
            "scripts/assemble_changelog.py is missing\n" % FRAGMENTS)
        return 1

    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("assemble_changelog", ASSEMBLER)
    assembler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assembler)

    with tempfile.TemporaryDirectory(prefix="changelog-fragments-") as temporary:
        subprocess.run(
            ["git", "checkout-index", "--prefix=%s/" % temporary, "-z", "--stdin"],
            cwd=ROOT, input=b"".join(p + b"\0" for p in paths), check=True)
        return assembler.main(
            ["--check", "--fragments", str(Path(temporary) / FRAGMENTS)])


if __name__ == "__main__":
    sys.exit(main())
