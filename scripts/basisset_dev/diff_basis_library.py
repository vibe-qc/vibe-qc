"""Summarise changes to ``python/vibeqc/basis_library/basis/``.

The bundled basis library is mechanically generated (see
``docs/basisset_dev/GENERATING_THE_BASIS_LIBRARY.md``). A regeneration
of even one custom file can produce hundreds of `.g94` / `.ecp` diffs
that drown out the actually-meaningful change. This helper enumerates
the diff in a form a reviewer can scan in a few dozen lines:

* added / removed / modified ``.g94`` files
* added / removed / modified ``.ecp`` sidecars, with pair-mismatch flags
  (``.g94`` removed but ``.ecp`` left behind, etc.)
* provenance-header diffs (``! Originating publication:``, ``! Fetched
  from:``, ``! Citation:`` lines) for every modified ``.g94`` so a
  silent attribution swap is impossible to miss

Read-only; uses ``git diff --name-status`` and ``git show`` only —
no rebuild of native dependencies, no Python deps outside the stdlib.

Usage
-----

::

    # working tree vs HEAD
    python scripts/basisset_dev/diff_basis_library.py

    # working tree vs a ref
    python scripts/basisset_dev/diff_basis_library.py origin/main

    # ref vs ref
    python scripts/basisset_dev/diff_basis_library.py origin/main HEAD
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


BASIS_PREFIX = "python/vibeqc/basis_library/basis/"

# Provenance-style header lines we always echo in the per-file diff.
# Matches the conventions used by scripts/basisset_dev/fetch_from_bse.py
# plus hand-curated custom .g94 headers.
_PROV_RE = re.compile(
    r"^\s*!\s*("
    r"originating publication|citation|reference|fetched from|"
    r"cite|published in|published as|doi|bundled with|basis set"
    r")\b",
    re.IGNORECASE,
)


def _run(args: list[str]) -> str:
    """Run a git command, return stdout. Raises on non-zero exit."""
    return subprocess.run(
        args, check=True, capture_output=True, text=True
    ).stdout


def _name_status(ref_a: str, ref_b: str | None) -> list[tuple[str, str]]:
    """Return list of (status, path) for diff entries under BASIS_PREFIX.

    ``ref_b`` of None compares against the working tree.
    """
    args = ["git", "diff", "--name-status"]
    args.append(ref_a)
    if ref_b is not None:
        args.append(ref_b)
    args += ["--", BASIS_PREFIX]
    out = _run(args)
    rows: list[tuple[str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][0]  # A / M / D / R… (rename gets two paths)
        path = parts[-1]
        rows.append((status, path))
    return rows


def _show(ref: str | None, path: str) -> str | None:
    """Return file contents at ``ref`` (None = working tree)."""
    if ref is None:
        p = Path(path)
        if not p.exists():
            return None
        return p.read_text(errors="replace")
    try:
        return _run(["git", "show", f"{ref}:{path}"])
    except subprocess.CalledProcessError:
        return None


def _prov_lines(text: str | None) -> list[str]:
    if text is None:
        return []
    return [
        L.rstrip()
        for L in text.splitlines()[:60]
        if _PROV_RE.match(L)
    ]


def _by_stem_and_ext(rows: list[tuple[str, str]]) -> dict[str, dict[str, str]]:
    """Group rows by basis stem. Returns ``{stem: {".g94": status, ".ecp": status}}``."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for status, path in rows:
        rel = path[len(BASIS_PREFIX):]
        if "/" in rel:
            continue
        stem, _, ext = rel.rpartition(".")
        if ext not in ("g94", "ecp"):
            continue
        out[stem][f".{ext}"] = status
    return out


def _print_section(title: str, lines: list[str]) -> None:
    if not lines:
        return
    print(f"\n## {title}")
    for L in lines:
        print(L)


def summarise(ref_a: str, ref_b: str | None) -> int:
    rows = _name_status(ref_a, ref_b)
    if not rows:
        print(f"No changes under {BASIS_PREFIX} between "
              f"{ref_a} and {ref_b or 'working tree'}.")
        return 0

    grouped = _by_stem_and_ext(rows)

    added_g94: list[str] = []
    removed_g94: list[str] = []
    modified_g94: list[str] = []
    added_ecp: list[str] = []
    removed_ecp: list[str] = []
    modified_ecp: list[str] = []
    pair_warnings: list[str] = []
    prov_diffs: list[str] = []

    for stem, exts in sorted(grouped.items()):
        g_status = exts.get(".g94")
        e_status = exts.get(".ecp")

        if g_status == "A":
            added_g94.append(stem)
            if e_status is None:
                pass  # orbital-only basis, fine
        elif g_status == "D":
            removed_g94.append(stem)
            if e_status != "D":
                # .g94 removed but sidecar left behind
                sister = "no change" if e_status is None else e_status
                pair_warnings.append(
                    f"  {stem}: .g94 removed but .ecp status = {sister} "
                    f"(potential orphan sidecar)"
                )
        elif g_status == "M":
            modified_g94.append(stem)

        if e_status == "A":
            added_ecp.append(stem)
            if g_status is None:
                pair_warnings.append(
                    f"  {stem}: .ecp added but no matching .g94 change "
                    f"(orphan sidecar?)"
                )
        elif e_status == "D":
            removed_ecp.append(stem)
            if g_status != "D":
                pair_warnings.append(
                    f"  {stem}: .ecp removed but .g94 status = "
                    f"{g_status or 'no change'} (lost ECP attribution?)"
                )
        elif e_status == "M":
            modified_ecp.append(stem)

        # Provenance-header diff on modified .g94.
        if g_status == "M":
            path = f"{BASIS_PREFIX}{stem}.g94"
            before = _prov_lines(_show(ref_a, path))
            after = _prov_lines(_show(ref_b, path))
            if before != after:
                prov_diffs.append(f"\n### {stem}.g94")
                if before:
                    prov_diffs.append("  - before:")
                    for L in before:
                        prov_diffs.append(f"      {L}")
                else:
                    prov_diffs.append("  - before: (no provenance header)")
                if after:
                    prov_diffs.append("  - after:")
                    for L in after:
                        prov_diffs.append(f"      {L}")
                else:
                    prov_diffs.append("  - after: (no provenance header)")

    label_b = ref_b or "working tree"
    print(f"# basis_library/basis/  diff:  {ref_a}  →  {label_b}")
    print(
        f"\nSummary: "
        f"+{len(added_g94)} / -{len(removed_g94)} / ~{len(modified_g94)} .g94, "
        f"+{len(added_ecp)} / -{len(removed_ecp)} / ~{len(modified_ecp)} .ecp"
    )

    _print_section("Pair-mismatch warnings", pair_warnings)
    _print_section("Added .g94", [f"  + {s}" for s in added_g94])
    _print_section("Removed .g94", [f"  - {s}" for s in removed_g94])
    _print_section("Added .ecp sidecars", [f"  + {s}.ecp" for s in added_ecp])
    _print_section("Removed .ecp sidecars", [f"  - {s}.ecp" for s in removed_ecp])
    _print_section("Modified .ecp sidecars", [f"  ~ {s}.ecp" for s in modified_ecp])
    _print_section("Modified .g94 (orbital data changed)",
                   [f"  ~ {s}" for s in modified_g94])
    _print_section("Provenance-header changes on modified .g94", prov_diffs)

    return 0 if not pair_warnings else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ref_a", nargs="?", default="HEAD",
                   help="left-hand git ref (default: HEAD)")
    p.add_argument("ref_b", nargs="?", default=None,
                   help="right-hand git ref (default: working tree)")
    args = p.parse_args(argv)
    return summarise(args.ref_a, args.ref_b)


if __name__ == "__main__":
    sys.exit(main())
