#!/usr/bin/env python3
"""Render or check the CRYSTAL23 capability matrix tables (#593).

The machine-readable source is ``benchmarks/crystal23_capability_matrix.json``.
``docs/crystal23_capability_matrix.md`` carries hand-written prose plus one
generated block between ``<!-- matrix:begin -->`` and ``<!-- matrix:end -->``.

    python scripts/crystal23_capability_matrix.py --write   # refresh the block
    python scripts/crystal23_capability_matrix.py --check   # exit 1 if stale

The script is native-free and imports nothing from ``vibeqc``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "benchmarks" / "crystal23_capability_matrix.json"
DOC_PATH = ROOT / "docs" / "crystal23_capability_matrix.md"
BEGIN = "<!-- matrix:begin -->"
END = "<!-- matrix:end -->"
STATE_ORDER = ("numerically-accepted", "source-tested", "implemented", "unverified", "unsupported")
ROW_FIELDS = ("id", "group", "crystal23", "vibeqc", "state", "evidence", "gap", "issues")


def load_matrix(path: Path = JSON_PATH) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_matrix(data)
    return data


def validate_matrix(data: dict) -> None:
    states = data["states"]
    if tuple(sorted(states)) != tuple(sorted(STATE_ORDER)):
        raise ValueError(f"state vocabulary must be {STATE_ORDER}; got {sorted(states)}")
    seen = set()
    for row in data["rows"]:
        if set(row) != set(ROW_FIELDS):
            raise ValueError(f"row {row.get('id')!r}: fields must be {ROW_FIELDS}")
        if not row["id"] or row["id"] in seen:
            raise ValueError(f"duplicate or empty row id {row['id']!r}")
        seen.add(row["id"])
        if row["state"] not in states:
            raise ValueError(f"row {row['id']!r}: unknown state {row['state']!r}")
        for name in ("group", "crystal23", "vibeqc", "gap"):
            if not str(row[name]).strip():
                raise ValueError(f"row {row['id']!r}: {name} is empty")
        if row["state"] in ("numerically-accepted", "unverified") and not row["evidence"].strip():
            raise ValueError(f"row {row['id']!r}: {row['state']} needs evidence text")
        for ref in row["issues"]:
            if not (ref.startswith("#") and ref[1:].isdigit()):
                raise ValueError(f"row {row['id']!r}: issue reference {ref!r} must be '#<number>'")
        if "|" in row["crystal23"] or "|" in row["vibeqc"]:
            raise ValueError(f"row {row['id']!r}: table cells must not contain '|'")


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "/")


def render_block(data: dict) -> str:
    rows = data["rows"]
    counts = {state: sum(r["state"] == state for r in rows) for state in STATE_ORDER}
    lines = [BEGIN, "", f"Updated {data['updated']}; {len(rows)} rows.", "",
             "| State | Rows | Meaning |", "|---|---:|---|"]
    for state in STATE_ORDER:
        lines.append(f"| `{state}` | {counts[state]} | {_cell(data['states'][state])} |")
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    for group, members in groups.items():
        lines += ["", f"### {group}", "",
                  "| Row | CRYSTAL23 capability | vibe-qc route | State | Evidence | Remaining gap | Issues |",
                  "|---|---|---|---|---|---|---|"]
        for r in members:
            issues = ", ".join(r["issues"]) or "none"
            lines.append(
                f"| `{r['id']}` | {_cell(r['crystal23'])} | {_cell(r['vibeqc'])} | `{r['state']}` "
                f"| {_cell(r['evidence']) or 'none'} | {_cell(r['gap'])} | {issues} |"
            )
    lines += ["", END]
    return "\n".join(lines)


def splice(doc: str, block: str) -> str:
    start, stop = doc.index(BEGIN), doc.index(END) + len(END)
    return doc[:start] + block + doc[stop:]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    data = load_matrix()
    doc = DOC_PATH.read_text(encoding="utf-8")
    updated = splice(doc, render_block(data))
    if args.write:
        DOC_PATH.write_text(updated, encoding="utf-8")
        print(f"wrote {DOC_PATH.relative_to(ROOT)} ({len(data['rows'])} rows)")
        return 0
    if updated != doc:
        print(f"{DOC_PATH.relative_to(ROOT)} is stale; run scripts/crystal23_capability_matrix.py --write",
              file=sys.stderr)
        return 1
    print("capability matrix doc is current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
