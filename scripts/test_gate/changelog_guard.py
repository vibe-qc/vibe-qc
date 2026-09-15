#!/usr/bin/env python3
"""Keep released CHANGELOG.md sections as released unless an amendment is pinned.

Every released ``## [vX.Y.Z]`` section of ``CHANGELOG.md`` is pinned by the
SHA-256 of its body in ``changelog_pins.toml`` beside this script. A body is
every line after the section header up to, but not including, the next
``## [`` header, joined with newlines. ``[Unreleased]`` is never pinned.

``check`` fails when a released section no longer matches its pin, when a
released section has no pin, when a pin names a section that does not exist,
or when a section header appears twice. Pinning whole bodies catches what a
heading comparison cannot: an entry under a generic ``### Added`` heading, a
paragraph appended to an existing entry, and the rebase that re-applies an
``[Unreleased]`` hunk under a header that a promotion commit has since
relabelled (#227, #229). ``check`` needs no git history, so it also runs in a
shallow CI clone.

A deliberate change to a released section is allowed only by re-pinning it
with a stated reason, in the same commit::

    python scripts/test_gate/changelog_guard.py pin vX.Y.Z --amended "why"

After cutting a release, pin its new section::

    python scripts/test_gate/changelog_guard.py pin vX.Y.Z

``audit`` compares each pinned section with the same section at its git tag
and reports differences that have no recorded reason. It needs the tags, so it
belongs in the release pre-flight rather than in the test suite.

It runs from ``tests/test_changelog_released_sections.py`` (tier T1) and from
``.githooks/pre-push``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CHANGELOG = ROOT / "CHANGELOG.md"
DEFAULT_PINS = HERE / "changelog_pins.toml"

SECTION_RE = re.compile(r"^## \[([^\]]+)\]")
RELEASED_RE = re.compile(r"^v\d")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

PINS_HEADER = """\
# Pins for the released sections of CHANGELOG.md (#229).
#
# Each entry is the SHA-256 of one released section's body: the lines after its
# "## [vX.Y.Z]" header up to the next "## [" header, joined with "\\n".
# scripts/test_gate/changelog_guard.py compares them with CHANGELOG.md, from
# tests/test_changelog_released_sections.py (tier T1) and .githooks/pre-push.
#
# Do not edit hashes by hand. After cutting a release, pin its new section:
#     python scripts/test_gate/changelog_guard.py pin vX.Y.Z
# A deliberate change to an already released section needs a stated reason:
#     python scripts/test_gate/changelog_guard.py pin vX.Y.Z --amended "why"
# An entry for work that has not been released belongs under [Unreleased].
"""


class GuardError(Exception):
    """A usage or data problem, as opposed to a failed check."""


def split_sections(text: str) -> list[tuple[str, list[str]]]:
    """Every ``## [name]`` section in file order, with its body lines."""
    sections: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    for line in text.split("\n"):
        match = SECTION_RE.match(line)
        if match:
            current = []
            sections.append((match.group(1), current))
            continue
        if current is not None:
            current.append(line)
    return sections


def section_order(text: str) -> list[str]:
    """Released section names in file order, duplicates included."""
    return [name for name, _ in split_sections(text) if RELEASED_RE.match(name)]


def released_bodies(text: str) -> tuple[dict[str, list[str]], list[str]]:
    """Released section bodies by name, and the names that occur more than once."""
    bodies: dict[str, list[str]] = {}
    duplicates: list[str] = []
    for name, body in split_sections(text):
        if not RELEASED_RE.match(name):
            continue
        if name in bodies:
            duplicates.append(name)
            continue
        bodies[name] = body
    return bodies, duplicates


def body_hash(body: list[str]) -> str:
    return hashlib.sha256("\n".join(body).encode("utf-8")).hexdigest()


def load_pins(path: Path) -> dict[str, dict[str, str]]:
    if tomllib is None:
        raise GuardError("Python 3.11 or newer is required to read the pins (tomllib)")
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    sections = data.get("sections", {})
    if not isinstance(sections, dict):
        raise GuardError(f"{path}: [sections] must be a table")
    pins: dict[str, dict[str, str]] = {}
    for name, entry in sections.items():
        if not isinstance(entry, dict):
            raise GuardError(f"{path}: pin for [{name}] must be a table")
        unknown = set(entry) - {"sha256", "amended"}
        if unknown:
            raise GuardError(f"{path}: pin for [{name}] has unknown keys {sorted(unknown)}")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise GuardError(f"{path}: pin for [{name}] has no valid sha256")
        reason = entry.get("amended")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise GuardError(f"{path}: pin for [{name}] has an empty amended reason")
        pins[name] = dict(entry)
    return pins


def dump_pins(pins: dict[str, dict[str, str]], order: list[str]) -> str:
    names = [name for name in dict.fromkeys(order) if name in pins]
    names += sorted(name for name in pins if name not in names)
    out = [PINS_HEADER]
    for name in names:
        entry = pins[name]
        out.append(f"[sections.{json.dumps(name)}]")
        out.append(f'sha256 = "{entry["sha256"]}"')
        if "amended" in entry:
            out.append(f"amended = {json.dumps(entry['amended'], ensure_ascii=False)}")
        out.append("")
    return "\n".join(out)


def check(text: str, pins: dict[str, dict[str, str]]) -> list[str]:
    """Problems with CHANGELOG ``text`` against ``pins``; empty when it passes."""
    bodies, duplicates = released_bodies(text)
    problems = [f"[{name}] appears more than once in CHANGELOG.md" for name in duplicates]
    for name, body in bodies.items():
        pin = pins.get(name)
        if pin is None:
            problems.append(
                f"[{name}] is a released section with no pin. If it was just cut, "
                f"run `python scripts/test_gate/changelog_guard.py pin {name}`."
            )
        elif body_hash(body) != pin["sha256"]:
            problems.append(
                f"[{name}] no longer matches its pin. An entry for unreleased work "
                f"belongs under [Unreleased]; compare with `git show {name}:CHANGELOG.md`. "
                f"If the change is a deliberate amendment, run "
                f"`python scripts/test_gate/changelog_guard.py pin {name} --amended \"<reason>\"` "
                f"and commit the pin with it."
            )
    for name in pins:
        if name not in bodies:
            problems.append(f"pin for [{name}] names no section in CHANGELOG.md")
    return problems


def pin(
    text: str,
    pins: dict[str, dict[str, str]],
    versions: list[str],
    amended: str | None = None,
    all_unpinned: bool = False,
) -> dict[str, dict[str, str]]:
    """Return ``pins`` with ``versions`` (and optionally every unpinned section) pinned.

    Changing the hash of an existing pin requires ``amended``; the reason is
    appended to any reason already recorded.
    """
    bodies, duplicates = released_bodies(text)
    if duplicates:
        raise GuardError(f"sections appear more than once: {', '.join(duplicates)}")
    if amended is not None and not amended.strip():
        raise GuardError("--amended needs a non-empty reason")
    if amended is not None and all_unpinned:
        raise GuardError("--amended applies to named versions, not to --all-unpinned")
    targets = list(dict.fromkeys(versions))
    if all_unpinned:
        targets += [name for name in bodies if name not in pins and name not in targets]
    if not targets:
        raise GuardError("nothing to pin")
    new = {name: dict(entry) for name, entry in pins.items()}
    for name in targets:
        if name not in bodies:
            raise GuardError(f"[{name}] is not a released section in CHANGELOG.md")
        digest = body_hash(bodies[name])
        old = pins.get(name)
        if old is not None and old["sha256"] != digest and amended is None:
            raise GuardError(
                f"[{name}] is already pinned and has changed; a deliberate amendment "
                f'needs --amended "<reason>"'
            )
        entry = {"sha256": digest}
        previous = old.get("amended") if old else None
        if amended is not None:
            entry["amended"] = f"{previous}; {amended}" if previous else amended
        elif previous:
            entry["amended"] = previous
        new[name] = entry
    return new


def audit(
    text: str, pins: dict[str, dict[str, str]], repo: Path
) -> tuple[list[str], list[str], list[str]]:
    """Compare pinned sections with their git tags.

    Returns ``(problems, checked, skipped)``: a problem is a pinned section
    whose content differs from its tag's without a recorded ``amended`` reason;
    sections whose tag is not available locally are skipped.
    """
    problems: list[str] = []
    checked: list[str] = []
    skipped: list[str] = []
    for name in dict.fromkeys(section_order(text)):
        entry = pins.get(name)
        if entry is None:
            continue
        shown = subprocess.run(
            ["git", "show", f"refs/tags/{name}:CHANGELOG.md"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        if shown.returncode != 0:
            skipped.append(name)
            continue
        checked.append(name)
        at_tag, _ = released_bodies(shown.stdout)
        if "amended" in entry:
            continue
        if name not in at_tag:
            problems.append(
                f"[{name}] is absent from CHANGELOG.md at tag {name}, and its pin records no reason"
            )
        elif body_hash(at_tag[name]) != entry["sha256"]:
            problems.append(
                f"[{name}] differs from its content at tag {name}, and its pin records "
                f'no reason; re-pin it with --amended "<reason>"'
            )
    return problems, checked, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Keep released CHANGELOG.md sections as released (#229)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--changelog", type=Path, default=DEFAULT_CHANGELOG)
        p.add_argument("--pins", type=Path, default=DEFAULT_PINS)

    common(sub.add_parser("check", help="fail if a released section changed without a pin"))
    pin_parser = sub.add_parser("pin", help="pin released sections")
    common(pin_parser)
    pin_parser.add_argument("versions", nargs="*", help="section names such as v0.17.2")
    pin_parser.add_argument("--amended", help="reason for changing an already released section")
    pin_parser.add_argument(
        "--all-unpinned", action="store_true", help="also pin every released section without a pin"
    )
    audit_parser = sub.add_parser("audit", help="compare pinned sections with their git tags")
    common(audit_parser)
    audit_parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    try:
        text = args.changelog.read_text(encoding="utf-8")
        if args.command == "pin":
            pins = load_pins(args.pins) if args.pins.exists() else {}
            new = pin(
                text, pins, args.versions, amended=args.amended, all_unpinned=args.all_unpinned
            )
            args.pins.write_text(dump_pins(new, section_order(text)), encoding="utf-8")
            added = sum(1 for name in new if name not in pins)
            changed = sum(1 for name in new if name in pins and new[name] != pins[name])
            print(f"changelog guard: pinned {added} new and updated {changed} section(s) in {args.pins}")
            return 0
        pins = load_pins(args.pins)
        if args.command == "check":
            problems = check(text, pins)
            if problems:
                for problem in problems:
                    print(f"changelog guard: {problem}", file=sys.stderr)
                return 1
            print(f"changelog guard: {len(released_bodies(text)[0])} released sections match their pins")
            return 0
        problems, checked, skipped = audit(text, pins, args.repo)
        for problem in problems:
            print(f"changelog guard: {problem}", file=sys.stderr)
        print(
            f"changelog guard: audited {len(checked)} tagged section(s), "
            f"{len(skipped)} without a local tag, {len(problems)} problem(s)"
        )
        return 1 if problems else 0
    except (GuardError, OSError) as exc:
        print(f"changelog guard: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
