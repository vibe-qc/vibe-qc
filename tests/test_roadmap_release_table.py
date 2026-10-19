"""The roadmap's release table cannot drift away from the repository.

`docs/roadmap.md` used to state the current release in prose, and that
prose went stale six releases in a row (issue #259). The release table
replaced it, and these tests are what stop the table going the same way:
every fact in it is cross-checked against a source that moves on its own
when a release is cut.

  1. Every codename in the table equals ``RELEASE_CODENAMES``.
  2. Every minor with a ``CHANGELOG.md`` section has a row.
  3. Every minor with a git tag in this repository has a row.
  4. Every row that claims "shipped" is backed by a changelog section
     or is a documented pre-changelog release, and no row claims to be
     planned while a tag or changelog section exists for it.
  5. The table lists one row per catalogued release line. Patch
     releases inherit their minor's codename and stay out; the three
     v0.7.x patches that carry their own codename are the documented
     exception.

The git-tag check is skipped outside a git checkout so a source tarball
still passes.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from vibeqc.banner import RELEASE_CODENAMES

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROADMAP = _REPO_ROOT / "docs" / "roadmap.md"
_CHANGELOG = _REPO_ROOT / "CHANGELOG.md"

# A table row: | `v0.9.0` | Knowles's Kingfisher | headline | shipped ... |
_ROW = re.compile(
    r"^\|\s*`(v\d+\.\d+\.\d+)`\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$"
)
# A released section header, either dash style the file has used.
_CHANGELOG_SECTION = re.compile(r"^## \[(v\d+\.\d+\.\d+)\]")

# CHANGELOG.md starts at v0.9.0; everything before it is covered by the
# file's "Historical Codename Artwork (pre-v0.9.0)" section instead. A row
# for one of these cannot be backed by a changelog section.
_PRE_CHANGELOG = {
    "v0.1.0",
    "v0.2.0",
    "v0.2.5",
    "v0.3.0",
    "v0.4.0",
    "v0.5.0",
    "v0.6.0",
    "v0.7.0",
    "v0.8.0",
}

# The subset of those whose release date this repository cannot source at
# all: their tags stayed in the monorepo and no roadmap milestone heading
# dates them, so "shipped, pre-split" is the only honest status.
_PRE_SPLIT_UNDATED = {
    "v0.2.0",
    "v0.2.5",
    "v0.3.0",
    "v0.6.0",
    "v0.7.0",
    "v0.8.0",
}

# vibe-view's `[v2.0.0]` stanza was removed from CHANGELOG.md on #368.
# Keep the escape hatch for any future foreign releases that leak in:
# they get no roadmap row and are silently ignored.
_FOREIGN_CHANGELOG_SECTIONS: frozenset[str] = frozenset()


def _table_rows() -> list[tuple[str, str, str, str]]:
    """Rows of the roadmap's "## Release table" section."""
    lines = _ROADMAP.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index("## Release table")
    except ValueError:  # pragma: no cover - the heading is the contract
        pytest.fail("docs/roadmap.md has no '## Release table' heading")

    rows = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        match = _ROW.match(line)
        if match:
            rows.append(match.groups())
    assert rows, "the release table parsed to zero rows"
    return rows


def _changelog_minors() -> set[str]:
    return {
        m.group(1)
        for line in _CHANGELOG.read_text(encoding="utf-8").splitlines()
        if (m := _CHANGELOG_SECTION.match(line)) and m.group(1).endswith(".0")
    }


def _tagged_minors() -> set[str]:
    """``vX.Y.0`` for every ``vX.Y.Z`` tag in this repository."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "tag", "-l", "v*"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout: cannot cross-check tags")
    minors = set()
    for tag in out.split():
        parts = tag.lstrip("v").split(".")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            minors.add(f"v{parts[0]}.{parts[1]}.0")
    if not minors:
        pytest.skip("no release tags in this checkout")
    return minors


# Patch releases inherit their minor's codename. These three carry their
# own entry in RELEASE_CODENAMES (docs/roadmap.md, "Release codenames") and
# are deliberately not release-table rows.
_PATCH_CODENAME_EXCEPTIONS = {"v0.7.1", "v0.7.2", "v0.7.3"}


def _catalogued_releases() -> dict[str, str]:
    """Release lines that must each have exactly one table row."""
    return {
        f"v{version}": name
        for version, name in RELEASE_CODENAMES.items()
        if f"v{version}" not in _PATCH_CODENAME_EXCEPTIONS
    }


# ---------------------------------------------------------------------------
# 1. Codenames agree with the runtime catalog
# ---------------------------------------------------------------------------


def test_every_row_codename_matches_the_banner_catalog():
    catalog = _catalogued_releases()
    mismatches = [
        (version, codename, catalog.get(version))
        for version, codename, _headline, _status in _table_rows()
        if catalog.get(version) != codename
    ]
    assert not mismatches, (
        "roadmap release table disagrees with RELEASE_CODENAMES in "
        f"python/vibeqc/banner.py (version, table, catalog): {mismatches}"
    )


def test_the_table_is_exactly_the_codename_catalog():
    listed = {version for version, *_ in _table_rows()}
    catalogued = set(_catalogued_releases())
    missing = sorted(catalogued - listed)
    extra = sorted(listed - catalogued)
    assert not missing, (
        "these releases carry a codename but have no roadmap "
        f"release-table row: {missing}"
    )
    assert not extra, (
        "these release-table rows have no RELEASE_CODENAMES entry: "
        f"{extra}"
    )


# ---------------------------------------------------------------------------
# 2-3. Every shipped minor the repository knows about has a row
# ---------------------------------------------------------------------------


def test_every_changelog_minor_has_a_row():
    listed = {version for version, *_ in _table_rows()}
    missing = sorted(
        _changelog_minors() - listed - _FOREIGN_CHANGELOG_SECTIONS
    )
    assert not missing, (
        "these minors have a CHANGELOG.md section but no roadmap "
        f"release-table row: {missing}"
    )


def test_every_tagged_minor_has_a_row():
    listed = {version for version, *_ in _table_rows()}
    missing = sorted(_tagged_minors() - listed)
    assert not missing, (
        f"these minors are tagged but have no release-table row: {missing}"
    )


# ---------------------------------------------------------------------------
# 4. Status columns tell the truth
# ---------------------------------------------------------------------------


def test_no_row_claims_to_be_planned_while_it_has_shipped():
    shipped = _changelog_minors() | _tagged_minors()
    lying = [
        (version, status)
        for version, _codename, _headline, status in _table_rows()
        if "planned" in status.lower() and version in shipped
    ]
    assert not lying, (
        "these rows are marked planned but have a tag or a changelog "
        f"section: {lying}"
    )


def test_every_shipped_row_is_backed_by_a_source():
    changelog = _changelog_minors()
    unbacked = [
        (version, status)
        for version, _codename, _headline, status in _table_rows()
        if status.lower().startswith("shipped")
        and version not in changelog
        and version not in _PRE_CHANGELOG
    ]
    assert not unbacked, (
        "these rows claim shipped but have no CHANGELOG.md section and are "
        f"not pre-changelog releases: {unbacked}"
    )


def test_undated_rows_are_exactly_the_pre_split_ones():
    marked = {
        version
        for version, _codename, _headline, status in _table_rows()
        if "pre-split" in status.lower()
    }
    assert marked == _PRE_SPLIT_UNDATED, (
        "the set of rows marked 'shipped, pre-split' moved. A release that "
        "gains a date or a changelog section should lose the marker; one "
        "that never had a tag here should keep it. "
        f"table={sorted(marked)} expected={sorted(_PRE_SPLIT_UNDATED)}"
    )


# ---------------------------------------------------------------------------
# 5. One row per catalogued release line
# ---------------------------------------------------------------------------


def test_patch_codename_exceptions_stay_out_of_the_table():
    listed = {version for version, *_ in _table_rows()}
    leaked = sorted(_PATCH_CODENAME_EXCEPTIONS & listed)
    assert not leaked, (
        "patch releases inherit their minor's codename and do not belong in "
        f"the release table: {leaked}"
    )
    stale = sorted(
        version
        for version in _PATCH_CODENAME_EXCEPTIONS
        if version.lstrip("v") not in RELEASE_CODENAMES
    )
    assert not stale, (
        "these exceptions no longer exist in RELEASE_CODENAMES; drop them "
        f"from _PATCH_CODENAME_EXCEPTIONS: {stale}"
    )
