"""Download links resolve, and split companions own their release artifacts.

Legacy files are archived privately. Core docs must not infer a current
companion version from a removed sibling checkout.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

# Matches the link target, e.g. ../_static/downloads/vibeview-2.12.1-...whl
_DOWNLOAD_LINK = re.compile(r"\(([^)\s]*_static/downloads/[^)\s]+)\)")


def _links() -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for md in DOCS.rglob("*.md"):
        if "_build" in md.parts:
            continue
        for target in _DOWNLOAD_LINK.findall(md.read_text(encoding="utf-8")):
            found.append((md, target))
    return found


def test_download_links_resolve() -> None:
    links = _links()
    missing = []
    for md, target in links:
        resolved = (md.parent / target).resolve()
        if not resolved.is_file():
            missing.append(f"{md.relative_to(REPO)} -> {target}")
    assert not missing, (
        "download links pointing at files that do not exist:\n  " + "\n  ".join(missing)
    )


def test_exactly_one_qvf_writer_artifact() -> None:
    """The unversioned toolkit download surface keeps only its current archive."""
    downloads = DOCS / "_static" / "downloads"
    if not downloads.is_dir():
        pytest.skip("no downloads directory")
    matches = sorted(p.name for p in downloads.glob("qvf-writer-*") if p.is_file())
    assert len(matches) <= 1, f"more than one qvf-writer artifact staged: {matches}"


def test_companion_downloads_belong_to_their_own_releases() -> None:
    """Core docs must not advertise retained legacy artifacts as current."""
    for md, target in _links():
        if "_vendored" in md.parts:
            continue  # upstream-owned snapshot, changed only by re-vendoring
        assert not re.search(r"(?:vibeview-.*\.whl|qvf-writer-.*\.tar\.gz)$", target), (
            f"{md.relative_to(REPO)} recommends legacy companion artifact {target}"
        )

    for page, repository in [
        ("tutorial/vibe_view_getting_started.md", "vibe-view"),
        ("qvf/index.md", "qvf"),
    ]:
        text = (DOCS / page).read_text(encoding="utf-8")
        assert f"https://github.com/vibe-qc/{repository}/releases" in text
        assert "legacy" in text.lower()

    assert (DOCS / "_static/downloads/README.md").is_file()


def test_qvf_snapshot_link_resolution_preserves_unrelated_references(monkeypatch) -> None:
    """A known upstream link resolves; other documents and typos still fail."""
    import runpy
    import sys

    # Loading the Sphinx configuration needs no Sphinx installation, and its
    # import-path setup must not affect the remainder of this pytest process.
    monkeypatch.setattr(sys, "path", list(sys.path))
    resolve = runpy.run_path(str(DOCS / "conf.py"))["_pinned_qvf_link"]
    snapshot = DOCS / "qvf" / "_vendored" / "GOVERNANCE.md"
    assert resolve(str(snapshot), "conformance/README.md") == "/qvf/conformance"

    unrelated = (
        (DOCS / "other" / "GOVERNANCE.md", "conformance/README.md"),
        (snapshot, "conformance/missing.md"),
        (snapshot.with_name("unlisted.md"), "conformance/README.md"),
    )
    for source, target in unrelated:
        assert resolve(str(source), target) is None, (source, target)
