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

    import re

    pin = re.search(r'QVF_TAG: "(v[0-9.]+)"', (DOCS.parent / ".gitlab-ci.yml").read_text())
    assert pin is not None
    guide = snapshot.with_name("library_guide.md")
    for target, upstream in (
        ("cli.md", "docs/cli.md"),
        ("fortran.md", "docs/fortran.md"),
        ("c_api.md", "docs/c_api.md"),
        ("../cpp/include/qvf/qvf_c.h", "cpp/include/qvf/qvf_c.h"),
        ("../js/README.md", "js/README.md"),
    ):
        assert resolve(str(guide), target) == (
            f"https://github.com/vibe-qc/qvf/blob/{pin.group(1)}/{upstream}"
        )

    unrelated = (
        (DOCS / "other" / "GOVERNANCE.md", "conformance/README.md"),
        (snapshot, "conformance/missing.md"),
        (snapshot.with_name("unlisted.md"), "conformance/README.md"),
    )
    for source, target in unrelated:
        assert resolve(str(source), target) is None, (source, target)


def test_included_and_external_document_links_render(tmp_path: Path) -> None:
    """Repository navigation survives inclusion in the published Sphinx site."""
    import subprocess
    import sys

    pytest.importorskip("sphinx")
    pytest.importorskip("myst_parser")
    source = tmp_path / "docs"
    (source / "user_guide").mkdir(parents=True)
    (tmp_path / "studies/skala-showcase").mkdir(parents=True)
    (source / "conf.py").write_text(
        "extensions = ['myst_parser']\nmaster_doc = 'index'\n", encoding="utf-8"
    )
    (source / "index.md").write_text(
        "# Navigation\n\n```{toctree}\ncontributing\nuser_guide/skala\ntesting\n```\n",
        encoding="utf-8",
    )
    (source / "contributing.md").write_text(
        "```{include} ../CONTRIBUTING.md\n```\n", encoding="utf-8"
    )
    (source / "testing.md").write_text("# Testing\n", encoding="utf-8")
    (tmp_path / "studies/skala-showcase/README.md").write_text(
        "# Campaign\n\n## Select BIPOLE for the LiF inputs\n", encoding="utf-8"
    )
    cases = (
        (REPO / "CONTRIBUTING.md", "testing policy and inventory",
         tmp_path / "CONTRIBUTING.md", "# Contributing\n\n"),
        (DOCS / "user_guide/skala.md", "campaign instructions",
         source / "user_guide/skala.md", "# SKALA\n\n"),
    )
    for original, label, fixture, heading in cases:
        # Read the link under test from the real source, preserving its target.
        match = re.search(r"\[" + re.escape(label) + r"\]\([^\n)]+\)",
                          original.read_text(encoding="utf-8"))
        assert match is not None, label
        fixture.write_text(heading + match.group(0) + "\n", encoding="utf-8")

    def build(name: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "sphinx", "-b", "html", "-W",
             "--keep-going", "-q", str(source), str(tmp_path / name)],
            capture_output=True, text=True, timeout=60,
        )

    rendered = build("html")
    assert rendered.returncode == 0, rendered.stderr
    for page, label, destination in (
        ("contributing.html", "testing policy and inventory",
         "https://vibe-qc.com/docs/testing.html"),
        ("user_guide/skala.html", "campaign instructions",
         "https://github.com/vibe-qc/vibe-qc/blob/main/studies/skala-showcase/README.md"
         "#select-bipole-for-the-lif-inputs"),
    ):
        html = (tmp_path / "html" / page).read_text(encoding="utf-8")
        assert f'href="{destination}"' in html, label

    # Strict checking must still reject an unrelated typo in the same source.
    with (source / "user_guide/skala.md").open("a", encoding="utf-8") as stream:
        stream.write("\n[unrelated reference](missing-document.md)\n")
    control = build("control-html")
    assert control.returncode != 0
    assert "missing-document" in control.stderr, control.stderr
