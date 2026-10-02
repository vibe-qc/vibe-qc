"""The changelog fragment directory and its assembler.

These cover the property the fragment scheme exists for: two changes landing
independently must never touch the same file, and assembling must not disturb
released sections, which `changelog_guard.py` pins.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "assemble_changelog.py"
FRAGMENTS = ROOT / "changelog.d"


def _module():
    spec = importlib.util.spec_from_file_location("assemble_changelog", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_assembler_exists_and_imports():
    assert SCRIPT.is_file(), "scripts/assemble_changelog.py is missing"
    assert hasattr(_module(), "assemble")


def test_every_fragment_is_well_formed():
    """A fragment must read as changelog prose, not as a stray note."""
    mod = _module()
    for name, text in mod.fragments(FRAGMENTS):
        first = text.splitlines()[0]
        assert first.startswith("### ") or first.startswith("- "), (
            "%s starts with %r; a fragment must open with a '### ' heading or a "
            "'- ' bullet so it reads correctly once folded in" % (name, first[:40])
        )


def test_assembling_leaves_released_sections_byte_identical(tmp_path):
    """The guard pins released bodies. Assembly must not be able to break a pin."""
    mod = _module()
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "## [Unreleased]\n\n### Added\n\n- existing entry\n\n"
        "## [v1.2.3]\n\n### Fixed\n\n- a released entry that is pinned\n"
    )
    d = tmp_path / "changelog.d"
    d.mkdir()
    (d / "999-new.md").write_text("### Fixed: something new (#999)\n\nProse.\n")

    out = mod.assemble(changelog, d)
    released = out[out.index("## [v1.2.3]") :]
    assert released == "## [v1.2.3]\n\n### Fixed\n\n- a released entry that is pinned\n"
    assert "### Fixed: something new (#999)" in out[: out.index("## [v1.2.3]")]
    assert "- existing entry" in out


def test_assembly_is_deterministic_regardless_of_discovery_order(tmp_path):
    """Two fragments must fold in filename order, not filesystem order."""
    mod = _module()
    changelog = tmp_path / "CHANGELOG.md"
    base = "## [Unreleased]\n\n## [v1.0.0]\n\n- old\n"
    changelog.write_text(base)
    d = tmp_path / "changelog.d"
    d.mkdir()
    (d / "b-second.md").write_text("- second\n")
    (d / "a-first.md").write_text("- first\n")
    out = mod.assemble(changelog, d)
    assert out.index("- first") < out.index("- second")


def test_readme_is_not_treated_as_a_fragment(tmp_path):
    mod = _module()
    d = tmp_path / "changelog.d"
    d.mkdir()
    (d / "README.md").write_text("# Changelog fragments\n\nInstructions, not an entry.\n")
    assert mod.fragments(d) == []


@pytest.mark.skipif(not FRAGMENTS.is_dir(), reason="no changelog.d in this tree")
def test_the_real_fragment_directory_assembles():
    mod = _module()
    out = mod.assemble(ROOT / "CHANGELOG.md", FRAGMENTS)
    assert out.startswith("## [Unreleased]")
