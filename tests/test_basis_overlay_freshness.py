"""The build overlay must not shadow a newer committed basis library.

``setup_native_deps.sh`` assembles ``build/basis_library`` once and nothing
invalidates it when the committed library changes, while the resolution in
``vibeqc.__init__`` *prefers* the overlay. A clone whose overlay predates a
basis-data commit therefore resolves the older library with no warning: the
observed case was ``def2-svp`` missing silver after 6664da63d (#88) added it,
which killed eleven guess tests in BasisSet construction and made the seven
negative ECP-contract cases among them look like a refusal regression.
"""

from __future__ import annotations

import os
import warnings

import pytest

import vibeqc


def _make_library(root, sizes):
    """Minimal tree that satisfies _basis_library_has_standard_payload."""
    basis = root / "basis"
    basis.mkdir(parents=True)
    for name, size in sizes.items():
        (basis / name).write_text("x" * size)
    return root


_CANARIES = ("def2-svp.g94", "6-311+g3df2p.g94", "pob-tzvp-rev2.g94")


def test_stale_overlay_warns_and_names_both_paths(tmp_path):
    """A size difference in any canary is reported, with the fix command."""
    overlay = _make_library(tmp_path / "build" / "basis_library",
                            dict.fromkeys(_CANARIES, 100))
    bundled = _make_library(tmp_path / "pkg" / "basis_library",
                            dict.fromkeys(_CANARIES, 300))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vibeqc._warn_if_basis_overlay_is_stale(overlay, bundled)
    assert len(caught) == 1
    text = str(caught[0].message)
    assert "out of date" in text
    assert str(overlay / "basis" / "def2-svp.g94") in text
    assert str(bundled / "basis" / "def2-svp.g94") in text
    assert "setup_basis_library.sh" in text


def test_current_overlay_is_silent(tmp_path):
    """No warning when the overlay matches the committed library."""
    overlay = _make_library(tmp_path / "build" / "basis_library",
                            dict.fromkeys(_CANARIES, 250))
    bundled = _make_library(tmp_path / "pkg" / "basis_library",
                            dict.fromkeys(_CANARIES, 250))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vibeqc._warn_if_basis_overlay_is_stale(overlay, bundled)
    assert not caught


def test_a_basis_absent_from_the_overlay_warns(tmp_path):
    """The case the canary comparison structurally cannot see.

    A newly added basis changes no existing file, so comparing the three
    canaries by size cannot notice it. This is what happened when the
    sixteen cc-pVnZ-PP sets landed (2026-09-08): every clone's overlay kept
    resolving without them and nothing said so, the symptom being
    ``BasisSet: no shells loaded for basis 'cc-pvdz-pp'`` naming a basis
    that is plainly committed.
    """
    overlay = _make_library(tmp_path / "build" / "basis_library",
                            dict.fromkeys(_CANARIES, 250))
    bundled = _make_library(tmp_path / "pkg" / "basis_library",
                            dict.fromkeys(_CANARIES, 250))
    for name in ("cc-pvdz-pp.g94", "cc-pvdz-pp.ecp", "cc-pvtz-pp.g94"):
        (bundled / "basis" / name).write_text("x" * 50)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vibeqc._warn_if_basis_overlay_is_stale(overlay, bundled)
    assert len(caught) == 1
    text = str(caught[0].message)
    assert "out of date" in text
    assert "3 file(s) missing" in text
    assert "cc-pvdz-pp.g94" in text          # names what to look for
    assert "setup_basis_library.sh" in text


def test_extra_files_in_the_overlay_are_not_staleness(tmp_path):
    """The overlay merges libint's own payload with ``custom/``, so it may
    legitimately carry names the committed tree does not. Only *missing*
    names mean the overlay is behind; warning on extras would fire on every
    clone whose libint ships a basis vibe-qc does not commit."""
    overlay = _make_library(tmp_path / "build" / "basis_library",
                            dict.fromkeys(_CANARIES, 250))
    bundled = _make_library(tmp_path / "pkg" / "basis_library",
                            dict.fromkeys(_CANARIES, 250))
    (overlay / "basis" / "some-libint-only-basis.g94").write_text("x" * 50)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vibeqc._warn_if_basis_overlay_is_stale(overlay, bundled)
    assert not caught


def test_this_clone_overlay_carries_the_pp_family():
    """The concrete regression: whatever the resolution picks must have the
    PP orbital sets, or heavy-element ECP work fails with a message that
    blames the basis name rather than the stale overlay."""
    root = vibeqc._resolve_basis_library()
    if root is None:
        pytest.skip("no basis library resolved in this environment")
    missing = [
        f"{aug}cc-p{core}{zeta}-pp{ext}"
        for zeta in ("dz", "tz", "qz", "5z")
        for core in ("v", "wcv")
        for aug in ("", "aug-")
        for ext in (".g94", ".ecp")
        if not (root / "basis" / f"{aug}cc-p{core}{zeta}-pp{ext}").is_file()
    ]
    assert not missing, (
        f"{root / 'basis'} is missing {len(missing)} PP file(s), e.g. "
        f"{missing[:3]}: this library predates the 2026-09-08 import. "
        "Run: bash scripts/setup_basis_library.sh"
    )


def test_missing_bundled_library_is_not_reported_as_stale(tmp_path):
    """An installed wheel has no committed tree to compare against."""
    overlay = _make_library(tmp_path / "build" / "basis_library",
                            dict.fromkeys(_CANARIES, 100))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vibeqc._warn_if_basis_overlay_is_stale(overlay, tmp_path / "absent")
    assert not caught


def test_this_clone_resolves_a_library_that_actually_carries_silver():
    """The regression itself: whatever the resolution picks must have the
    heavy def2 elements #88 added, or ECP work silently resolves pre-#88
    data. Reads the resolved library rather than the committed one."""
    root = vibeqc._resolve_basis_library()
    if root is None:
        pytest.skip("no basis library resolved in this environment")
    text = (root / "basis" / "def2-svp.g94").read_text()
    assert "\nAg     0\n" in text, (
        f"{root / 'basis' / 'def2-svp.g94'} has no silver block: this "
        "library predates #88. Run: bash scripts/setup_basis_library.sh"
    )


# ---------------------------------------------------------------------------
# Content, not size (#235): an in-place correction that keeps a file's length
# was invisible to the size-only canary comparison.
# ---------------------------------------------------------------------------


def _catch(overlay, bundled):
    """Run the guard on the two trees and return the warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vibeqc._warn_if_basis_overlay_is_stale(overlay, bundled)
    return caught


def test_same_size_sidecar_edit_is_detected(tmp_path):
    """#207 changed pob-tzvp-rev2.ecp's radial powers at constant length
    (27,983 bytes before and after). Same size, different bytes: warn."""
    files = dict.fromkeys(_CANARIES, 250)
    files["pob-tzvp-rev2.ecp"] = 120
    overlay = _make_library(tmp_path / "build" / "basis_library", files)
    bundled = _make_library(tmp_path / "pkg" / "basis_library", files)
    stale = overlay / "basis" / "pob-tzvp-rev2.ecp"
    fixed = bundled / "basis" / "pob-tzvp-rev2.ecp"
    fixed.write_text("2" * 120)  # the corrected sidecar: same length
    assert stale.stat().st_size == fixed.stat().st_size

    caught = _catch(overlay, bundled)
    assert len(caught) == 1
    text = str(caught[0].message)
    assert "out of date" in text
    assert str(stale) in text and str(fixed) in text
    assert "different content" in text
    assert "setup_basis_library.sh" in text


def test_same_size_canary_edit_is_detected(tmp_path):
    """The three orbital canaries are compared by bytes as well."""
    files = dict.fromkeys(_CANARIES, 250)
    overlay = _make_library(tmp_path / "build" / "basis_library", files)
    bundled = _make_library(tmp_path / "pkg" / "basis_library", files)
    (bundled / "basis" / "6-311+g3df2p.g94").write_text("y" * 250)

    caught = _catch(overlay, bundled)
    assert len(caught) == 1
    edited = bundled / "basis" / "6-311+g3df2p.g94"
    assert str(edited) in str(caught[0].message)


def test_every_committed_sidecar_is_compared(tmp_path):
    """Not only the canaries and not only ``pob-tzvp-rev2.ecp``: any
    ``.ecp`` the committed tree carries is checked against its overlay
    copy, so a fix hard-coded to one sidecar name would fail here."""
    files = dict.fromkeys(_CANARIES, 250)
    files.update({"a.ecp": 40, "b.ecp": 40, "c.ecp": 40})
    overlay = _make_library(tmp_path / "build" / "basis_library", files)
    bundled = _make_library(tmp_path / "pkg" / "basis_library", files)
    (bundled / "basis" / "c.ecp").write_text("z" * 40)
    caught = _catch(overlay, bundled)
    assert len(caught) == 1
    assert str(bundled / "basis" / "c.ecp") in str(caught[0].message)


def test_identical_sidecars_stay_silent(tmp_path):
    """Equal bytes everywhere: the content pass adds no false alarm."""
    files = dict.fromkeys(_CANARIES, 250)
    files.update({"a.ecp": 40, "pob-tzvp-rev2.ecp": 120})
    overlay = _make_library(tmp_path / "build" / "basis_library", files)
    bundled = _make_library(tmp_path / "pkg" / "basis_library", files)
    assert not _catch(overlay, bundled)


def test_both_staleness_classes_are_reported_in_one_warning(tmp_path):
    """An overlay behind by a missing basis AND a same-size sidecar edit
    (the #235 field case: built before #207 and before a later addition)
    reports both in the one warning, so the inventory line cannot hide the
    content line."""
    files = dict.fromkeys(_CANARIES, 250)
    files["pob-tzvp-rev2.ecp"] = 120
    overlay = _make_library(tmp_path / "build" / "basis_library", files)
    bundled = _make_library(tmp_path / "pkg" / "basis_library", files)
    (bundled / "basis" / "pob-tzvp-rev2.ecp").write_text("2" * 120)
    (bundled / "basis" / "cc-pvdz-pp.g94").write_text("x" * 50)

    caught = _catch(overlay, bundled)
    assert len(caught) == 1
    text = str(caught[0].message)
    assert "1 file(s) missing from the overlay: cc-pvdz-pp.g94" in text
    assert "different content: pob-tzvp-rev2.ecp" in text
    assert "may be older than the committed ones" in text


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs a non-root POSIX user so chmod 000 denies reads",
)
def test_an_unreadable_pair_does_not_silence_a_later_stale_sidecar(tmp_path):
    """One file the guard cannot read is skipped on its own; the stale
    sidecar sorted after it is still reported."""
    files = dict.fromkeys(_CANARIES, 250)
    files.update({"a.ecp": 40, "pob-tzvp-rev2.ecp": 120})
    overlay = _make_library(tmp_path / "build" / "basis_library", files)
    bundled = _make_library(tmp_path / "pkg" / "basis_library", files)
    (bundled / "basis" / "pob-tzvp-rev2.ecp").write_text("2" * 120)
    unreadable = overlay / "basis" / "a.ecp"
    os.chmod(unreadable, 0)
    try:
        caught = _catch(overlay, bundled)
    finally:
        os.chmod(unreadable, 0o644)
    assert len(caught) == 1
    assert "different content: pob-tzvp-rev2.ecp" in str(caught[0].message)

