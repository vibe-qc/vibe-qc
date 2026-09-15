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
