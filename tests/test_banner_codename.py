"""Banner codename catalog + lookup.

Pin the contract:

  1. ``codename_for_version`` returns the documented codename for
     each registered release.
  2. Dev-version inputs (``"0.5.0.dev0"``, ``"0.5.0a1"``,
     ``"0.5.0rc1"``) inherit the codename of the upcoming release.
  3. Patch releases inherit the parent minor's codename.
  4. Unknown / unregistered versions return ``None``.
  5. The runtime banner string for the current build embeds the
     codename verbatim.
"""

from __future__ import annotations

from vibeqc.banner import (
    RELEASE_CODENAMES,
    VIBEQC_VERSION,
    banner,
    codename_for_version,
)

# ---------------------------------------------------------------------------
# Catalog contract
# ---------------------------------------------------------------------------


def test_v010_codename_matches_catalog():
    assert codename_for_version("0.1.0") == "Roothaan's Raven"


def test_v040_codename_matches_catalog():
    assert codename_for_version("0.4.0") == "Hellmann's Hedgehog"
    assert codename_for_version("0.4.0") == "Hellmann's Hedgehog"


def test_v050_codename_is_wilsons_otter():
    assert codename_for_version("0.5.0") == "Wilson's Otter"


def test_release_codenames_dict_is_public_and_well_formed():
    """RELEASE_CODENAMES is part of the public banner API (so
    docs/conf.py can import + display it on the landing page).

    Originally the policy was "minor releases only" — patches inherit
    the parent minor's codename via the fallback in
    codename_for_version. In practice we've added patch-specific
    overrides (0.7.1 Pulay's Triangle, 0.7.2 Boys' Crucible, 0.7.3
    Whitten's Bridge — each a different theme from 0.7.0 Löwdin's
    Compass), so the real rule is "every key is a well-formed
    PEP-440-ish X.Y.Z release version" — minor- or patch-level.

    Enforced here: keys are exactly three integer-valued
    dot-separated components. Dev / alpha / beta / rc suffixes
    aren't allowed in catalog keys (the codename_for_version
    resolver strips those before lookup).
    """
    assert isinstance(RELEASE_CODENAMES, dict)
    for key in RELEASE_CODENAMES:
        parts = key.split(".")
        assert len(parts) == 3, (
            f"RELEASE_CODENAMES key {key!r} has {len(parts)} "
            f"dot-separated parts, expected 3 (X.Y.Z)"
        )
        for i, part in enumerate(parts):
            assert part.isdigit(), (
                f"RELEASE_CODENAMES key {key!r}: part {i} ({part!r}) "
                f"is not a plain integer; PEP-440 suffixes belong in "
                "codename_for_version's resolution layer, not in "
                "catalog keys"
            )


# ---------------------------------------------------------------------------
# Inheritance: dev suffixes + patch versions
# ---------------------------------------------------------------------------


def test_dev_version_inherits_upcoming_codename():
    """v0.5.0.dev0 / .dev1 / a1 / b1 / rc1 all inherit v0.5.0's
    codename — the codename is color bound to the release, and
    dev builds heading toward that release wear it."""
    for variant in (
        "0.5.0.dev0",
        "0.5.0.dev1",
        "0.5.0a1",
        "0.5.0b2",
        "0.5.0rc3",
    ):
        assert codename_for_version(variant) == "Wilson's Otter", (
            f"variant {variant!r} did not inherit Wilson's Otter"
        )


def test_patch_version_inherits_parent_minor_codename():
    """v0.4.1 / v0.4.2 / v0.4.99 inherit v0.4.0."""
    for patch in ("0.4.1", "0.4.2", "0.4.99"):
        assert codename_for_version(patch) == "Hellmann's Hedgehog", (
            f"patch {patch!r} did not inherit Hellmann's Hedgehog"
        )
    # Same for v0.5: patches inherit the v0.5.0 codename.
    assert codename_for_version("0.5.1") == "Wilson's Otter"


def test_dev_patch_combo_resolves_correctly():
    """A dev build of a patch release — say ``0.4.2.dev0`` — should
    strip the dev suffix, then fall back to the parent minor's
    codename."""
    assert codename_for_version("0.4.2.dev0") == "Hellmann's Hedgehog"
    assert codename_for_version("0.5.1.dev0") == "Wilson's Otter"


# ---------------------------------------------------------------------------
# Unknown versions
# ---------------------------------------------------------------------------


def test_unknown_minor_returns_none():
    """A minor we haven't tagged returns ``None`` rather than
    raising — keeps banner code simple on installs against an
    arbitrary git checkout.

    Note: this test used to use ``0.7.0`` and ``0.7.0.dev0`` as the
    "future, unassigned" sentinels. Both v0.7 (Löwdin's Compass / Pulay's
    Triangle / Boys' Crucible / Whitten's Bridge) and v0.8.0 (Grimme's
    Gecko) are now in the catalog, so the sentinels moved to a
    forward-looking unassigned minor (``0.99.0``). Update this test in
    tandem when you ship a new minor whose codename was previously
    listed here as the "unassigned" placeholder — pre-commit doesn't
    run pytest, so a stale sentinel here doesn't fail the commit, but
    it does fail the test suite the next time it's run.
    """
    assert codename_for_version("99.99.99") is None
    assert codename_for_version("0.99.0") is None  # future, unassigned
    assert codename_for_version("0.99.0.dev0") is None
    assert codename_for_version("2.0.0") is None


# ---------------------------------------------------------------------------
# Banner integration
# ---------------------------------------------------------------------------


def test_current_banner_carries_codename_when_applicable():
    """If the build is running against a version that has a codename
    (e.g. 0.5.0.dev0 inheriting Wilson's Otter), the rendered banner
    string must include it. If the build is on a version with no
    codename, the banner must NOT carry a quoted codename."""
    expected_codename = codename_for_version(VIBEQC_VERSION)
    text = banner()
    if expected_codename is not None:
        assert f'"{expected_codename}"' in text, (
            f"banner did not embed codename {expected_codename!r}: {text!r}"
        )
    else:
        # No quoted codename should appear on the version-descriptor
        # line (skip the box-drawing top line).
        first_line = text.splitlines()[1]
        assert '"' not in first_line, (
            f"unnamed-version banner unexpectedly embeds a quoted "
            f"codename: {first_line!r}"
        )
