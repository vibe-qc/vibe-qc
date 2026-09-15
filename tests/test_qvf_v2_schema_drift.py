"""Drift guard: the v2 manifest schema must be v1 + the documented delta.

v2 started as a hand-copied snapshot of v1 and silently fell behind it.
By the time this guard was written, v2 was missing ten section kinds
(``bond_orders``, ``spectra.epr``, ``volume.rdg``, ``volume.potential``,
``basis.ao``, ``fermi_surface``, ``phonon.bands``, ``phonon.dos``,
``equation_of_state``, ``topology.qtaim``), the four root blocks
(``thermochemistry``, ``dipole_moment``, ``constraints``,
``extensions``), the Section ``critical`` flag, and the fat-band
``projections`` member -- while its own ``description`` claimed it
"differs from v1 only in that SectionReactionPath carries the per-frame
lattice + dimensionality".

Because the manifest root sets ``additionalProperties: false`` and
``Section`` is a closed ``oneOf``, any archive that carried both a
periodic ``reaction.path`` (which forces ``qvf_version=2``) and one of
those sections was rejected by :func:`write_qvf`'s own validation gate.

The fix makes v2 *derived* from v1, so this file pins the invariant:

* the runtime v2 schema equals v1 everywhere except the five documented
  delta points, asserted structurally rather than by re-running the
  derivation (a test that only compared against ``_derive_v2_schema``
  would rubber-stamp any new undocumented change to that function);
* v2 is a strict superset of v1 -- every v1 section kind and root
  property survives into v2;
* the generated on-disk copy matches the derivation, i.e. someone ran
  ``scripts/gen_qvf_v2_schema.py`` after touching v1.

Sibling in spirit to ``qvf-writer/python/tests/test_packaging_schema.py``.
"""

from __future__ import annotations

import json

import pytest

from vibeqc.output.formats.qvf import (
    _SCHEMA_PATH_V1,
    _SCHEMA_PATH_V2,
    _derive_v2_schema,
    _load_canonical_schema,
)

# The complete, documented v1 -> v2 delta. Anything outside this set that
# differs between the two schemas is drift.
_DELTA_ROOT_KEYS = {"$id", "title", "description"}
_DELTA_ROOT_PROPERTIES = {"qvf_version"}
# The v2 delta collapsed to identity metadata when the qvf_version 2 bump was
# withdrawn (2026-07-10) and `lattice` moved into v1: no $defs differ at all.
_DELTA_DEFS: set[str] = set()


@pytest.fixture(scope="module")
def v1() -> dict:
    with open(_SCHEMA_PATH_V1, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def v2() -> dict:
    return _load_canonical_schema(2)


class TestV2IsV1PlusDocumentedDelta:
    def test_root_keys_differ_only_in_delta(self, v1, v2):
        assert set(v1) == set(v2), "v2 must not add or drop top-level keys"
        differing = {k for k in v1 if v1[k] != v2[k]}
        # `properties` and `$defs` differ, but only in their delta members —
        # pinned by the two tests below.
        assert differing <= _DELTA_ROOT_KEYS | {"properties", "$defs"}

    def test_root_properties_differ_only_in_qvf_version(self, v1, v2):
        p1, p2 = v1["properties"], v2["properties"]
        assert set(p1) == set(p2), (
            "v2 root properties drifted from v1: "
            f"only-v1={sorted(set(p1) - set(p2))} "
            f"only-v2={sorted(set(p2) - set(p1))}"
        )
        differing = {k for k in p1 if p1[k] != p2[k]}
        assert differing == _DELTA_ROOT_PROPERTIES

    def test_defs_are_identical(self, v1, v2):
        """Since the qvf_version 2 bump was withdrawn and `lattice` moved into
        v1, the v2 delta is pure identity metadata: `$defs` must match exactly.

        Any difference here means someone re-introduced a v2-only schema
        feature — precisely the fork that caused the original bug.
        """
        d1, d2 = v1["$defs"], v2["$defs"]
        assert set(d2) - set(d1) == set(), f"v2 added $defs: {sorted(set(d2) - set(d1))}"
        assert set(d1) - set(d2) == set(), f"v2 dropped $defs: {sorted(set(d1) - set(d2))}"
        differing = {k for k in d1 if d1[k] != d2[k]}
        assert differing <= _DELTA_DEFS, (
            f"v2 silently altered shared $defs: {sorted(differing - _DELTA_DEFS)}"
        )

    def test_qvf_version_const_is_the_only_version_change(self, v1, v2):
        assert v1["properties"]["qvf_version"]["const"] == 1
        assert v2["properties"]["qvf_version"]["const"] == 2
        assert v2["properties"]["qvf_version"]["type"] == "integer"

    def test_v1_owns_the_optional_lattice_member(self, v1, v2):
        """Governance ruling 2026-07-10: `lattice` is an *optional* member of
        **v1**. A periodic reaction path is detected by its presence, not by a
        version bump. This is the assertion that pins the ruling.
        """
        m1 = v1["$defs"]["SectionReactionPath"]["properties"]["members"]
        m2 = v2["$defs"]["SectionReactionPath"]["properties"]["members"]
        # v1 carries it, and v2 (being v1 modulo the version const) matches.
        assert "lattice" in m1["properties"]
        assert m1["properties"]["lattice"] == {"$ref": "#/$defs/ReactionPathLattice"}
        assert "ReactionPathLattice" in v1["$defs"]
        assert m1["properties"] == m2["properties"]
        # Optional: a molecular path (no lattice) stays valid.
        assert m1["required"] == m2["required"]
        assert "lattice" not in m1["required"]


class TestV2IsAStrictSupersetOfV1:
    """Every v1 section kind must still validate inside a v2 archive."""

    def test_every_v1_section_branch_survives(self, v1, v2):
        refs1 = {b["$ref"] for b in v1["$defs"]["Section"]["oneOf"]}
        refs2 = {b["$ref"] for b in v2["$defs"]["Section"]["oneOf"]}
        assert not refs1 - refs2, (
            f"v2 Section.oneOf dropped v1 branches: {sorted(refs1 - refs2)}"
        )

    def test_section_keeps_the_critical_flag(self, v2):
        # QVF spec § 5.5 — dropped by the old hand-maintained fork.
        assert "critical" in v2["$defs"]["Section"]["properties"]

    def test_bands_keeps_fat_band_projections(self, v2):
        bands = v2["$defs"]["SectionBands"]["properties"]["members"]["properties"]
        assert "projections" in bands

    @pytest.mark.parametrize(
        "root_block",
        ["thermochemistry", "dipole_moment", "constraints", "extensions"],
    )
    def test_root_blocks_present(self, v2, root_block):
        assert root_block in v2["properties"]

    @pytest.mark.parametrize(
        "def_name",
        [
            "SectionBasisAO",
            "SectionBondOrders",
            "SectionEquationOfState",
            "SectionFermiSurface",
            "SectionPhononBands",
            "SectionPhononDOS",
            "SectionSpectraEPR",
            "SectionTopologyQTAIM",
            "SectionVolumePotential",
            "SectionVolumeRDG",
        ],
    )
    def test_section_defs_present(self, v2, def_name):
        assert def_name in v2["$defs"]


class TestGeneratedFileIsCurrent:
    def test_on_disk_v2_matches_the_derivation(self, v1):
        """``scripts/gen_qvf_v2_schema.py`` must have been re-run after v1 moved.

        The runtime never reads this file -- it derives v2 in memory -- but
        external validators resolving the ``$id`` do.
        """
        assert _SCHEMA_PATH_V2.is_file()
        with open(_SCHEMA_PATH_V2, encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk == _derive_v2_schema(v1), (
            "qvf_manifest_v2.schema.json is stale — regenerate it with "
            "`python scripts/gen_qvf_v2_schema.py`"
        )

    def test_v2_is_a_valid_draft202012_schema(self, v2):
        import jsonschema

        jsonschema.Draft202012Validator.check_schema(v2)


class TestLoaderContract:
    def test_deriving_v2_does_not_mutate_the_cached_v1(self):
        """v1 and v2 share a module-level cache; the derivation must deep-copy.

        ``_load_canonical_schema(2)`` loads v1, caches it, then derives. If
        the derivation mutated in place, every later ``_load_canonical_schema(1)``
        caller would validate v1 archives against a v2 schema.
        """
        _load_canonical_schema(2)
        cached_v1 = _load_canonical_schema(1)
        assert cached_v1["properties"]["qvf_version"]["const"] == 1
        assert cached_v1["$id"].endswith("/qvf/1/manifest.schema.json")
        # v1 owns `lattice` (ruling 2026-07-10); the derivation must not strip
        # it, nor may it stamp v1 with the v2 identity metadata.
        assert "ReactionPathLattice" in cached_v1["$defs"]
        rxn = cached_v1["$defs"]["SectionReactionPath"]
        assert "lattice" in rxn["properties"]["members"]["properties"]

    def test_unknown_version_rejected(self):
        with pytest.raises(ValueError, match="unknown qvf_version"):
            _load_canonical_schema(3)
