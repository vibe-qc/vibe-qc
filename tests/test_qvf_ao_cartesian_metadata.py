"""``basis.ao`` metadata must name a Cartesian component, not a fabricated m.

:func:`vibeqc.output.formats.qvf.qvf_ao_data` used to derive the magnetic
quantum number as ``ao_local - l`` for every shell.  That identity holds
only for a **pure** shell, whose ``2l+1`` components are the real spherical
harmonics ordered ``m = -l ... +l``.  A **Cartesian** shell
(``ShellInfo.pure == False``) has ``(l+1)(l+2)/2`` components and no
magnetic quantum number at all, so the subtraction ran past ``+l`` and
emitted a number that denotes nothing: a Cartesian d shell came out as
``m = -2 ... +3`` across its six components.

Nothing else was affected.  The volumetric payload is correct because it
comes from ``evaluate_ao_on_grid``, and the AO index ranges are correct
because ``vibeqc.cube._ao_ranges`` already handles both cases.  The bug was
confined to the ``angular_momentum`` label in ``ao_metadata``.

A Cartesian component is now named by ``cartesian_powers = [lx, ly, lz]``
in libint's lexicographic in-shell order — the ordering QVF's format spec
mandates in Appendix A.2 — as an *optional* member of the ``basis.ao``
``ao_metadata`` contract.  Optional keeps the change additive under the QVF
versioning policy (``qvf-writer/GOVERNANCE.md``): no ``qvf_version`` bump,
and every spherical archive written to date stays valid byte-for-byte.

Reachable only through the explicit-``ShellInfo`` ``BasisSet`` constructor
(``cpp/src/bindings.cpp``).  The named-basis route forces ``set_pure(true)``
(``cpp/src/basis.cpp``), so every shipped basis set is spherical — which is
why this survived unnoticed.

Sibling context: ``_basis_shell_payload`` (the ``wavefunction.gto`` path)
rejects Cartesian shells outright, because the MO coefficients it writes
carry a normalization convention Cartesian shells break.  That reasoning
does **not** transfer here — ``basis.ao`` ships evaluated grid data, not
coefficients — so ``qvf_ao_data`` describes Cartesian shells rather than
refusing them.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vibeqc._vibeqc_core import Atom, BasisSet, Molecule, ShellInfo
from vibeqc.cube import CubeGrid, _ao_ranges, _cartesian_component_powers
from vibeqc.output.formats.qvf import (
    _SCHEMA_PATH_V1,
    qvf_ao_data,
    validate_qvf,
    write_qvf,
)
from vibeqc.output.plan import OutputPlan

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The three schema copies that must track the canonical SSOT.  vibe-view's
# is a symlink to it; the two qvf-writer copies are deliberate copies kept
# in lock-step by hand (`handovers/HANDOVER_QVF_CXX_WRITER.md`).
_SCHEMA_COPIES = (
    _REPO_ROOT / "qvf-writer" / "spec" / "qvf_manifest.schema.json",
    _REPO_ROOT / "qvf-writer" / "python" / "qvf_manifest.schema.json",
    _REPO_ROOT / "vibe-view" / "src" / "vibeview" / "schema.json",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _molecule() -> Molecule:
    return Molecule([Atom(8, [0.0, 0.0, 0.0])])


def _shell(l: int, pure: bool) -> ShellInfo:
    sh = ShellInfo()
    sh.atom_index = 0
    sh.l = l
    sh.pure = pure
    sh.exponents = [1.0]
    sh.coefficients = [1.0]
    sh.origin = [0.0, 0.0, 0.0]
    return sh


def _basis(mol: Molecule, l: int, pure: bool) -> BasisSet:
    return BasisSet(mol, [_shell(l, pure)], name="<cartesian-metadata-probe>")


def _axis_grid() -> CubeGrid:
    """A 3x3x3 grid whose voxel centers sit on the Cartesian axes.

    Voxel centers run -0.5, 0.0, +0.5 bohr per axis, so ``(2, 1, 1)`` is
    the point ``(+0.5, 0, 0)``, ``(1, 2, 1)`` is ``(0, +0.5, 0)`` and
    ``(1, 1, 2)`` is ``(0, 0, +0.5)``.  On those three points a monomial
    ``x^lx y^ly z^lz`` is nonzero iff the other two powers vanish, which is
    what makes the payload identifiable from the metadata alone.
    """
    return CubeGrid(
        origin=np.array([-0.5, -0.5, -0.5]),
        spacing=np.array([0.5, 0.5, 0.5]),
        shape=(3, 3, 3),
    )


_ON_X = (2, 1, 1)
_ON_Y = (1, 2, 1)
_ON_Z = (1, 1, 2)


def _ao_entries(l: int, pure: bool, **kwargs) -> list[dict]:
    mol = _molecule()
    return qvf_ao_data(
        _basis(mol, l, pure),
        mol,
        grid=_axis_grid(),
        include_contracted=kwargs.pop("include_contracted", True),
        include_primitives=kwargs.pop("include_primitives", False),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The regression itself
# ---------------------------------------------------------------------------


class TestCartesianComponentsNeverClaimASphericalM:
    def test_cartesian_d_shell_emits_no_out_of_range_m(self):
        """The bug's signature: m ran -2 ... +3 over the six components."""
        entries = _ao_entries(2, pure=False)
        assert len(entries) == 6, "a Cartesian d shell has (l+1)(l+2)/2 = 6 AOs"
        for e in entries:
            am = e["ao_metadata"]["angular_momentum"]
            assert am[0] == 2
            assert -2 <= am[1] <= 2, (
                "angular_momentum[1] left the spherical range [-l, +l]: "
                f"{am} — a Cartesian component has no magnetic quantum number"
            )

    def test_cartesian_shell_carries_powers_in_libint_order(self):
        entries = _ao_entries(2, pure=False)
        powers = [tuple(e["ao_metadata"]["cartesian_powers"]) for e in entries]
        # libint: lexicographic, lx decreasing first, then ly (spec App. A.2).
        assert powers == [
            (2, 0, 0),
            (1, 1, 0),
            (1, 0, 1),
            (0, 2, 0),
            (0, 1, 1),
            (0, 0, 2),
        ]
        assert powers == _cartesian_component_powers(2)

    @pytest.mark.parametrize("l", [0, 1, 2, 3])
    def test_powers_sum_to_l_and_count_matches_the_ao_range(self, l):
        entries = _ao_entries(l, pure=False)
        assert len(entries) == (l + 1) * (l + 2) // 2
        mol = _molecule()
        lo, hi = _ao_ranges(_basis(mol, l, pure=False))[0]
        assert hi - lo == len(entries)
        for e in entries:
            meta = e["ao_metadata"]
            assert sum(meta["cartesian_powers"]) == meta["angular_momentum"][0] == l

    def test_pure_shell_keeps_its_spherical_m_and_omits_powers(self):
        entries = _ao_entries(2, pure=True)
        assert len(entries) == 5, "a pure d shell has 2l+1 = 5 AOs"
        assert [e["ao_metadata"]["angular_momentum"] for e in entries] == [
            [2, -2],
            [2, -1],
            [2, 0],
            [2, 1],
            [2, 2],
        ]
        for e in entries:
            assert "cartesian_powers" not in e["ao_metadata"], (
                "a spherical AO must not carry Cartesian powers — its absence "
                "is what tells a consumer angular_momentum[1] is a real m"
            )


class TestMetadataMatchesThePayload:
    """The point of the fix: the label must describe the shipped grid data."""

    @pytest.mark.parametrize("l", [1, 2, 3])
    def test_cartesian_powers_identify_the_evaluated_monomial(self, l):
        for e in _ao_entries(l, pure=False):
            lx, ly, lz = e["ao_metadata"]["cartesian_powers"]
            data = e["data"]
            # x^lx y^ly z^lz on the +x axis point is nonzero iff ly = lz = 0,
            # and correspondingly for the +y and +z axis points.
            for point, own in ((_ON_X, lx), (_ON_Y, ly), (_ON_Z, lz)):
                expect_nonzero = own == l
                assert bool(abs(data[point]) > 1e-12) is expect_nonzero, (
                    f"component ({lx},{ly},{lz}) of the Cartesian l={l} shell "
                    f"has the wrong value at grid point {point}: "
                    f"{data[point]!r}"
                )

    def test_pure_p_components_are_y_z_x_in_that_order(self):
        """Pinning the ``m = -1, 0, +1`` ordering the metadata now claims.

        libint's real solid harmonics put ``m = -1`` (p_y) first, then
        ``m = 0`` (p_z), then ``m = +1`` (p_x).  This is the convention
        ``ao_local - l`` encodes, and the reason it is right for pure
        shells and wrong for Cartesian ones.
        """
        entries = _ao_entries(1, pure=True)
        on_axis = {_ON_X: "x", _ON_Y: "y", _ON_Z: "z"}
        seen = []
        for e in entries:
            live = [
                axis
                for point, axis in on_axis.items()
                if abs(e["data"][point]) > 1e-12
            ]
            assert len(live) == 1, f"a p component lit up {live}"
            seen.append((e["ao_metadata"]["angular_momentum"][1], live[0]))
        assert seen == [(-1, "y"), (0, "z"), (1, "x")]


class TestPrimitiveSections:
    """``evaluate_single_primitive_on_grid`` returns the shell's component 0.

    ``ao_index`` already pointed at component 0 (the shell's AO range
    start) while ``angular_momentum`` claimed a nominal ``m = 0``.  For
    ``l > 0`` those are different AOs, so the metadata contradicted both
    its own ``ao_index`` and the grid data beside it.
    """

    def test_pure_primitive_names_component_zero_not_m_zero(self):
        entries = _ao_entries(
            1, pure=True, include_contracted=False, include_primitives=True
        )
        assert len(entries) == 1
        meta = entries[0]["ao_metadata"]
        assert meta["is_primitive"] is True
        assert meta["angular_momentum"] == [1, -1]
        assert "cartesian_powers" not in meta
        mol = _molecule()
        assert meta["ao_index"] == _ao_ranges(_basis(mol, 1, pure=True))[0][0]
        # m = -1 is p_y: nonzero on +y only.
        data = entries[0]["data"]
        assert abs(data[_ON_Y]) > 1e-12
        assert abs(data[_ON_X]) < 1e-12
        assert abs(data[_ON_Z]) < 1e-12

    def test_cartesian_primitive_names_component_zero_by_its_powers(self):
        entries = _ao_entries(
            2, pure=False, include_contracted=False, include_primitives=True
        )
        assert len(entries) == 1
        meta = entries[0]["ao_metadata"]
        assert meta["cartesian_powers"] == [2, 0, 0]
        assert meta["angular_momentum"] == [2, 0]
        # (2, 0, 0) is x²: nonzero on +x only.
        data = entries[0]["data"]
        assert abs(data[_ON_X]) > 1e-12
        assert abs(data[_ON_Y]) < 1e-12
        assert abs(data[_ON_Z]) < 1e-12


# ---------------------------------------------------------------------------
# The QVF contract
# ---------------------------------------------------------------------------


def _stub_result():
    return type(
        "Result",
        (),
        {"converged": True, "energy": -75.0, "n_iter": 1, "scf_trace": ()},
    )()


class TestSectionIdsDistinguishComponents:
    """Found while writing the round-trip above: every component of a shell
    used to get the same section id and the same path inside the archive.

    ``_ao_label`` / ``_ao_section_id`` accepted an ``m`` argument and never
    used it, so a shell's AOs collided -- ``write_qvf`` refused its own
    output with "duplicate section id" for *any* shell with ``l > 0``,
    Cartesian or not.  An s shell has one component and keeps its old id.
    """

    def test_s_shell_ids_are_unchanged(self):
        entries = _ao_entries(0, pure=True)
        assert [e["section_id"] for e in entries] == ["ao_O_s_s0_contracted"]

    def test_pure_shell_ids_carry_m(self):
        entries = _ao_entries(1, pure=True)
        assert [e["section_id"] for e in entries] == [
            "ao_O_p_m-1_s0_contracted",
            "ao_O_p_m0_s0_contracted",
            "ao_O_p_m1_s0_contracted",
        ]

    def test_cartesian_shell_ids_carry_the_monomial(self):
        entries = _ao_entries(2, pure=False)
        assert [e["section_id"] for e in entries] == [
            "ao_O_d_xx_s0_contracted",
            "ao_O_d_xy_s0_contracted",
            "ao_O_d_xz_s0_contracted",
            "ao_O_d_yy_s0_contracted",
            "ao_O_d_yz_s0_contracted",
            "ao_O_d_zz_s0_contracted",
        ]

    @pytest.mark.parametrize("pure", [True, False])
    @pytest.mark.parametrize("l", [0, 1, 2, 3])
    def test_ids_and_labels_are_unique_within_a_shell(self, l, pure):
        entries = _ao_entries(l, pure=pure)
        ids = [e["section_id"] for e in entries]
        labels = [e["label"] for e in entries]
        assert len(set(ids)) == len(ids), f"colliding section ids: {ids}"
        assert len(set(labels)) == len(labels), f"colliding labels: {labels}"

    @pytest.mark.parametrize("pure", [True, False])
    def test_ids_match_the_qvf_id_charset(self, pure):
        # Design § 2.4: a section id matches ^[A-Za-z0-9_.-]+$ — which is why
        # a negative m is spelled `m-1` and not with a leading '+' for m >= 0.
        for l in (0, 1, 2, 3):
            for e in _ao_entries(l, pure=pure):
                assert re.fullmatch(r"[A-Za-z0-9_.-]+", e["section_id"]), e[
                    "section_id"
                ]


class TestNamedBasisRoute:
    """The shipped path, which the duplicate-id bug also broke.

    Cartesian shells need the explicit-``ShellInfo`` constructor, but the
    id collision needed only ``l > 0`` — so exporting contracted AOs of any
    ordinary basis with a p shell produced an archive ``write_qvf`` refused.
    ``examples/vibe_view/showcase_basis_functions.py`` does exactly that
    with pob-TZVP on water.
    """

    def test_water_sto3g_contracted_aos_round_trip(self, tmp_path):
        mol = Molecule(
            [
                Atom(8, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 1.43, -0.93]),
                Atom(1, [0.0, -1.43, -0.93]),
            ]
        )
        basis = BasisSet(mol, "sto-3g")
        entries = qvf_ao_data(
            basis,
            mol,
            grid=_axis_grid(),
            include_contracted=True,
            include_primitives=False,
        )
        # O 1s, O 2s, O 2p x3, H 1s x2 = 7 AOs.
        assert len(entries) == basis.nbasis == 7
        ids = [e["section_id"] for e in entries]
        assert len(set(ids)) == len(ids), f"colliding section ids: {ids}"
        # A named basis is always spherical (`set_pure(true)`), so no AO here
        # may carry Cartesian powers.
        for e in entries:
            assert "cartesian_powers" not in e["ao_metadata"]
        assert "ao_O_p_m-1_s2_contracted" in ids

        plan = OutputPlan.from_run_job_kwargs(
            output=tmp_path / "water_ao", method="rhf", basis="sto-3g", functional=None
        )
        path = write_qvf(
            tmp_path / "water_ao",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            ao_data=entries,
        )
        report = validate_qvf(path)
        assert report["valid"], f"Validation errors: {report.get('errors', [])}"


class TestArchiveContract:
    def test_cartesian_archive_validates_and_keeps_the_powers(self, tmp_path):
        mol = _molecule()
        plan = OutputPlan.from_run_job_kwargs(
            output=tmp_path / "cart_ao", method="rhf", basis="sto-3g", functional=None
        )
        path = write_qvf(
            tmp_path / "cart_ao",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            ao_data=_ao_entries(2, pure=False),
        )
        report = validate_qvf(path)
        assert report["valid"], f"Validation errors: {report.get('errors', [])}"
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        secs = [s for s in manifest["sections"] if s["kind"] == "basis.ao"]
        assert len(secs) == 6
        assert [tuple(s["ao_metadata"]["cartesian_powers"]) for s in secs] == (
            _cartesian_component_powers(2)
        )

    def test_spherical_archive_omits_the_key_entirely(self, tmp_path):
        mol = _molecule()
        plan = OutputPlan.from_run_job_kwargs(
            output=tmp_path / "sph_ao", method="rhf", basis="sto-3g", functional=None
        )
        path = write_qvf(
            tmp_path / "sph_ao",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            ao_data=_ao_entries(2, pure=True),
        )
        assert validate_qvf(path)["valid"]
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        for s in manifest["sections"]:
            if s["kind"] == "basis.ao":
                assert "cartesian_powers" not in s["ao_metadata"]


@pytest.fixture(scope="module")
def ao_meta_schema() -> dict:
    with open(_SCHEMA_PATH_V1, encoding="utf-8") as f:
        schema = json.load(f)
    return schema["$defs"]["SectionBasisAO"]["properties"]["ao_metadata"]


class TestSchemaContract:
    def test_cartesian_powers_is_declared(self, ao_meta_schema):
        spec = ao_meta_schema["properties"]["cartesian_powers"]
        assert spec["minItems"] == spec["maxItems"] == 3
        assert spec["items"] == {"type": "integer", "minimum": 0}

    def test_the_change_is_additive(self, ao_meta_schema):
        """No ``qvf_version`` bump is allowed, so no required member moved.

        ``qvf-writer/GOVERNANCE.md`` makes a required-member change a
        breaking change.  ``cartesian_powers`` is therefore optional and
        ``angular_momentum`` stays required — element 0 (``l``) is
        meaningful for both shell kinds.
        """
        assert "cartesian_powers" not in ao_meta_schema["required"]
        assert "angular_momentum" in ao_meta_schema["required"]

    def test_angular_momentum_documents_the_cartesian_caveat(self, ao_meta_schema):
        desc = ao_meta_schema["properties"]["angular_momentum"]["description"]
        assert "cartesian_powers" in desc, (
            "the pair is only interpretable as [l, m] when cartesian_powers "
            "is absent; the schema must say so"
        )

    @pytest.mark.parametrize("copy_path", _SCHEMA_COPIES, ids=lambda p: p.parent.name)
    def test_schema_copies_are_in_lock_step(self, copy_path):
        if not copy_path.exists():
            pytest.skip(f"{copy_path} not present (partial checkout)")
        canonical = hashlib.sha256(Path(_SCHEMA_PATH_V1).read_bytes()).hexdigest()
        mirrored = hashlib.sha256(copy_path.read_bytes()).hexdigest()
        assert canonical == mirrored, (
            f"{copy_path} diverged from the canonical schema. Re-sync:\n"
            "  cp python/vibeqc/output/formats/qvf_manifest.schema.json "
            "qvf-writer/spec/qvf_manifest.schema.json\n"
            "  cp qvf-writer/spec/qvf_manifest.schema.json "
            "qvf-writer/python/qvf_manifest.schema.json"
        )
