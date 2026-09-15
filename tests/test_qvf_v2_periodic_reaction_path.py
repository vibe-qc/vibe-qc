"""Periodic `reaction.path` round-trip tests (the `lattice` member).

A periodic reaction path carries a `lattice` binary member (columns = a, b, c,
in bohr) + a `dim` integer in the metadata JSON, so vibe-view can render the
unit cell and wrap atoms across periodic boundaries for slab / NEB paths.

`lattice` is an **optional member of qvf_version 1**. Its *presence* is what
marks a path periodic — never the manifest version. `qvf_version: 2` was
withdrawn by the governance ruling of 2026-07-10 (an optional member is the
additive case and must not bump the version); see qvf-writer/GOVERNANCE.md,
"Version history". Archives stamped 2 by vibe-qc v0.10.0-v0.15.x remain
readable, and the v2 schema is retained as a frozen artifact.

These tests cover:

* a molecular reaction.path ships as qvf_version=1 with no `lattice` member.
* a periodic reaction.path with a shared lattice also ships as qvf_version=1,
  carries `lattice` (shape [3,3]) + `dim` (scalar), and round-trips through
  vibe-view's QVFReader.
* a periodic reaction.path with a per-frame lattice ships `lattice` with shape
  [n_frames, 3, 3] (forward-compat with variable-cell scans).
* a mixed molecular/periodic reaction.path is rejected at write time --
  "raise, don't drop" discipline.
* an archive stamped qvf_version=2 still validates (deprecated read path).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vibeqc.output.formats.qvf import (
    QVF_FORMAT_VERSION,
    QVF_FORMAT_VERSION_V2,
    _SCHEMA_PATH_V2,
    _load_canonical_schema,
    validate_qvf,
    write_qvf,
)
from vibeqc.output.plan import OutputPlan


# ---- Stubs ---------------------------------------------------------------


class _Atom:
    def __init__(self, Z, xyz):
        self.Z = Z
        self.xyz = xyz


def _mol_stub(symbols_xyz):
    """Stub Molecule-shaped object — exposes .atoms (not .unit_cell)."""
    return type(
        "Molecule",
        (),
        {
            "atoms": [_Atom(Z, xyz) for Z, xyz in symbols_xyz],
            "charge": 0,
            "multiplicity": 1,
        },
    )()


def _slab_stub(symbols_xyz, lattice, dim=2):
    """Stub PeriodicSystem-shaped object — exposes .unit_cell + .lattice."""
    return type(
        "PeriodicSystem",
        (),
        {
            "unit_cell": [_Atom(Z, xyz) for Z, xyz in symbols_xyz],
            "lattice": np.asarray(lattice, dtype=float),
            "dim": int(dim),
            "charge": 0,
            "multiplicity": 1,
        },
    )()


def _plan(tmp_path: Path) -> OutputPlan:
    return OutputPlan.from_run_job_kwargs(
        output=tmp_path / "v2rt", method="rhf", basis="sto-3g", functional=None
    )


def _result_stub():
    return type(
        "Result",
        (),
        {"converged": True, "energy": -75.983, "fermi_energy": -0.247},
    )()


def _slab_lattice():
    """Tiny (1×1) slab cell, vacuum along c. Bohr."""
    return [[5.4, 0.0, 0.0], [0.0, 5.4, 0.0], [0.0, 0.0, 20.0]]


def _slab_frames(n: int):
    """n frames of a 2-atom adsorbate moving along the surface normal."""
    L = _slab_lattice()
    frames = []
    for i in range(n):
        z = 2.0 + 0.3 * i  # adsorbate height varies frame to frame
        frames.append(
            _slab_stub(
                [
                    (8, (2.7, 2.7, 1.0)),
                    (1, (2.7, 2.7, z)),
                ],
                L,
                dim=2,
            )
        )
    return frames


def _mol_frames(n: int):
    frames = []
    for i in range(n):
        frames.append(
            _mol_stub(
                [
                    (8, (0.0, 0.0, 0.1173)),
                    (1, (0.0, 1.4315, -0.9314 + 0.05 * i)),
                    (1, (0.0, -1.4315, -0.9314 + 0.05 * i)),
                ]
            )
        )
    return frames


def _waypoints_for(n: int):
    """Reactant + TS at midpoint + product, capped to n's range."""
    mid = max(1, n // 2)
    return [
        {"frame_index": 0, "label": "reactant", "kind": "reactant"},
        {"frame_index": mid, "label": "TS", "kind": "transition_state"},
        {"frame_index": n - 1, "label": "product", "kind": "product"},
    ]


# ---- Tests ---------------------------------------------------------------


class TestMolecularPathStillV1:
    """A molecular reaction.path must still ship as qvf_version=1."""

    def test_molecular_path_is_v1(self, tmp_path):
        frames = _mol_frames(5)
        path = write_qvf(
            tmp_path / "mol_rxn",
            _plan(tmp_path),
            molecule=_mol_stub([(8, (0.0, 0.0, 0.0))]),
            result=_result_stub(),
            method="rhf",
            basis="sto-3g",
            reaction_path={
                "frames": frames,
                "waypoints": _waypoints_for(5),
            },
        )
        report = validate_qvf(path)
        assert report["valid"], report["errors"]
        with zipfile.ZipFile(path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            assert mf["qvf_version"] == QVF_FORMAT_VERSION
            rxn = next(s for s in mf["sections"] if s["kind"] == "reaction.path")
            assert "lattice" not in rxn["members"]
            meta = json.loads(zf.read(rxn["members"]["metadata"]["path"]))
            assert "dim" not in meta
            assert "dim_per_frame" not in meta


class TestPeriodicPathV2:
    """A reaction.path with PeriodicSystem frames ships as v1 + `lattice`.

    (Class name retained for history; the v2 bump it was written for was
    withdrawn on 2026-07-10 — the periodic marker is the `lattice` member.)
    """

    def test_shared_lattice_emits_3x3(self, tmp_path):
        frames = _slab_frames(5)
        path = write_qvf(
            tmp_path / "slab_rxn",
            _plan(tmp_path),
            molecule=_mol_stub([(8, (0.0, 0.0, 0.0))]),
            result=_result_stub(),
            method="rhf",
            basis="sto-3g",
            reaction_path={
                "frames": frames,
                "waypoints": _waypoints_for(5),
                "energies": [-75.5, -75.45, -75.4, -75.5, -75.6],
            },
        )
        report = validate_qvf(path)
        assert report["valid"], report["errors"]
        with zipfile.ZipFile(path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            # Ruling 2026-07-10: periodic paths stay v1; `lattice` marks them.
            assert mf["qvf_version"] == QVF_FORMAT_VERSION
            assert mf["schema_uri"].endswith("/1/manifest.schema.json")
            rxn = next(s for s in mf["sections"] if s["kind"] == "reaction.path")
            lattice_member = rxn["members"]["lattice"]
            assert lattice_member["dtype"] == "float64"
            assert lattice_member["shape"] == [3, 3]
            # Verify lattice content matches the slab cell.
            raw = zf.read(lattice_member["path"])
            L = np.frombuffer(raw, dtype=np.float64).reshape(3, 3)
            np.testing.assert_allclose(L, np.asarray(_slab_lattice()))
            # dim is a scalar in metadata since every frame shares it.
            meta = json.loads(zf.read(rxn["members"]["metadata"]["path"]))
            assert meta["dim"] == 2
            assert "dim_per_frame" not in meta

    def test_per_frame_lattice_emits_stack(self, tmp_path):
        """When frames carry different lattices, the binary lattice
        member ships as [n_frames, 3, 3]."""
        frames = _slab_frames(3)
        # Mutate the third frame's lattice so frames are not all equal.
        L_alt = np.asarray(_slab_lattice(), dtype=float)
        L_alt[2, 2] = 22.0
        frames[2].lattice = L_alt

        path = write_qvf(
            tmp_path / "slab_varcell",
            _plan(tmp_path),
            molecule=_mol_stub([(8, (0.0, 0.0, 0.0))]),
            result=_result_stub(),
            method="rhf",
            basis="sto-3g",
            reaction_path={
                "frames": frames,
                "waypoints": _waypoints_for(3),
            },
        )
        report = validate_qvf(path)
        assert report["valid"], report["errors"]
        with zipfile.ZipFile(path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            assert mf["qvf_version"] == QVF_FORMAT_VERSION
            rxn = next(s for s in mf["sections"] if s["kind"] == "reaction.path")
            lattice_member = rxn["members"]["lattice"]
            assert lattice_member["shape"] == [3, 3, 3]


class TestMixedFramesRejected:
    def test_mixed_molecular_periodic_raises(self, tmp_path):
        frames = _mol_frames(2) + _slab_frames(2)
        with pytest.raises(ValueError, match="mixed"):
            write_qvf(
                tmp_path / "mixed",
                _plan(tmp_path),
                molecule=_mol_stub([(8, (0.0, 0.0, 0.0))]),
                result=_result_stub(),
                method="rhf",
                basis="sto-3g",
                reaction_path={
                    "frames": frames,
                    "waypoints": _waypoints_for(4),
                },
            )


class TestVibeViewReader:
    def test_periodic_path_round_trips_through_reader(self, tmp_path):
        """End-to-end: write a v2 periodic reaction.path, read it back
        through vibe-view's QVFReader, and confirm the new lattice +
        dim fields survive."""
        vibeview_qvf = pytest.importorskip(
            "vibeview.qvf",
            reason="vibe-view QVF reader is not installed in this environment",
        )
        QVFReader = vibeview_qvf.QVFReader

        frames = _slab_frames(4)
        path = write_qvf(
            tmp_path / "v2_reader",
            _plan(tmp_path),
            molecule=_mol_stub([(8, (0.0, 0.0, 0.0))]),
            result=_result_stub(),
            method="rhf",
            basis="sto-3g",
            reaction_path={
                "frames": frames,
                "waypoints": _waypoints_for(4),
            },
        )
        reader = QVFReader(path)
        try:
            assert reader.manifest.qvf_version == QVF_FORMAT_VERSION
            rxn_section_id = next(
                s.id for s in reader.manifest.sections if s.kind == "reaction.path"
            )
            data = reader.read_reaction_path(rxn_section_id)
            assert data.lattice is not None
            assert data.lattice.shape == (3, 3)
            np.testing.assert_allclose(data.lattice, np.asarray(_slab_lattice()))
            assert data.dim == 2
            assert data.dim_per_frame is None
            # Atoms + frames intact from v1 path.
            assert len(data.atoms) == 2
            assert data.coords.shape == (4, 2, 3)
        finally:
            reader.close()


class TestSchemaV2Available:
    def test_v2_schema_loads_and_declares_version_2(self):
        schema = _load_canonical_schema(2)
        assert _SCHEMA_PATH_V2.is_file()
        assert schema["properties"]["qvf_version"]["const"] == 2
        # And the lattice extension is present.
        assert "ReactionPathLattice" in schema["$defs"]
        rxn = schema["$defs"]["SectionReactionPath"]
        assert "lattice" in rxn["properties"]["members"]["properties"]


# ---------------------------------------------------------------------------
# Regression — DOS sections inside a v2 archive.
#
# The v2 manifest schema was missing the dos.total / dos.projected
# section definitions, so a periodic reaction.path (which forces
# qvf_version=2) that also carried DOS failed the write-time validation
# gate. v2 must be a strict superset of v1.
# (Found via examples/vibe_view/showcase_qvf_all_sections.py.)
# ---------------------------------------------------------------------------


class TestV2WithDOS:
    def test_v2_periodic_path_plus_dos_validates(self, tmp_path):
        frames = _slab_frames(3)
        energies = np.linspace(-25.0, 15.0, 64)
        dos = np.exp(-0.5 * (energies / 3.0) ** 2)
        path = write_qvf(
            tmp_path / "v2_dos",
            _plan(tmp_path),
            system=frames[0],
            method="rhf",
            basis="sto-3g",
            reaction_path={"frames": frames, "waypoints": _waypoints_for(3)},
            dos_data={"energies": energies, "dos": dos, "n_spin": 1},
            pdos_data={
                "energies": energies,
                "projections": np.vstack([dos, dos]),
                "n_spin": 1,
                "channels": [
                    {"atom_index": 0, "symbol": "O", "l": 0, "label": "O 2s"},
                    {"atom_index": 0, "symbol": "O", "l": 1, "label": "O 2p"},
                ],
            },
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
        assert manifest["qvf_version"] == QVF_FORMAT_VERSION
        assert validate_qvf(path)["valid"]
        kinds = {s["kind"] for s in manifest["sections"]}
        assert {"reaction.path", "dos.total", "dos.projected"} <= kinds


# ---------------------------------------------------------------------------
# Regression — v2 archives carrying *any* other v1 content.
#
# The DOS case above was patched by hand-copying two section definitions
# into what was then a forked v2 schema. That fixed the symptom, not the
# cause: v2 was still a stale snapshot and stayed silently missing ten
# other section kinds and four root blocks. `bond_orders` (a section, so
# gated by the closed `Section.oneOf`) plus root `thermochemistry` (gated
# by the root's `additionalProperties: false`) exercise both rejection
# paths at once, on the archive rather than on the schema.
#
# v2 is now derived from v1 (`_derive_v2_schema`), so this holds for every
# v1 section by construction; `tests/test_qvf_v2_schema_drift.py` pins that.
# ---------------------------------------------------------------------------


class TestV2StrictSupersetOfV1:
    def test_v2_periodic_path_plus_bond_orders_plus_thermochemistry(self, tmp_path):
        import jsonschema

        frames = _slab_frames(3)
        thermo = {
            "zpve_eh": 0.0213,
            "enthalpy_eh": -75.9012,
            "entropy_cal_mol_k": 45.21,
            "gibbs_free_energy_eh": -75.9327,
            "temperature_k": 298.15,
            "pressure_atm": 1.0,
        }
        path = write_qvf(
            tmp_path / "v2_bo_thermo",
            _plan(tmp_path),
            system=frames[0],
            method="rhf",
            basis="sto-3g",
            reaction_path={"frames": frames, "waypoints": _waypoints_for(3)},
            bond_orders_data={
                "method": "mayer",
                "pairs": [
                    {
                        "i": 0,
                        "j": 1,
                        "order": 0.93,
                        "distance_ang": 0.97,
                        "symbol_i": "O",
                        "symbol_j": "H",
                    }
                ],
            },
            thermochemistry_data=thermo,
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))

        assert manifest["qvf_version"] == QVF_FORMAT_VERSION
        kinds = {s["kind"] for s in manifest["sections"]}
        assert {"reaction.path", "bond_orders"} <= kinds
        assert manifest["thermochemistry"] == thermo

        # A periodic reaction path carrying *any* other section + root metadata
        # validates under v1. (This is the bug the v2 fork caused: the forked
        # schema was missing bond_orders / thermochemistry, so such an archive
        # raised at write time.)
        jsonschema.Draft202012Validator(_load_canonical_schema(1)).validate(manifest)

        # Deprecated read path: the retained v2 schema still accepts the same
        # manifest stamped qvf_version=2, as vibe-qc v0.10.0-v0.15.x wrote it.
        # v2 is now v1 modulo this const, so it is trivially a superset.
        legacy = dict(manifest, qvf_version=QVF_FORMAT_VERSION_V2)
        jsonschema.Draft202012Validator(_load_canonical_schema(2)).validate(legacy)
        assert validate_qvf(path)["valid"]

        # And the reaction.path really is the periodic (v2) flavour.
        rxn = next(s for s in manifest["sections"] if s["kind"] == "reaction.path")
        assert "lattice" in rxn["members"]


class TestLegacyV2ArchivesStillRead:
    """The `qvf_version: 2` deprecation promise (governance ruling 2026-07-10).

    Producers must not emit 2, but archives stamped 2 by vibe-qc
    v0.10.0-v0.15.x exist in the wild and must keep validating. This rebuilds
    such an archive byte-for-byte apart from the manifest's version stamp and
    checks that `validate_qvf` still accepts it.
    """

    def test_validate_qvf_accepts_a_qvf_version_2_archive(self, tmp_path):
        frames = _slab_frames(3)
        path = write_qvf(
            tmp_path / "modern",
            _plan(tmp_path),
            system=frames[0],
            method="rhf",
            basis="sto-3g",
            reaction_path={"frames": frames, "waypoints": _waypoints_for(3)},
        )
        # Sanity: today's writer emits v1 with a `lattice` member.
        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            names = zf.namelist()
            payloads = {n: zf.read(n) for n in names}
        assert manifest["qvf_version"] == QVF_FORMAT_VERSION
        rxn = next(s for s in manifest["sections"] if s["kind"] == "reaction.path")
        assert "lattice" in rxn["members"]

        # Re-stamp as a legacy v2 archive (manifest.json carries no checksum of
        # itself, so only the version + schema_uri change).
        legacy_manifest = dict(
            manifest,
            qvf_version=QVF_FORMAT_VERSION_V2,
            schema_uri="https://vibe-qc.org/spec/qvf/2/manifest.schema.json",
        )
        legacy_path = tmp_path / "legacy_v2.qvf"
        with zipfile.ZipFile(legacy_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in names:
                if name == "manifest.json":
                    zf.writestr(name, json.dumps(legacy_manifest, indent=2))
                else:
                    zf.writestr(name, payloads[name])

        report = validate_qvf(legacy_path)
        assert report["valid"], report["errors"]
