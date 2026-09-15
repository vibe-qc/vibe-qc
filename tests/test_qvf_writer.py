"""Tests for the QVF writer — consumer-aligned manifest shape.

All tests verify the writer produces archives compatible with the
vibe-view consumer (vibe-view/src/vibeview/qvf.py).
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from vibeqc.output.formats.qvf import (
    QVF_FORMAT_VERSION,
    _ao_isovalue_default,
    _ao_label,
    _ao_section_id,
    _BOHR_TO_ANGSTROM,
    _l_to_shell_type,
    _primitive_norm,
    _sha256_hex,
    qvf_bloch_wf_data,
    qvf_natural_wf_data,
    qvf_wf_data,
    scf_history_from_result,
    validate_qvf,
    write_qvf,
)
from vibeqc.output.plan import OutputPlan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plan(tmp_path: Path, **overrides) -> OutputPlan:
    kwargs = dict(
        output=tmp_path / "test_qvf", method="rhf", basis="sto-3g", functional=None
    )
    kwargs.update(overrides)
    return OutputPlan.from_run_job_kwargs(**kwargs)


def _read_manifest(zf: zipfile.ZipFile) -> dict:
    return json.loads(zf.read("manifest.json").decode("utf-8"))


def _stub_molecule():
    class _Atom:
        def __init__(self, Z, xyz):
            self.Z = Z
            self.xyz = xyz

    return type(
        "Molecule",
        (),
        {
            "atoms": [
                _Atom(8, (0.0, 0.0, 0.1173)),
                _Atom(1, (0.0, 1.4315, -0.9314)),
                _Atom(1, (0.0, -1.4315, -0.9314)),
            ],
            "charge": 0,
            "multiplicity": 1,
        },
    )()


def _stub_trace_step(it, energy, delta_e):
    return type(
        "Step",
        (),
        {"iter": it, "energy": energy, "delta_e": delta_e, "grad_norm": 0.0},
    )()


def _stub_result():
    return type(
        "Result",
        (),
        {
            "converged": True,
            "energy": -75.983,
            "fermi_energy": -0.247,
            "n_iter": 3,
            "scf_trace": (
                _stub_trace_step(1, -75.0, -75.0),
                _stub_trace_step(2, -75.9, -0.9),
                _stub_trace_step(3, -75.983, -0.083),
            ),
        },
    )()


def _stub_population():
    return type(
        "Pop",
        (),
        {
            "mulliken_atoms": [
                (0, "O", 8.0, -0.65),
                (1, "H", 1.0, 0.325),
                (2, "H", 1.0, 0.325),
            ],
            "loewdin_atoms": [
                (0, "O", 8.0, -0.58),
                (1, "H", 1.0, 0.29),
                (2, "H", 1.0, 0.29),
            ],
            "hirshfeld_atoms": [
                (0, "O", 8.0, -0.45),
                (1, "H", 1.0, 0.225),
                (2, "H", 1.0, 0.225),
            ],
            "mayer_bonds": [],
            "dipole": None,
            "errors": {},
        },
    )()


def _stub_hessian():
    n = 9
    freqs = np.array([-5.0, 0, 0, 0, 0, 0, 1200, 3500, 3800], dtype=np.float64)
    return type(
        "Hess",
        (),
        {
            "frequencies_cm1": freqs,
            "normal_modes": np.eye(n, dtype=np.float64),
            # H2O masses (amu), matching _stub_molecule's O,H,H ordering.
            "masses_amu": np.array([15.999, 1.008, 1.008], dtype=np.float64),
        },
    )()


def _stub_bands():
    from dataclasses import dataclass

    @dataclass
    class KPath:
        kpoints_cart: np.ndarray
        kpoints_frac: np.ndarray
        distances: np.ndarray
        labels: list

    kp = KPath(
        kpoints_cart=np.zeros((20, 3)),
        kpoints_frac=np.column_stack(
            [np.linspace(0, 0.5, 20), np.zeros(20), np.zeros(20)]
        ),
        distances=np.linspace(0, 0.5, 20).astype(np.float64),
        labels=[(0.0, "Γ"), (0.25, "X"), (0.5, "M")],
    )
    energies = np.random.randn(20, 5).astype(np.float64) * 0.1 - 0.5
    return type("BS", (), {"kpath": kp, "energies": energies, "e_fermi": -0.25})()


# ---------------------------------------------------------------------------
# Unit tests — per section kind
# ---------------------------------------------------------------------------


class TestStructureSection:
    def test_molecular_structure(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "h2o_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
        )
        assert path.suffix == ".qvf"
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            assert "structure/structure.json" in names
            manifest = _read_manifest(zf)
            s = [s for s in manifest["sections"] if s["kind"] == "structure"][0]
            assert s["id"] == "structure"
            assert "structure" in s["members"]
            sm = s["members"]["structure"]
            assert sm["format"] == "json"
            struct = json.loads(zf.read(sm["path"]))
            assert len(struct["atoms"]) == 3
            assert struct["atoms"][0]["symbol"] == "O"
            assert struct["pbc"] == [False, False, False]
            assert _sha256_hex(zf.read(sm["path"])) == sm["sha256"]

    def test_periodic_structure(self, tmp_path):
        atoms = list(_stub_molecule().atoms)
        lattice = np.array([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]], dtype=np.float64)
        system = type(
            "Sys", (), {"atoms": atoms, "lattice": lattice, "dimensionality": 3}
        )()
        plan = _plan(tmp_path, job_kind="periodic_scf")
        path = write_qvf(
            tmp_path / "per_qvf",
            plan,
            system=system,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = [s for s in _read_manifest(zf)["sections"] if s["kind"] == "structure"][
                0
            ]
            struct = json.loads(zf.read(s["members"]["structure"]["path"]))
            assert struct["pbc"] == [True, True, True]
            assert struct["dimensionality"] == 3
            assert "lattice_vectors" in struct
            np.testing.assert_allclose(
                struct["lattice_vectors"],
                (lattice * _BOHR_TO_ANGSTROM).T,
            )

    def test_periodic_structure_uses_unit_cell(self, tmp_path):
        """Real bound PeriodicSystem exposes ``unit_cell`` (not
        ``atoms``) plus ``dim`` ∈ {1,2,3}. Previously the writer
        silently emitted no structure section because it looked for
        ``.atoms`` only, leaving run_periodic_job(output_qvf=True)
        archives without geometry while still reporting success."""
        atoms = list(_stub_molecule().atoms)
        lattice = np.array([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]], dtype=np.float64)
        system = type(
            "PeriodicSys",
            (),
            {"unit_cell": atoms, "lattice": lattice, "dim": 3},
        )()
        plan = _plan(tmp_path, job_kind="periodic_scf")
        path = write_qvf(
            tmp_path / "per_uc_qvf",
            plan,
            system=system,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = next(
                s for s in _read_manifest(zf)["sections"] if s["kind"] == "structure"
            )
            struct = json.loads(zf.read(s["members"]["structure"]["path"]))
            assert len(struct["atoms"]) == 3
            assert struct["pbc"] == [True, True, True]
            assert struct["dimensionality"] == 3
            assert "lattice_vectors" in struct

    def test_periodic_structure_lattice_vectors_are_rows_angstrom(self, tmp_path):
        """QVF structure lattices are Angstrom row vectors.

        ``PeriodicSystem.lattice`` is bohr column vectors, so this skewed
        lattice catches both missing transpose and missing unit conversion.
        """
        atoms = list(_stub_molecule().atoms)
        lattice = np.array(
            [[2.0, 3.0, 5.0], [7.0, 11.0, 13.0], [17.0, 19.0, 23.0]],
            dtype=np.float64,
        )
        system = type(
            "PeriodicSys",
            (),
            {"unit_cell": atoms, "lattice": lattice, "dim": 1},
        )()
        plan = _plan(tmp_path, job_kind="periodic_scf")
        path = write_qvf(
            tmp_path / "per_skew_qvf",
            plan,
            system=system,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = next(
                s for s in _read_manifest(zf)["sections"] if s["kind"] == "structure"
            )
            struct = json.loads(zf.read(s["members"]["structure"]["path"]))
            assert struct["pbc"] == [True, False, False]
            assert struct["dimensionality"] == 1
            np.testing.assert_allclose(
                struct["lattice_vectors"],
                (lattice * _BOHR_TO_ANGSTROM).T,
            )

    def test_periodic_structure_accepts_bvk_lattice_override(self, tmp_path):
        """CCM callers can emit the BvK torus cell in ``structure``."""
        atoms = list(_stub_molecule().atoms)
        primitive = np.diag([4.0, 5.0, 6.0]).astype(np.float64)
        bvk = primitive * np.array([2.0, 3.0, 1.0], dtype=np.float64)
        system = type(
            "PeriodicSys",
            (),
            {"unit_cell": atoms, "lattice": primitive, "dim": 3},
        )()
        plan = _plan(tmp_path, job_kind="periodic_scf")
        path = write_qvf(
            tmp_path / "per_bvk_qvf",
            plan,
            system=system,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            structure_lattice_bohr=bvk,
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = next(
                s for s in _read_manifest(zf)["sections"] if s["kind"] == "structure"
            )
            struct = json.loads(zf.read(s["members"]["structure"]["path"]))
            np.testing.assert_allclose(
                struct["lattice_vectors"],
                (bvk * _BOHR_TO_ANGSTROM).T,
            )

    def test_periodic_structure_2d_partial_pbc(self, tmp_path):
        """For dim=2 (slab) only the in-plane axes are periodic."""
        atoms = list(_stub_molecule().atoms)
        lattice = np.array([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]], dtype=np.float64)
        system = type(
            "PeriodicSys2D",
            (),
            {"unit_cell": atoms, "lattice": lattice, "dim": 2},
        )()
        plan = _plan(tmp_path, job_kind="periodic_scf")
        path = write_qvf(
            tmp_path / "per_2d_qvf",
            plan,
            system=system,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = next(
                s for s in _read_manifest(zf)["sections"] if s["kind"] == "structure"
            )
            struct = json.loads(zf.read(s["members"]["structure"]["path"]))
            assert struct["pbc"] == [True, True, False]
            assert struct["dimensionality"] == 2


class TestVolumeDensitySection:
    def test_single_density(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        data = np.random.randn(10, 10, 10).astype(np.float32)
        path = write_qvf(
            tmp_path / "dens_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            volume_data={"Electron density": (data, np.zeros(3), np.eye(3) * 0.5)},
        )
        with zipfile.ZipFile(path, "r") as zf:
            secs = [
                s
                for s in _read_manifest(zf)["sections"]
                if s["kind"] == "volume.density"
            ]
            assert len(secs) == 1
            s = secs[0]
            dm = s["members"]["data"]
            assert dm["dtype"] == "float32"
            assert dm["format"] == "binary"
            # Grid is now a JSON member
            gm = s["members"]["grid"]
            assert gm["format"] == "json"
            grid = json.loads(zf.read(gm["path"]))
            assert grid["shape"] == [10, 10, 10]
            assert "voxel_vectors" in grid
            raw = zf.read(dm["path"])
            vol = np.frombuffer(raw, dtype=np.float32).reshape(10, 10, 10)
            np.testing.assert_allclose(vol.ravel(), data.ravel())

    def test_float64_volume(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        data = np.random.randn(6, 6, 6).astype(np.float64)
        path = write_qvf(
            tmp_path / "dens64_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            volume_dtype="float64",
            volume_data={
                "Density": (data, np.array([-1.0, -1.0, -1.0]), np.eye(3) * 0.3)
            },
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = [
                s
                for s in _read_manifest(zf)["sections"]
                if s["kind"] == "volume.density"
            ][0]
            assert s["members"]["data"]["dtype"] == "float64"

    def test_grid_voxel_vectors_are_per_voxel(self, tmp_path):
        """voxel_vectors should be per-voxel step vectors, not full span."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        spacing = 0.3
        n_voxels = 10
        data = np.random.randn(n_voxels, n_voxels, n_voxels).astype(np.float32)
        # span should be per-voxel: diag(spacing)
        span_arg = np.eye(3) * spacing
        origin_arg = np.array([-1.5, -1.5, -1.5])
        path = write_qvf(
            tmp_path / "pervox_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            volume_data={"Density": (data, origin_arg, span_arg)},
        )
        with zipfile.ZipFile(path, "r") as zf:
            m = _read_manifest(zf)
            s = [s for s in m["sections"] if s["kind"] == "volume.density"][0]
            gpath = s["members"]["grid"]["path"]
            grid = json.loads(zf.read(gpath).decode("utf-8"))
            # Origin round-trips
            assert grid["origin"] == [-1.5, -1.5, -1.5]
            # voxel_vectors should be per-voxel step
            assert grid["voxel_vectors"][0][0] == pytest.approx(spacing)
            assert grid["voxel_vectors"][1][1] == pytest.approx(spacing)
            assert grid["voxel_vectors"][2][2] == pytest.approx(spacing)
            # shape matches data
            assert grid["shape"] == [n_voxels, n_voxels, n_voxels]


class TestVolumeGenericSection:
    """Escape-hatch volumetric kind for scalar fields that don't fit
    density/orbital/spin/elf/difference (e.g. exchange-correlation
    potential, a user-supplied custom field). Structurally identical
    to volume.density."""

    def test_generic_round_trip(self, tmp_path):
        plan = _plan(tmp_path)
        data = np.random.randn(6, 6, 6).astype(np.float32)
        path = write_qvf(
            tmp_path / "gen_qvf",
            plan,
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            generic_volume_data={
                "custom_field": (data, np.zeros(3), np.eye(3) * 0.3),
            },
        )
        with zipfile.ZipFile(path, "r") as zf:
            sec = next(
                s
                for s in _read_manifest(zf)["sections"]
                if s["kind"] == "volume.generic"
            )
            assert sec["label"] == "custom_field"
            assert set(sec["members"]) == {"data", "grid"}
            assert sec["members"]["data"]["dtype"] == "float32"
            assert sec["members"]["data"]["shape"] == [6, 6, 6]
            # Round-trip the bytes verbatim
            raw = zf.read(sec["members"]["data"]["path"])
            roundtripped = np.frombuffer(raw, dtype=np.float32).reshape(6, 6, 6)
            np.testing.assert_allclose(roundtripped, data)


class TestVolumeOrbitalSection:
    def test_single_mo(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        data = np.random.randn(8, 8, 8).astype(np.float32)
        mo = [
            {
                "label": "HOMO",
                "data": data,
                "origin": np.zeros(3),
                "span": np.eye(3) * 0.4,
                "band_index": 4,
                "energy_eh": -0.5,
                "component": "real",
            }
        ]
        path = write_qvf(
            tmp_path / "mo_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            mo_data=mo,
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = [
                s
                for s in _read_manifest(zf)["sections"]
                if s["kind"] == "volume.orbital"
            ][0]
            assert s["label"] == "HOMO"
            assert s.get("component") == "real"
            assert "grid" in s["members"]


class TestAtomPropertiesSection:
    def test_population(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "props_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            population_summary=_stub_population(),
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = [
                s
                for s in _read_manifest(zf)["sections"]
                if s["kind"] == "atom_properties"
            ][0]
            m = s["members"]
            assert "mulliken_charge" in m
            assert "loewdin_charge" in m
            assert "hirshfeld_charge" in m
            assert m["mulliken_charge"]["shape"] == [3]
            chg = np.frombuffer(zf.read(m["mulliken_charge"]["path"]), dtype=np.float64)
            np.testing.assert_allclose(chg, [-0.65, 0.325, 0.325])
            hirshfeld = np.frombuffer(
                zf.read(m["hirshfeld_charge"]["path"]),
                dtype=np.float64,
            )
            np.testing.assert_allclose(hirshfeld, [-0.45, 0.225, 0.225])


class TestTrajectorySection:
    def test_trajectory(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        frames = [mol, mol, mol]
        path = write_qvf(
            tmp_path / "traj_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            trajectory_frames=frames,
            trajectory_energies=[-75.5, -75.7, -75.98],
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = [
                s for s in _read_manifest(zf)["sections"] if s["kind"] == "trajectory"
            ][0]
            m = s["members"]
            assert "metadata" in m
            assert "coords" in m
            assert m["coords"]["format"] == "binary"
            meta = json.loads(zf.read(m["metadata"]["path"]))
            assert "atoms" in meta
            assert meta["energies"] == [-75.5, -75.7, -75.98]


class TestVibrationsSection:
    def test_vibrations(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "vib_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            hessian_result=_stub_hessian(),
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = [
                s for s in _read_manifest(zf)["sections"] if s["kind"] == "vibrations"
            ][0]
            m = s["members"]
            assert "metadata" in m
            assert "displacements" in m
            meta = json.loads(zf.read(m["metadata"]["path"]))
            assert "frequencies" in meta
            assert "atoms" in meta
            # A5-01/A5-05: real equilibrium geometry + atomic numbers, not
            # [0,0,0] / Z=0 (which made the viewer animate at the origin).
            atoms = meta["atoms"]
            assert [a["atomic_number"] for a in atoms] == [8, 1, 1]
            positions = np.array([a["position"] for a in atoms])
            assert not np.allclose(positions, 0.0)
            # O at (0,0,0.1173) bohr → ~0.0621 Å in z.
            assert positions[0][2] == pytest.approx(0.1173 * 0.529177210903, abs=1e-6)

    def test_vibrations_displacements_are_un_mass_weighted(self, tmp_path):
        """A5-02: a stretch mode where O and H carry equal *mass-weighted*
        amplitude must, after un-mass-weighting, show H moving ~sqrt(M_O/M_H)
        ≈ 4x more than O."""
        import numpy as _np

        mol = _stub_molecule()  # O, H, H
        # Mass-weighted mode: O_z and H1_z move with equal weight.
        modes = _np.zeros((9, 9))
        modes[2, 6] = 1.0 / _np.sqrt(2.0)  # O  z
        modes[5, 6] = 1.0 / _np.sqrt(2.0)  # H1 z
        hess = type(
            "H",
            (),
            {
                "frequencies_cm1": _np.array(
                    [0, 0, 0, 0, 0, 0, 1200, 0, 0], dtype=float
                ),
                "normal_modes": modes,
                "masses_amu": _np.array([15.999, 1.008, 1.008]),
            },
        )()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "vmw",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            hessian_result=hess,
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = [
                s for s in _read_manifest(zf)["sections"] if s["kind"] == "vibrations"
            ][0]
            dm = s["members"]["displacements"]
            disp = _np.frombuffer(zf.read(dm["path"]), dtype=dm["dtype"]).reshape(
                dm["shape"]
            )
        mode6 = disp[6]  # (n_atoms, 3)
        o_amp = _np.linalg.norm(mode6[0])
        h_amp = _np.linalg.norm(mode6[1])
        ratio = h_amp / o_amp
        assert ratio == pytest.approx(_np.sqrt(15.999 / 1.008), rel=0.02)


class TestWavefunctionKPoint:
    """`wavefunction.gto` records the k-point for periodic (Γ-point) systems
    and omits it for molecular ones (QVF spec § 4.6 — periodic now allowed)."""

    class _Shell:
        l = 0
        pure = True
        atom_index = 0
        exponents = [0.5]
        coefficients = [1.0]

    class _Basis:
        def shells(self):
            return [TestWavefunctionKPoint._Shell()]

    class _Result:
        mo_coeffs = np.array([[1.0]])
        mo_energies = np.array([-0.5])

    class _Mol:
        def n_electrons(self):
            return 2

    def test_gamma_point_recorded_for_periodic(self):
        wf = qvf_wf_data(
            self._Result(), self._Basis(), self._Mol(), k_point=[0.0, 0.0, 0.0]
        )
        assert wf["mo_metadata"]["k_point"] == [0.0, 0.0, 0.0]

    def test_molecular_omits_k_point(self):
        wf = qvf_wf_data(self._Result(), self._Basis(), self._Mol())
        assert "k_point" not in wf["mo_metadata"]
        assert wf["mo_metadata"]["occupation_semantics"] == "electron_occupation"

    def test_natural_orbitals_declare_electron_occupation_semantics(self):
        wf = qvf_natural_wf_data(
            np.array([[1.0]]),
            np.array([2.0]),
            self._Basis(),
            self._Mol(),
        )
        assert wf["mo_metadata"]["occupation_semantics"] == "electron_occupation"

    def test_restricted_result_occupations_are_preserved(self):
        class _FractionalResult:
            mo_coeffs = np.eye(2)
            mo_energies = np.array([-0.5, 0.1])
            occupations = np.array([1.75, 0.25])

        wf = qvf_wf_data(_FractionalResult(), self._Basis(), self._Mol())
        assert wf["mo_metadata"]["occupations"] == pytest.approx([1.75, 0.25])

    def test_restricted_empty_occupation_sentinel_uses_aufbau_default(self):
        class _LegacyResult:
            mo_coeffs = np.eye(2)
            mo_energies = np.array([-0.5, 0.1])
            occupations = np.empty(0)

        wf = qvf_wf_data(_LegacyResult(), self._Basis(), self._Mol())
        assert wf["mo_metadata"]["occupations"] == pytest.approx([2.0, 0.0])

    def test_restricted_default_uses_effective_ecp_electron_count(self):
        class _ECPResult:
            mo_coeffs = np.array([[1.0, 0.0, 0.0, 0.0]])
            mo_energies = np.array([-1.0, -0.5, 0.1, 0.2])
            ecp_total_ncore = 2

        class _PhysicalSixElectronMol:
            def n_electrons(self):
                return 6

        wf = qvf_wf_data(
            _ECPResult(), self._Basis(), _PhysicalSixElectronMol()
        )
        assert wf["mo_metadata"]["occupations"] == pytest.approx(
            [2.0, 2.0, 0.0, 0.0]
        )

    def test_unrestricted_result_occupations_are_preserved(self):
        class _FractionalResult:
            mo_coeffs_alpha = np.eye(2)
            mo_coeffs_beta = np.eye(2)
            mo_energies_alpha = np.array([-0.5, 0.1])
            mo_energies_beta = np.array([-0.4, 0.2])
            occupations_alpha = np.array([0.8, 0.2])
            occupations_beta = np.array([0.7, 0.3])

        wf = qvf_wf_data(_FractionalResult(), self._Basis(), self._Mol())
        assert wf["mo_metadata"]["alpha"]["occupations"] == pytest.approx(
            [0.8, 0.2]
        )
        assert wf["mo_metadata"]["beta"]["occupations"] == pytest.approx(
            [0.7, 0.3]
        )

    def test_unrestricted_empty_occupation_sentinels_use_aufbau_defaults(self):
        class _LegacyResult:
            mo_coeffs_alpha = np.eye(2)
            mo_coeffs_beta = np.eye(2)
            mo_energies_alpha = np.array([-0.5, 0.1])
            mo_energies_beta = np.array([-0.4, 0.2])
            occupations_alpha = np.empty(0)
            occupations_beta = np.empty(0)

        wf = qvf_wf_data(_LegacyResult(), self._Basis(), self._Mol())
        assert wf["mo_metadata"]["alpha"]["occupations"] == pytest.approx(
            [1.0, 0.0]
        )
        assert wf["mo_metadata"]["beta"]["occupations"] == pytest.approx(
            [1.0, 0.0]
        )

    @pytest.mark.parametrize(
        "attr",
        ["occupations", "occupations_alpha", "occupations_beta"],
    )
    def test_nonempty_malformed_occupation_size_is_rejected(self, attr):
        if attr == "occupations":
            result = type(
                "_MalformedRestrictedResult",
                (),
                {
                    "mo_coeffs": np.eye(2),
                    "mo_energies": np.array([-0.5, 0.1]),
                    attr: np.array([2.0]),
                },
            )()
        else:
            result = type(
                "_MalformedUnrestrictedResult",
                (),
                {
                    "mo_coeffs_alpha": np.eye(2),
                    "mo_coeffs_beta": np.eye(2),
                    "mo_energies_alpha": np.array([-0.5, 0.1]),
                    "mo_energies_beta": np.array([-0.4, 0.2]),
                    "occupations_alpha": np.array([1.0, 0.0]),
                    "occupations_beta": np.array([1.0, 0.0]),
                    attr: np.array([1.0]),
                },
            )()

        with pytest.raises(ValueError, match=rf"{attr} has 1 entries; expected 2"):
            qvf_wf_data(result, self._Basis(), self._Mol())

    def test_complex_typed_gamma_coefficients_are_projected_quietly(self):
        import warnings

        class _ComplexResult:
            mo_coeffs = np.array([[1.0 + 1.0e-12j]])
            mo_energies = np.array([-0.5])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wf = qvf_wf_data(
                _ComplexResult(),
                self._Basis(),
                self._Mol(),
                k_point=[0.0, 0.0, 0.0],
            )

        assert not any(
            "ComplexWarning" in warning.category.__name__ for warning in caught
        )
        assert not np.iscomplexobj(wf["mo_coefficients"])
        np.testing.assert_allclose(wf["mo_coefficients"], [[1.0]])

    def test_complex_bloch_coefficients_are_stored_as_real_imag_pairs(self):
        class _ComplexResult:
            mo_coeffs = np.array([[1.0 + 0.25j]])
            mo_energies = np.array([-0.5])

        wf = qvf_wf_data(
            _ComplexResult(),
            self._Basis(),
            self._Mol(),
            k_point=[0.25, 0.0, 0.0],
        )

        assert wf["mo_metadata"]["k_point"] == [0.25, 0.0, 0.0]
        assert wf["mo_metadata"]["coefficient_encoding"] == (
            "complex_split_last_axis"
        )
        assert wf["mo_metadata"]["coefficient_components"] == ["real", "imag"]
        assert wf["mo_coefficients"].shape == (1, 1, 2)
        np.testing.assert_allclose(wf["mo_coefficients"][0, 0], [1.0, 0.25])

    def test_complex_bloch_coefficients_validate_in_archive(self, tmp_path):
        class _ComplexResult:
            mo_coeffs = np.array([[1.0 + 0.25j]])
            mo_energies = np.array([-0.5])

        wf = qvf_wf_data(
            _ComplexResult(),
            self._Basis(),
            self._Mol(),
            k_point=[0.25, 0.0, 0.0],
        )
        path = write_qvf(
            tmp_path / "complex_bloch_wf",
            _plan(tmp_path, output=tmp_path / "complex_bloch_wf"),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            wf_data=wf,
        )

        report = validate_qvf(path)
        assert report["valid"], report["errors"]
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            section = next(
                s for s in manifest["sections"] if s["kind"] == "wavefunction.gto"
            )
            coeff_member = section["members"]["mo_coefficients"]
            assert coeff_member["shape"] == [1, 1, 2]
            raw = zf.read(coeff_member["path"])
            coeff = np.frombuffer(raw, dtype=np.float64).reshape(1, 1, 2)
            metadata = json.loads(
                zf.read(section["members"]["mo_metadata"]["path"]).decode("utf-8")
            )
        np.testing.assert_allclose(coeff[0, 0], [1.0, 0.25])
        assert metadata["coefficient_encoding"] == "complex_split_last_axis"

    def test_all_k_bloch_restart_payload_validates_in_archive(self, tmp_path):
        class _Shell0:
            l = 0
            pure = True
            atom_index = 0
            exponents = [0.5]
            coefficients = [1.0]

        class _Shell1(_Shell0):
            atom_index = 1

        class _Basis2:
            def shells(self):
                return [_Shell0(), _Shell1()]

        class _BlochResult:
            mo_coeffs = [
                np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex),
                np.array(
                    [
                        [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
                        [1.0j / np.sqrt(2.0), -1.0j / np.sqrt(2.0)],
                    ],
                    dtype=complex,
                ),
            ]
            mo_energies = [np.array([-0.5, 0.1]), np.array([-0.4, 0.2])]
            occupations = [np.array([2.0, 0.0]), np.array([1.5, 0.5])]

        bloch = qvf_bloch_wf_data(
            _BlochResult(),
            _Basis2(),
            k_points=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        )
        path = write_qvf(
            tmp_path / "all_k_bloch_wf",
            _plan(tmp_path, output=tmp_path / "all_k_bloch_wf"),
            system=type(
                "Sys",
                (),
                {
                    "unit_cell": _stub_molecule().atoms[:2],
                    "lattice": np.eye(3) * 5.0,
                    "dim": 3,
                },
            )(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            bloch_wf_data=bloch,
        )

        report = validate_qvf(path)
        assert report["valid"], report["errors"]
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            kinds = {section["kind"] for section in manifest["sections"]}
            assert "wavefunction.gto" not in kinds
            section = next(
                section
                for section in manifest["sections"]
                if section["kind"] == "x_vibeqc.bloch_wavefunction"
            )
            metadata = json.loads(
                zf.read(section["members"]["mo_metadata"]["path"]).decode("utf-8")
            )
            coeff_member = section["members"]["mo_coefficients_k1"]
            coeff = np.frombuffer(
                zf.read(coeff_member["path"]), dtype=np.float64
            ).reshape(coeff_member["shape"])

        assert section["id"] == "wf_bloch_kpoints"
        assert metadata["n_kpoints"] == 2
        assert metadata["coefficient_encoding"] == "complex_split_last_axis"
        assert coeff.shape == (2, 2, 2)
        np.testing.assert_allclose(
            coeff[0, :, :],
            [[1.0 / np.sqrt(2.0), 0.0], [0.0, 1.0 / np.sqrt(2.0)]],
        )

    def test_single_k_bloch_empty_occupation_sentinel_uses_aufbau_default(self):
        class _Shell0:
            l = 0
            pure = True
            atom_index = 0
            exponents = [0.5]
            coefficients = [1.0]

        class _Shell1(_Shell0):
            atom_index = 1

        class _Basis2:
            def shells(self):
                return [_Shell0(), _Shell1()]

        class _LegacyResult:
            mo_coeffs = [np.eye(2)]
            mo_energies = [np.array([-0.5, 0.1])]
            occupations = [np.empty(0)]

        class _Mol:
            def n_electrons(self):
                return 2

        bloch = qvf_bloch_wf_data(
            _LegacyResult(),
            _Basis2(),
            _Mol(),
            k_points=[[0.0, 0.0, 0.0]],
        )

        np.testing.assert_allclose(bloch["occupations"][0], [2.0, 0.0])


class TestWavefunctionBasisNorm:
    """QVF spec § 4.6: basis coefficients apply to *normalised* primitive
    Gaussians. ``qvf_wf_data`` must divide libint's stored coefficients
    (which already include the primitive norm) by ``_primitive_norm``."""

    # ---------- s-shell helpers ----------
    @staticmethod
    def _make_shell(l, exponents, coefficients, atom_index=0, pure=True):
        return type(
            "Shell",
            (),
            {
                "l": l,
                "pure": pure,
                "atom_index": atom_index,
                "exponents": exponents,
                "coefficients": coefficients,
            },
        )()

    @staticmethod
    def _result(mo_coeffs=None, mo_energies=None):
        return type(
            "Result",
            (),
            {
                "mo_coeffs": np.array(mo_coeffs or [[1.0]]),
                "mo_energies": np.array(mo_energies or [-0.5]),
            },
        )()

    @staticmethod
    def _mol(n_electrons=2):
        class M:
            def n_electrons(self):
                return n_electrons

        return M()

    def _basis(self, shells):
        class B:
            def shells(s):
                return shells

        return B()

    def test_coefficients_divided_by_primitive_norm(self):
        """Output coefficients = libint coefficients / N(α, l)."""
        exps = [0.5, 2.0]
        coeffs = [1.0, 2.0]
        sh = self._make_shell(l=0, exponents=exps, coefficients=coeffs)
        wf = qvf_wf_data(self._result(), self._basis([sh]), self._mol())
        out_shell = wf["basis"][0]
        for a, c_in, c_out in zip(exps, coeffs, out_shell["coefficients"]):
            expected = c_in / _primitive_norm(a, 0)
            assert c_out == pytest.approx(expected, rel=1e-14)

    def test_roundtrip_reproduces_libint_coefficients(self):
        """Output coefficient × N(α, l) reproduces the original."""
        exps = [0.3, 1.0, 3.0]
        coeffs = [0.15, 0.5, 0.35]
        sh = self._make_shell(l=1, exponents=exps, coefficients=coeffs)
        wf = qvf_wf_data(self._result(), self._basis([sh]), self._mol())
        out_shell = wf["basis"][0]
        for a, c_orig, c_out in zip(exps, coeffs, out_shell["coefficients"]):
            assert c_out * _primitive_norm(a, 1) == pytest.approx(c_orig, rel=1e-14)

    def test_multi_shell_multi_l(self):
        """Mixed-l shells each get their correct per-primitive divisor."""
        shells = [
            self._make_shell(l=0, exponents=[0.5], coefficients=[1.0], atom_index=0),
            self._make_shell(
                l=1, exponents=[0.8, 2.0], coefficients=[0.4, 0.6], atom_index=1
            ),
            self._make_shell(l=2, exponents=[1.2], coefficients=[1.0], atom_index=0),
        ]
        wf = qvf_wf_data(self._result(), self._basis(shells), self._mol())
        out_shells = wf["basis"]
        assert len(out_shells) == 3
        for sh_in, sh_out in zip(shells, out_shells):
            l = sh_in.l
            for a, c_in, c_out in zip(
                sh_in.exponents, sh_in.coefficients, sh_out["coefficients"]
            ):
                assert c_out * _primitive_norm(float(a), l) == pytest.approx(
                    c_in, rel=1e-14
                )


class TestSpectraIRSection:
    def test_spectra_ir_missing_intensities(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "ir_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            hessian_result=_stub_hessian(),
        )
        with zipfile.ZipFile(path, "r") as zf:
            secs = [
                s for s in _read_manifest(zf)["sections"] if s["kind"] == "spectra.ir"
            ]
            assert len(secs) == 0  # stub Hessian has no dipole_derivatives


class TestBandsSection:
    def test_bands(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "bands_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            band_structure=_stub_bands(),
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = [s for s in _read_manifest(zf)["sections"] if s["kind"] == "bands"][0]
            m = s["members"]
            assert "kpath" in m
            assert "eigenvalues" in m
            kp = json.loads(zf.read(m["kpath"]["path"]))
            assert "fermi" in kp
            assert len(kp["segments"]) >= 1


class TestCitationsSection:
    def test_citations(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        bib = "% test\n@article{test, author={A}}\n"
        path = write_qvf(
            tmp_path / "cite_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            bibtex_content=bib,
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = [s for s in _read_manifest(zf)["sections"] if s["kind"] == "citations"][
                0
            ]
            assert zf.read(s["members"]["references"]["path"]).decode("utf-8") == bib


class TestTerminalRunStatus:
    """A finished ordinary run must *state* that it finished.

    Spec § 3.2: an absent ``run_status`` means "not stated" and a consumer
    MUST NOT infer one, so an archive that omits it never declares it is
    done — the viewer shows no lifecycle status for it.
    """

    def test_rule_matches_the_container_finalizer(self):
        from vibeqc.output.formats.qvf import terminal_run_status

        assert terminal_run_status(_stub_result()) == "converged"

        class _NotConverged:
            converged = False

        assert terminal_run_status(_NotConverged()) == "failed"

        class _Silent:
            """A result that does not report convergence at all."""

        # Same fallback the container finalizer uses: only an explicit
        # falsy `converged` means failed.
        assert terminal_run_status(_Silent()) == "converged"

    def test_terminal_archive_declares_converged(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "status_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            run_status="converged",
        )
        with zipfile.ZipFile(path, "r") as zf:
            prov = _read_manifest(zf)["provenance"]
            assert prov["run_status"] == "converged"
            assert prov["scf_converged"] is True

    def test_periodic_stamps_from_the_archived_result(self):
        """`run_status` must describe the same result the archive does.

        `periodic_runner` snapshots `_qvf_archive_result` before geometry
        optimization rebinds `result` to its own object -- whose
        `converged` means "geometry converged", not "SCF converged". Since
        `run_status` shares the provenance block with `scf_converged`,
        stamping it from the post-optimization `result` would let one
        provenance block describe two different objects. A real periodic
        optimization is far too slow for a unit test, so pin it in the
        source instead.
        """
        import re
        from pathlib import Path

        import vibeqc

        src = (
            Path(vibeqc.__file__).parent / "periodic_runner.py"
        ).read_text(encoding="utf-8")
        # Each QVF dispatch block must stamp from the object it archives.
        # (The semiempirical route legitimately archives the post-opt
        # `result`; the basis-driven route archives the pre-opt snapshot.)
        blocks = [b for b in src.split("dispatch_role(") if '"qvf"' in b[:40]]
        assert blocks, "periodic runner no longer dispatches a qvf role"
        checked = 0
        for block in blocks:
            archived = re.search(r"(?<![\w.])result=(\w+)", block)
            stamped = re.search(
                r"run_status=_?terminal_run_status\((\w+)\)", block
            )
            if archived is None or stamped is None:
                continue
            checked += 1
            assert archived.group(1) == stamped.group(1), (
                "periodic run_status is stamped from "
                f"{stamped.group(1)!r} but the archive is built from "
                f"{archived.group(1)!r}; one provenance block would then "
                "describe two different results"
            )
        assert checked >= 2, (
            f"expected both periodic QVF dispatches to be checked, saw {checked}"
        )

    def test_terminal_archive_declares_failed(self, tmp_path):
        """A run that did not converge still settles -- as ``failed``."""
        from vibeqc.output.formats.qvf import terminal_run_status

        mol = _stub_molecule()
        plan = _plan(tmp_path)
        result = _stub_result()
        result.converged = False
        path = write_qvf(
            tmp_path / "failed_qvf",
            plan,
            molecule=mol,
            result=result,
            method="rhf",
            basis="sto-3g",
            run_status=terminal_run_status(result),
        )
        with zipfile.ZipFile(path, "r") as zf:
            prov = _read_manifest(zf)["provenance"]
            assert prov["run_status"] == "failed"
            assert prov["scf_converged"] is False


class TestRunRecordSection:
    def test_run_record_input_and_log(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        record = {
            "program": "vibe-qc",
            "program_version": "0.15.57",
            "input_text": "import vibeqc as vq\nvq.run_job(...)\n",
            "input_filename": "input-water.py",
            "log_text": "vibe-qc 0.15.57\nSCF converged\n",
            "log_filename": "water.out",
            "started_utc": "2026-07-24T12:00:00Z",
            "finished_utc": "2026-07-24T12:00:41Z",
        }
        path = write_qvf(
            tmp_path / "rr_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            run_record=record,
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            s = [
                s for s in manifest["sections"] if s["kind"] == "run.record"
            ][0]
            assert s["program"] == "vibe-qc"
            assert s["program_version"] == "0.15.57"
            got_input = zf.read(s["members"]["input"]["path"]).decode("utf-8")
            assert got_input == record["input_text"]
            got_log = zf.read(s["members"]["log"]["path"]).decode("utf-8")
            assert got_log == record["log_text"]
            files = json.loads(zf.read(s["members"]["files"]["path"]))
            assert files["input"]["filename"] == "input-water.py"
            assert files["log"]["filename"] == "water.out"
            assert "truncated" not in files["log"]

    def test_run_record_log_only_with_truncation_flag(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        record = {
            "program": "vibe-qc",
            "log_text": "partial log\n",
            "log_filename": "job.out",
            "log_truncated": True,
        }
        path = write_qvf(
            tmp_path / "rr_trunc_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            run_record=record,
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = [
                s
                for s in _read_manifest(zf)["sections"]
                if s["kind"] == "run.record"
            ][0]
            assert "input" not in s["members"]
            files = json.loads(zf.read(s["members"]["files"]["path"]))
            assert files["log"]["truncated"] is True

    def test_run_record_requires_program_and_payload(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        with pytest.raises(ValueError, match="program"):
            write_qvf(
                tmp_path / "rr_bad1",
                plan,
                molecule=mol,
                result=_stub_result(),
                method="rhf",
                basis="sto-3g",
                run_record={"log_text": "x"},
            )
        with pytest.raises(ValueError, match="input_text"):
            write_qvf(
                tmp_path / "rr_bad2",
                plan,
                molecule=mol,
                result=_stub_result(),
                method="rhf",
                basis="sto-3g",
                run_record={"program": "vibe-qc"},
            )

    def test_assemble_run_record_reads_plan_log(self, tmp_path, monkeypatch):
        from vibeqc.output.formats.qvf import assemble_run_record

        plan = _plan(tmp_path)
        log_file = plan.files_by_role("log")[0]
        log_path = Path(log_file.path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("banner\nSCF converged\n", encoding="utf-8")
        # Pretend the driving script is a real .py file.
        script = tmp_path / "input-h2o.py"
        script.write_text("import vibeqc\n", encoding="utf-8")
        monkeypatch.setattr(
            sys.modules["__main__"], "__file__", str(script), raising=False
        )
        record = assemble_run_record(plan, wall_seconds=41.0)
        assert record is not None
        assert record["program"] == "vibe-qc"
        assert record["log_text"].endswith("SCF converged\n")
        assert record["log_filename"] == log_path.name
        assert record["input_text"] == "import vibeqc\n"
        assert record["input_filename"] == "input-h2o.py"
        # Only basenames -- never absolute paths -- may enter the record.
        for value in record.values():
            if isinstance(value, str):
                assert str(tmp_path) not in value or value in (
                    record["input_text"],
                    record["log_text"],
                )

    def test_assemble_run_record_none_when_nothing_available(
        self, tmp_path, monkeypatch
    ):
        from vibeqc.output.formats.qvf import assemble_run_record

        plan = _plan(tmp_path)  # log file never written to disk
        monkeypatch.delattr(
            sys.modules["__main__"], "__file__", raising=False
        )
        assert assemble_run_record(plan) is None


class TestJobSpecSection:
    def test_pending_container_round_trip(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        spec = {
            "job_type": "molecular",
            "method": "rks",
            "basis": "def2-svp",
            "functional": "pbe0",
            "charge": 0,
            "multiplicity": 1,
            "tasks": ["optimize", "hessian"],
            "options": {"fmax": 0.03},
        }
        path = write_qvf(
            tmp_path / "pending_qvf",
            plan,
            molecule=mol,
            job_spec=spec,
            run_status="pending",
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            assert manifest["provenance"]["run_status"] == "pending"
            s = [
                s for s in manifest["sections"] if s["kind"] == "job.spec"
            ][0]
            got = json.loads(zf.read(s["members"]["spec"]["path"]))
            assert got == spec

    def test_periodic_spec_with_kpoints(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        spec = {
            "job_type": "periodic",
            "method": "rks",
            "functional": "pbe",
            "kpoints": [4, 4, 4],
        }
        path = write_qvf(
            tmp_path / "pending_periodic_qvf",
            plan,
            molecule=mol,
            job_spec=spec,
            run_status="pending",
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = [
                s
                for s in _read_manifest(zf)["sections"]
                if s["kind"] == "job.spec"
            ][0]
            got = json.loads(zf.read(s["members"]["spec"]["path"]))
            assert got["job_type"] == "periodic"
            assert got["kpoints"] == [4, 4, 4]

    def test_job_spec_requires_valid_job_type(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        with pytest.raises(ValueError, match="job_type"):
            write_qvf(
                tmp_path / "js_bad1",
                plan,
                molecule=mol,
                job_spec={"method": "rhf"},
            )
        with pytest.raises(ValueError, match="job_type"):
            write_qvf(
                tmp_path / "js_bad2",
                plan,
                molecule=mol,
                job_spec={"job_type": "orbital"},
            )

    def test_job_spec_rejects_malformed_kpoints(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        with pytest.raises(ValueError, match="kpoints"):
            write_qvf(
                tmp_path / "js_bad_k",
                plan,
                molecule=mol,
                job_spec={"job_type": "periodic", "kpoints": [4, 4]},
            )
        with pytest.raises(ValueError, match="kpoints"):
            write_qvf(
                tmp_path / "js_bad_k0",
                plan,
                molecule=mol,
                job_spec={"job_type": "periodic", "kpoints": [4, 4, 0]},
            )


class TestProvenance:
    def test_provenance_in_manifest(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "prov_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rks",
            functional="PBE",
            basis="6-31G*",
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            prov = manifest.get("provenance", {})
            assert prov.get("method") == "rks"
            assert prov.get("functional") == "PBE"
            assert prov.get("scf_converged") is True
            assert prov.get("n_scf_iterations") == 3
            assert "scf_energy" in prov
            # A molecule is 0-D.
            assert prov.get("dimensionality") == 0

    def test_provenance_hostname_honours_kwarg_optout(self, tmp_path):
        """Regression: the hostname opt-out redacted only the `.system`
        manifest, never the archive.

        Both levers are documented for "runs you plan to share publicly",
        and the `.qvf` is the artefact the docs push hardest for sharing
        (hand to a colleague, attach to a paper's SI), so it was the worst
        place to keep leaking the machine name. Verified against v0.15.76:
        `record_hostname=False` produced `hostname = "<redacted>"` in the
        `.system` and the real FQDN in the archive.
        """
        mol = _stub_molecule()
        path = write_qvf(
            tmp_path / "prov_host_off",
            _plan(tmp_path),
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            record_hostname=False,
        )
        with zipfile.ZipFile(path, "r") as zf:
            prov = _read_manifest(zf).get("provenance", {})
        # Present but masked: consumers key off the field's existence.
        assert prov.get("hostname") == "<redacted>"

    def test_provenance_hostname_honours_env_optout(self, tmp_path, monkeypatch):
        """`VIBEQC_NO_HOSTNAME=1` is the lever the docs recommend for
        paper artifacts, so it must reach the archive too."""
        monkeypatch.setenv("VIBEQC_NO_HOSTNAME", "1")
        path = write_qvf(
            tmp_path / "prov_host_env",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
        )
        with zipfile.ZipFile(path, "r") as zf:
            prov = _read_manifest(zf).get("provenance", {})
        assert prov.get("hostname") == "<redacted>"

    def test_provenance_hostname_recorded_by_default(self, tmp_path, monkeypatch):
        """The opt-out is opt-in: an ordinary run still records the host,
        which is what makes a reported wall time interpretable."""
        monkeypatch.delenv("VIBEQC_NO_HOSTNAME", raising=False)
        path = write_qvf(
            tmp_path / "prov_host_on",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
        )
        with zipfile.ZipFile(path, "r") as zf:
            prov = _read_manifest(zf).get("provenance", {})
        assert prov.get("hostname")
        assert prov.get("hostname") != "<redacted>"

    def test_provenance_dimensionality_from_periodic_system(self, tmp_path):
        """Regression: the bound PeriodicSystem exposes ``dim`` (not
        ``dimensionality``), and the provenance writer used to read only
        the latter -- so every real 3-D periodic run recorded
        ``"dimensionality": 0`` in manifest.json (qc-input-library
        artifact 01706-c-diamond-rhf-sto3g-k222-qvf, v0.15.28)."""
        atoms = list(_stub_molecule().atoms)
        lattice = np.eye(3, dtype=np.float64) * 5.0
        system = type(
            "PeriodicSys",
            (),
            {"unit_cell": atoms, "lattice": lattice, "dim": 3},
        )()
        plan = _plan(tmp_path, job_kind="periodic_scf")
        path = write_qvf(
            tmp_path / "prov_dim_qvf",
            plan,
            system=system,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
        )
        with zipfile.ZipFile(path, "r") as zf:
            prov = _read_manifest(zf).get("provenance", {})
            assert prov.get("dimensionality") == 3

    def test_bidi_control_chars_neutralised_in_archive(self, tmp_path):
        # QVF JSON members are written ensure_ascii=False, so without
        # scrubbing a bidi override / zero-width char in user provenance
        # (or a section label) would land raw in a zip member. Assert no
        # member name or text content carries the dangerous class, and the
        # manifest still parses with the legitimate text preserved.
        from vibeqc.output._text_safety import is_unsafe_output_char

        rlo, zwsp = chr(0x202E), chr(0x200B)
        path = write_qvf(
            tmp_path / "bidi_qvf",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rks",
            functional=f"PBE{rlo}evil",
            basis=f"6-31G*{zwsp}",
        )
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                assert not any(is_unsafe_output_char(c) for c in name), (
                    f"raw danger char in zip member name {name!r}"
                )
                if name.endswith(".dat"):
                    continue  # binary payload, not text
                text = zf.read(name).decode("utf-8")
                assert not any(is_unsafe_output_char(c) for c in text), (
                    f"raw danger char in member {name}"
                )
            prov = _read_manifest(zf).get("provenance", {})
            assert prov.get("method") == "rks"
            assert "PBE" in prov.get("functional", "")  # legit part preserved
            assert rlo not in prov.get("functional", "")  # bidi neutralised


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_full_roundtrip(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        density = np.random.randn(8, 8, 8).astype(np.float32)
        origin = np.array([-2.0, -2.0, -2.0])
        span = np.eye(3) * 0.5
        path = write_qvf(
            tmp_path / "full_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            volume_data={"Density": (density, origin, span)},
            mo_data=[
                {
                    "label": "HOMO",
                    "data": np.random.randn(8, 8, 8).astype(np.float32),
                    "origin": origin,
                    "span": span,
                    "band_index": 4,
                    "energy_eh": -0.5,
                    "component": "real",
                }
            ],
            population_summary=_stub_population(),
            hessian_result=_stub_hessian(),
            bibtex_content="% bib\n",
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            assert manifest["qvf_version"] == QVF_FORMAT_VERSION
            kinds = {s["kind"] for s in manifest["sections"]}
            assert {
                "structure",
                "volume.density",
                "volume.orbital",
                "atom_properties",
                "vibrations",
                "citations",
            }.issubset(kinds)
            for sec in manifest["sections"]:
                for _k, m in sec.get("members", {}).items():
                    sha = m.get("sha256")
                    if sha is not None:
                        got = _sha256_hex(zf.read(m["path"]))
                        assert got == sha, f"sha256: {m['path']}"
            dens_sec = [
                s for s in manifest["sections"] if s["kind"] == "volume.density"
            ][0]
            dens_raw = zf.read(dens_sec["members"]["data"]["path"])
            dens_arr = np.frombuffer(dens_raw, dtype=np.float32).reshape(8, 8, 8)
            np.testing.assert_allclose(dens_arr[3, 3, 3], density[3, 3, 3])
        report = validate_qvf(path)
        assert report["valid"], f"fail: {report['errors']}"

    def test_writer_qvf_opens_with_vibe_view_reader_when_installed(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        density = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        origin = np.array([-1.0, -2.0, -3.0])
        voxel_vectors = np.diag([0.2, 0.3, 0.4])
        path = write_qvf(
            tmp_path / "viewer_integration",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            volume_data={"Electron density": (density, origin, voxel_vectors)},
        )

        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            assert manifest["qvf_version"] == QVF_FORMAT_VERSION
            sections = {section["id"]: section for section in manifest["sections"]}
            assert set(sections) == {"structure", "vol_dens_0"}

            structure = sections["structure"]
            assert structure["kind"] == "structure"
            assert set(structure["members"]) == {"structure"}
            struct_member = structure["members"]["structure"]
            assert struct_member["path"] == "structure/structure.json"
            struct_payload = json.loads(zf.read(struct_member["path"]))
            assert [atom["symbol"] for atom in struct_payload["atoms"]] == [
                "O",
                "H",
                "H",
            ]
            assert struct_payload["pbc"] == [False, False, False]

            volume = sections["vol_dens_0"]
            assert volume["kind"] == "volume.density"
            assert set(volume["members"]) == {"data", "grid"}
            data_member = volume["members"]["data"]
            assert data_member["dtype"] == "float32"
            assert data_member["shape"] == [2, 2, 2]
            grid_member = volume["members"]["grid"]
            assert grid_member["format"] == "json"
            grid = json.loads(zf.read(grid_member["path"]))
            assert grid["origin"] == [-1.0, -2.0, -3.0]
            assert grid["shape"] == [2, 2, 2]
            assert grid["voxel_vectors"] == [
                [0.2, 0.0, 0.0],
                [0.0, 0.3, 0.0],
                [0.0, 0.0, 0.4],
            ]

        vibeview_qvf = pytest.importorskip(
            "vibeview.qvf",
            reason="vibe-view QVF reader is not installed in this environment",
        )
        with vibeview_qvf.QVFReader(path) as reader:
            structure = reader.read_structure()
            assert len(structure.atoms) == 3
            assert structure.atoms[0].symbol == "O"
            assert reader.has_section("vol_dens_0")
            grid = reader.read_volume_grid("vol_dens_0")
            assert grid.shape == (2, 2, 2)
            np.testing.assert_allclose(grid.origin, origin)
            np.testing.assert_allclose(grid.voxel_vectors, voxel_vectors)
            np.testing.assert_allclose(reader.read_volume_data("vol_dens_0"), density)

    def test_volume_potential_roundtrip(self, tmp_path):
        """volume.potential writes and round-trips with the same member
        structure as volume.density."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        esp = np.arange(27, dtype=np.float64).reshape(3, 3, 3) / 27.0
        origin = np.array([-1.0, -2.0, -3.0])
        span = np.diag([0.2, 0.3, 0.4])
        path = write_qvf(
            tmp_path / "pot_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            potential_data={"ESP": (esp, origin, span)},
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            kinds = {s["kind"] for s in manifest["sections"]}
            assert "volume.potential" in kinds
            esp_sec = [
                s for s in manifest["sections"] if s["kind"] == "volume.potential"
            ][0]
            assert esp_sec["label"] == "ESP"
            assert set(esp_sec["members"]) == {"data", "grid"}
            dm = esp_sec["members"]["data"]
            assert dm["dtype"] == "float32"
            assert dm["shape"] == [3, 3, 3]
            gm = esp_sec["members"]["grid"]
            assert gm["format"] == "json"
            grid = json.loads(zf.read(gm["path"]))
            assert grid["origin"] == [-1.0, -2.0, -3.0]
        report = validate_qvf(path)
        assert report["valid"], f"fail: {report['errors']}"

    def test_volume_rdg_roundtrip(self, tmp_path):
        """volume.rdg writes and round-trips with the same member
        structure as volume.density."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        rdg = np.random.randn(4, 4, 4).astype(np.float32)
        origin = np.array([-1.0, -2.0, -3.0])
        span = np.diag([0.2, 0.3, 0.4])
        path = write_qvf(
            tmp_path / "rdg_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            rdg_data={"RDG": (rdg, origin, span)},
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            kinds = {s["kind"] for s in manifest["sections"]}
            assert "volume.rdg" in kinds
            rdg_sec = [s for s in manifest["sections"] if s["kind"] == "volume.rdg"][0]
            assert rdg_sec["label"] == "RDG"
            assert set(rdg_sec["members"]) == {"data", "grid"}
        report = validate_qvf(path)
        assert report["valid"], f"fail: {report['errors']}"

    def test_eos_roundtrip(self, tmp_path):
        """equation_of_state writes and validates."""
        mol = _stub_molecule()
        plan = _plan(tmp_path, job_kind="periodic_scf")
        vols = np.linspace(100.0, 120.0, 7)
        engs = -5000.0 + 0.01 * (vols - 110.0) ** 2
        path = write_qvf(
            tmp_path / "eos_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            eos_data={
                "volumes": vols,
                "energies": engs,
                "fit": {
                    "model": "birch_murnaghan",
                    "V0": 110.0,
                    "E0": -5000.0,
                    "B0": 75.0,
                    "B0_prime": 4.0,
                },
            },
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            kinds = {s["kind"] for s in manifest["sections"]}
            assert "equation_of_state" in kinds
            eos_sec = [
                s for s in manifest["sections"] if s["kind"] == "equation_of_state"
            ][0]
            assert set(eos_sec["members"]) == {"volumes", "energies", "fit"}
            fit = json.loads(zf.read(eos_sec["members"]["fit"]["path"]))
            assert fit["model"] == "birch_murnaghan"
            assert fit["V0"] == 110.0
        report = validate_qvf(path)
        assert report["valid"], f"fail: {report['errors']}"

    def test_fermi_surface_roundtrip(self, tmp_path):
        """fermi_surface writes and validates with 4D energies."""
        mol = _stub_molecule()
        plan = _plan(tmp_path, job_kind="periodic_scf")
        nk1, nk2, nk3, nb = 4, 4, 4, 2
        energies = np.random.randn(nk1, nk2, nk3, nb).astype(np.float64)
        lattice = np.eye(3) * 4.0
        path = write_qvf(
            tmp_path / "fermi_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            fermi_surface_data={
                "nk1": nk1,
                "nk2": nk2,
                "nk3": nk3,
                "energies": energies,
                "band_indices": [3, 4],
                "lattice_vectors": lattice,
                "fermi_energy_ev": -4.71,
            },
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            kinds = {s["kind"] for s in manifest["sections"]}
            assert "fermi_surface" in kinds
            fs_sec = [s for s in manifest["sections"] if s["kind"] == "fermi_surface"][
                0
            ]
            assert set(fs_sec["members"]) == {"mesh", "energies"}
            dm = fs_sec["members"]["energies"]
            assert dm["dtype"] == "float64"
            assert dm["shape"] == [nk1, nk2, nk3, nb]
        report = validate_qvf(path)
        assert report["valid"], f"fail: {report['errors']}"

    def test_phonon_bands_roundtrip(self, tmp_path):
        """phonon_bands writes and validates."""
        mol = _stub_molecule()
        plan = _plan(tmp_path, job_kind="periodic_scf")
        n_qpts, n_modes = 50, 9
        path = write_qvf(
            tmp_path / "phb_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            phonon_bands_data={
                "qpath": {
                    "n_atoms": 3,
                    "n_modes": n_modes,
                    "has_eigenvectors": False,
                    "segments": [
                        {
                            "label_start": "G",
                            "label_end": "X",
                            "k_start": [0, 0, 0],
                            "k_end": [0.5, 0, 0],
                            "n_points": n_qpts,
                        }
                    ],
                },
                "frequencies": np.random.randn(n_qpts, n_modes).astype(np.float64),
            },
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            kinds = {s["kind"] for s in manifest["sections"]}
            assert "phonon_bands" in kinds
            pb = [s for s in manifest["sections"] if s["kind"] == "phonon_bands"][0]
            assert set(pb["members"]) == {"qpath", "frequencies"}
        report = validate_qvf(path)
        assert report["valid"], f"fail: {report['errors']}"

    def test_phonon_dos_roundtrip(self, tmp_path):
        """phonon_dos writes and validates."""
        mol = _stub_molecule()
        plan = _plan(tmp_path, job_kind="periodic_scf")
        n_pts = 200
        freq = np.linspace(0, 500, n_pts)
        dos_arr = np.exp(-0.5 * ((freq - 200) / 50) ** 2)
        path = write_qvf(
            tmp_path / "phd_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            phonon_dos_data={
                "frequencies": freq,
                "dos": dos_arr,
                "meta": {"smearing": 5.0, "smearing_type": "gaussian"},
            },
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            kinds = {s["kind"] for s in manifest["sections"]}
            assert "phonon_dos" in kinds
            pd = [s for s in manifest["sections"] if s["kind"] == "phonon_dos"][0]
            assert set(pd["members"]) == {"meta", "frequencies", "dos"}
        report = validate_qvf(path)
        assert report["valid"], f"fail: {report['errors']}"

    def test_root_thermochemistry(self, tmp_path):
        """Root thermochemistry metadata writes and validates."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "thermo_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            thermochemistry_data={
                "zpve_eh": 0.0294,
                "enthalpy_eh": -76.3077,
                "entropy_cal_mol_k": 52.3,
                "gibbs_free_energy_eh": -76.3321,
                "temperature_k": 298.15,
                "pressure_atm": 1.0,
            },
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            assert manifest["thermochemistry"]["zpve_eh"] == 0.0294
            assert manifest["thermochemistry"]["temperature_k"] == 298.15
        report = validate_qvf(path)
        assert report["valid"], f"fail: {report['errors']}"

    def test_root_dipole_moment(self, tmp_path):
        """Root dipole_moment metadata writes and validates."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "dipole_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            dipole_moment_data={
                "total_debye": 1.85,
                "vector_debye": [0.0, 0.0, 1.85],
                "origin": "center_of_mass",
            },
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            assert manifest["dipole_moment"]["total_debye"] == 1.85
            assert manifest["dipole_moment"]["vector_debye"] == [0.0, 0.0, 1.85]
        report = validate_qvf(path)
        assert report["valid"], f"fail: {report['errors']}"

    def test_root_dipole_moment_accepts_numeric_origin_bohr(self, tmp_path):
        """An explicit reference point stays numeric instead of stringified."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        origin = [0.0, 0.0, 1.1418788550444696]
        path = write_qvf(
            tmp_path / "dipole_origin_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            dipole_moment_data={
                "total_debye": 1.85,
                "vector_debye": [0.0, 0.0, 1.85],
                "origin": origin,
            },
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            assert manifest["dipole_moment"]["origin"] == origin
            assert isinstance(manifest["dipole_moment"]["origin"], list)
        report = validate_qvf(path)
        assert report["valid"], f"fail: {report['errors']}"

    def test_root_constraints(self, tmp_path):
        """Root constraints metadata writes and validates."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "constraints_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            constraints_data={
                "frozen_atoms": [3, 4],
                "distance_constraints": [{"atoms": [0, 1], "target_angstrom": 1.5}],
            },
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            assert manifest["constraints"]["frozen_atoms"] == [3, 4]
            assert manifest["constraints"]["distance_constraints"][0]["atoms"] == [0, 1]
        report = validate_qvf(path)
        assert report["valid"], f"fail: {report['errors']}"

    def test_root_extensions_block(self, tmp_path):
        """Root extensions governance block writes and validates."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "ext_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            extensions={
                "x_vendor_ecp": {
                    "version": "1.0",
                    "schema_uri": "https://vendor.example.org/qvf/ecp.schema.json",
                    "critical": False,
                },
                "x_other_plugin": {
                    "version": "2.1",
                    "critical": False,
                },
            },
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            assert manifest["extensions"]["x_vendor_ecp"]["version"] == "1.0"
            assert not manifest["extensions"]["x_vendor_ecp"]["critical"]
        report = validate_qvf(path)
        assert report["valid"], f"fail: {report['errors']}"

    def test_vendor_json_sections(self, tmp_path):
        """Caller-supplied vendor JSON sections write through the normal writer."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        payload = {
            "route": "aiccm2026dev-b",
            "finite_torus_convention": {
                "coulomb_kernel": "3d-periodic-g0",
                "exchange_q0": "bvk-ewald",
            },
        }
        path = write_qvf(
            tmp_path / "vendor_json_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            extensions={"x_vibeqc": {"version": "1.0", "critical": False}},
            vendor_json_sections=[
                {
                    "id": "x_vibeqc_aiccm2026dev_b_convention",
                    "kind": "x_vibeqc.aiccm2026dev_b_convention",
                    "member": "convention",
                    "label": "χ-CCM-B finite-torus convention",
                    "payload": payload,
                }
            ],
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            section = next(
                s for s in manifest["sections"]
                if s["kind"] == "x_vibeqc.aiccm2026dev_b_convention"
            )
            member = section["members"]["convention"]
            got = json.loads(zf.read(member["path"]))
            assert got == payload
            assert section["label"] == "χ-CCM-B finite-torus convention"
            assert manifest["extensions"]["x_vibeqc"]["version"] == "1.0"
        report = validate_qvf(path)
        assert report["valid"], f"fail: {report['errors']}"

    def test_empty_qvf(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "e_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
        )
        assert validate_qvf(path)["valid"]


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.slow
    def test_small_scf_produces_valid_qvf(self, tmp_path):
        from vibeqc import Atom, Molecule
        from vibeqc._vibeqc_core import BasisSet
        from vibeqc.output.citations import assemble, load_default_database
        from vibeqc.output.citations.bibtex import format_bibtex
        from vibeqc.runner import run_job

        mol = Molecule(
            [
                Atom(8, (0, 0, 0.1173)),
                Atom(1, (0, 1.4315, -0.9314)),
                Atom(1, (0, -1.4315, -0.9314)),
            ],
            charge=0,
            multiplicity=1,
        )
        result = run_job(
            mol,
            basis="sto-3g",
            method="rhf",
            output=tmp_path / "h2o_e2e",
            write_population_file=True,
            citations=True,
        )
        basis_obj = BasisSet(mol, "sto-3g")
        pop = None
        try:
            from vibeqc.properties import loewdin_charges, mulliken_charges

            q_mul = mulliken_charges(result, basis_obj, mol)
            q_low = loewdin_charges(result, basis_obj, mol)

            class _P:
                pass

            p = _P()
            zl = [int(a.Z) for a in mol.atoms]
            p.mulliken_atoms = [
                (i, f"Z{z}", float(z), float(z - q_mul[i])) for i, z in enumerate(zl)
            ]
            p.loewdin_atoms = [
                (i, f"Z{z}", float(z), float(z - q_low[i])) for i, z in enumerate(zl)
            ]
            p.mayer_bonds = []
            p.dipole = None
            p.errors = {}
            pop = p
        except Exception:
            pass
        try:
            db = load_default_database()
            bibtex = format_bibtex(assemble(db, method="rhf", basis_name="sto-3g"))
        except Exception:
            bibtex = "% no\n"
        plan = OutputPlan.from_run_job_kwargs(
            output=tmp_path / "h2o_q",
            method="rhf",
            basis="sto-3g",
            functional=None,
            write_population=True,
            citations=True,
        )
        kw = dict(
            molecule=mol,
            result=result,
            method="rhf",
            basis="sto-3g",
            bibtex_content=bibtex,
        )
        if pop is not None:
            kw["population_summary"] = pop
        qvf_path = write_qvf(tmp_path / "h2o_q", plan, **kw)
        report = validate_qvf(qvf_path)
        assert report["valid"], f"E2E fail: {report['errors']}"
        with zipfile.ZipFile(qvf_path, "r") as zf:
            kinds = {s["kind"] for s in _read_manifest(zf)["sections"]}
            assert "structure" in kinds
            assert "citations" in kinds

    @pytest.mark.slow
    def test_run_job_qvf_includes_bond_orders_and_dipole(self, tmp_path):
        """run_job(output_qvf=True) auto-emits bond_orders and dipole_moment
        from the population summary."""
        from vibeqc import Atom, Molecule
        from vibeqc.runner import run_job

        mol = Molecule(
            [
                Atom(8, (0, 0, 0.1173)),
                Atom(1, (0, 1.4315, -0.9314)),
                Atom(1, (0, -1.4315, -0.9314)),
            ],
            charge=0,
            multiplicity=1,
        )
        result = run_job(
            mol,
            basis="sto-3g",
            method="rhf",
            output=tmp_path / "h2o_int",
            write_population_file=True,
            citations=False,
            optimize=False,
            hessian=False,
            output_qvf=True,
        )
        qvf_path = tmp_path / "h2o_int.qvf"
        assert qvf_path.exists()
        report = validate_qvf(qvf_path)
        assert report["valid"], f"fail: {report['errors']}"
        population_payload = json.loads(
            (tmp_path / "h2o_int.population.json").read_text(encoding="utf-8")
        )

        with zipfile.ZipFile(qvf_path, "r") as zf:
            manifest = _read_manifest(zf)
            kinds = {s["kind"] for s in manifest["sections"]}
            # bond_orders should be auto-emitted
            assert "bond_orders" in kinds, f"bond_orders missing; kinds={sorted(kinds)}"
            bo_sec = [s for s in manifest["sections"] if s["kind"] == "bond_orders"][0]
            bo_payload = json.loads(zf.read(bo_sec["members"]["bond_orders"]["path"]))
            assert bo_payload["method"] == "mayer"
            # H2O should have at least the two O-H bonds
            assert len(bo_payload["pairs"]) >= 2
            for p in bo_payload["pairs"]:
                assert "order" in p
                assert "distance_ang" in p
                assert "symbol_i" in p
                assert "symbol_j" in p
            # dipole_moment should be in manifest root
            assert "dipole_moment" in manifest, (
                "dipole_moment missing from manifest root"
            )
            d = manifest["dipole_moment"]
            assert "total_debye" in d
            assert "vector_debye" in d
            assert len(d["vector_debye"]) == 3
            assert isinstance(d["origin"], list)
            assert len(d["origin"]) == 3
            np.testing.assert_allclose(
                d["origin"],
                population_payload["dipole"]["origin_bohr"],
            )
            props = next(
                section
                for section in manifest["sections"]
                if section["kind"] == "atom_properties"
            )
            assert "hirshfeld_charge" in props["members"]
            hirshfeld = np.frombuffer(
                zf.read(props["members"]["hirshfeld_charge"]["path"]),
                dtype=np.float64,
            )
            np.testing.assert_allclose(
                hirshfeld,
                [row["charge"] for row in population_payload["hirshfeld"]],
            )
            # H2O dipole should be non-zero (~1.7-1.9 Debye for STO-3G)
            assert abs(d["total_debye"]) > 0.5


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------


class TestValidateQVF:
    def test_valid_file_passes(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "v_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
        )
        report = validate_qvf(path)
        assert report["valid"]

    def test_missing_file(self, tmp_path):
        assert not validate_qvf(tmp_path / "nope.qvf")["valid"]

    def test_non_zip(self, tmp_path):
        (tmp_path / "bad.qvf").write_text("no")
        assert not validate_qvf(tmp_path / "bad.qvf")["valid"]

    def test_no_manifest(self, tmp_path):
        with zipfile.ZipFile(tmp_path / "nm.qvf", "w") as zf:
            zf.writestr("d.bin", b"\x00")
        assert not validate_qvf(tmp_path / "nm.qvf")["valid"]

    def test_bad_json(self, tmp_path):
        with zipfile.ZipFile(tmp_path / "bj.qvf", "w") as zf:
            zf.writestr("manifest.json", b"{!")
        assert not validate_qvf(tmp_path / "bj.qvf")["valid"]

    def test_invalid_json_member(self, tmp_path):
        payload = b"{not json"
        with zipfile.ZipFile(tmp_path / "bad_member.qvf", "w") as zf:
            zf.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "qvf_version": 1,
                        "source": {"program": "t", "version": "0", "calculation": "t"},
                        "sections": [
                            {
                                "id": "structure",
                                "kind": "structure",
                                "members": {
                                    "structure": {
                                        "path": "structure/structure.json",
                                        "format": "json",
                                        "sha256": _sha256_hex(payload),
                                    },
                                },
                            },
                        ],
                    }
                ).encode(),
            )
            zf.writestr("structure/structure.json", payload)
        report = validate_qvf(tmp_path / "bad_member.qvf")
        assert not report["valid"]
        assert any("does not parse" in e for e in report["errors"])

    def test_duplicate_section_id(self, tmp_path):
        payload = json.dumps({"atoms": []}).encode()
        section = {
            "id": "structure",
            "kind": "structure",
            "members": {
                "structure": {
                    "path": "structure/structure.json",
                    "format": "json",
                    "sha256": _sha256_hex(payload),
                },
            },
        }
        with zipfile.ZipFile(tmp_path / "duplicate.qvf", "w") as zf:
            zf.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "qvf_version": 1,
                        "source": {"program": "t", "version": "0", "calculation": "t"},
                        "sections": [section, section],
                    }
                ).encode(),
            )
            zf.writestr("structure/structure.json", payload)
        report = validate_qvf(tmp_path / "duplicate.qvf")
        assert not report["valid"]
        assert any("duplicate section id" in e for e in report["errors"])

    def test_sha256_mismatch(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "corr_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
        )
        cp = tmp_path / "corr.qvf"
        with zipfile.ZipFile(path, "r") as zi:
            with zipfile.ZipFile(cp, "w") as zo:
                for n in zi.namelist():
                    d = zi.read(n)
                    if n == "structure/structure.json":
                        d = bytearray(d)
                        d[0] ^= 0xFF
                        d = bytes(d)
                    zo.writestr(n, d)
        assert not validate_qvf(cp)["valid"]

    def test_unknown_kind(self, tmp_path):
        with zipfile.ZipFile(tmp_path / "uk.qvf", "w") as zf:
            zf.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "qvf_version": 1,
                        "source": {"program": "t", "version": "0", "calculation": "t"},
                        "sections": [{"id": "s1", "kind": "nonsense_kind"}],
                    }
                ).encode(),
            )
        assert not validate_qvf(tmp_path / "uk.qvf")["valid"]

    def test_vendor_kind(self, tmp_path):
        # Vendor (x_*) sections are accepted and noted as not-shape-validated.
        # Minimal valid shape: an empty `members` dict satisfies
        # SectionVendor (which only constrains the shape of members that
        # exist, not their count).
        with zipfile.ZipFile(tmp_path / "vk.qvf", "w") as zf:
            zf.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "qvf_version": 1,
                        "source": {"program": "t", "version": "0", "calculation": "t"},
                        "sections": [
                            {"id": "s1", "kind": "x_vibeqc.custom", "members": {}},
                        ],
                    }
                ).encode(),
            )
        report = validate_qvf(tmp_path / "vk.qvf")
        assert report["valid"], f"errors: {report['errors']}"
        assert any("vendor" in s.lower() for s in report["summary"])

    def test_reserved_kind(self, tmp_path):
        # Reserved kinds (documented for forward-compat, no writer yet)
        # are noted but their member shape is not validated.
        with zipfile.ZipFile(tmp_path / "rk.qvf", "w") as zf:
            zf.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "qvf_version": 1,
                        "source": {"program": "t", "version": "0", "calculation": "t"},
                        "sections": [
                            {
                                "id": "s1",
                                "kind": "volume.orbital_projection",
                                "members": {},
                            },
                        ],
                    }
                ).encode(),
            )
        report = validate_qvf(tmp_path / "rk.qvf")
        assert report["valid"], f"errors: {report['errors']}"
        assert any("reserved" in s.lower() for s in report["summary"])

    def test_critical_on_reserved_kind_is_error(self, tmp_path):
        """critical=true on a reserved kind must fail validation."""
        with zipfile.ZipFile(tmp_path / "cr.qvf", "w") as zf:
            zf.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "qvf_version": 1,
                        "source": {"program": "t", "version": "0", "calculation": "t"},
                        "sections": [
                            {
                                "id": "s1",
                                "kind": "volume.orbital_projection",
                                "critical": True,
                                "members": {},
                            },
                        ],
                    }
                ).encode(),
            )
        report = validate_qvf(tmp_path / "cr.qvf")
        assert not report["valid"]
        assert any("critical" in e.lower() for e in report["errors"])

    def test_critical_on_vendor_kind_passes(self, tmp_path):
        """critical=true on a vendor kind is accepted (consumer concern)."""
        with zipfile.ZipFile(tmp_path / "cv.qvf", "w") as zf:
            zf.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "qvf_version": 1,
                        "source": {"program": "t", "version": "0", "calculation": "t"},
                        "sections": [
                            {
                                "id": "s1",
                                "kind": "x_vibeqc.custom",
                                "critical": True,
                                "members": {},
                            },
                        ],
                    }
                ).encode(),
            )
        report = validate_qvf(tmp_path / "cv.qvf")
        assert report["valid"], f"errors: {report['errors']}"

    def test_extensions_block_validates(self, tmp_path):
        """Root extensions block with valid structure passes."""
        with zipfile.ZipFile(tmp_path / "ext.qvf", "w") as zf:
            zf.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "qvf_version": 1,
                        "source": {"program": "t", "version": "0", "calculation": "t"},
                        "extensions": {
                            "x_vendor_ecp": {
                                "version": "1.0",
                                "critical": False,
                            },
                        },
                        "sections": [],
                    }
                ).encode(),
            )
        report = validate_qvf(tmp_path / "ext.qvf")
        assert report["valid"], f"errors: {report['errors']}"

    def test_missing_required_field(self, tmp_path):
        with zipfile.ZipFile(tmp_path / "mf.qvf", "w") as zf:
            zf.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "qvf_version": 1,
                        "source": {"program": "t", "version": "0", "calculation": "t"},
                    }
                ).encode(),
            )
        assert not validate_qvf(tmp_path / "mf.qvf")["valid"]

    def test_zip_bomb_guard(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "zb_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
        )
        assert validate_qvf(path)["valid"]  # guard doesn't false-positive


# ---------------------------------------------------------------------------
# Guardrail tests
# ---------------------------------------------------------------------------


class TestGuardrails:
    def test_volume_exceeds_max(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)

        class _H:
            shape = (1025, 1024, 1024)
            ndim = 3

        with pytest.raises(ValueError, match="exceeds"):
            write_qvf(
                tmp_path / "big",
                plan,
                molecule=mol,
                result=_stub_result(),
                method="rhf",
                basis="sto-3g",
                volume_data={"b": (_H(), np.zeros(3), np.eye(3))},
            )

    def test_missing_molecule_warns(self, tmp_path):
        plan = _plan(tmp_path)
        with pytest.warns(UserWarning, match="no .molecule."):
            write_qvf(
                tmp_path / "nm",
                plan,
                result=_stub_result(),
                method="rhf",
                basis="sto-3g",
            )

    def test_empty_atoms_warns(self, tmp_path):
        plan = _plan(tmp_path)
        em = type("M", (), {"atoms": [], "charge": 0, "multiplicity": 1})()
        with pytest.warns(UserWarning, match="has no atoms"):
            write_qvf(
                tmp_path / "ea",
                plan,
                molecule=em,
                result=_stub_result(),
                method="rhf",
                basis="sto-3g",
            )

    def test_bad_volume_shape_raises_not_silently_skipped(self, tmp_path):
        """Pre-fix the writer dropped non-3-D volume payloads with
        `continue`, so a user passing bad data got a QVF missing that
        section without any signal. Now it surfaces a ValueError that
        names the offending label and kind."""
        plan = _plan(tmp_path)
        bad = np.zeros((4, 4), dtype=np.float32)  # 2-D
        with pytest.raises(ValueError, match=r"volume.density\['bad_dens'\]"):
            write_qvf(
                tmp_path / "bad_shape",
                plan,
                molecule=_stub_molecule(),
                result=_stub_result(),
                method="rhf",
                basis="sto-3g",
                volume_data={"bad_dens": (bad, np.zeros(3), np.eye(3))},
            )

    def test_complex_density_volume_residual_is_projected_quietly(self, tmp_path):
        import warnings

        plan = _plan(tmp_path)
        real = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
        data = real + 1.0e-10j

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path = write_qvf(
                tmp_path / "complex_residual",
                plan,
                molecule=_stub_molecule(),
                result=_stub_result(),
                method="rhf",
                basis="sto-3g",
                volume_data={"Density": (data, np.zeros(3), np.eye(3))},
            )

        assert not any(
            "ComplexWarning" in warning.category.__name__ for warning in caught
        )
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            section = next(
                s for s in manifest["sections"] if s["kind"] == "volume.density"
            )
            raw = zf.read(section["members"]["data"]["path"])
        stored = np.frombuffer(raw, dtype=np.float32).reshape(real.shape)
        np.testing.assert_allclose(stored, real.astype(np.float32))

    def test_complex_density_volume_large_imaginary_part_is_gated(self, tmp_path):
        plan = _plan(tmp_path)
        bad = np.ones((2, 2, 2), dtype=np.complex128) * (1.0 + 1.0e-3j)
        with pytest.raises(ValueError, match="volume.density.*non-negligible"):
            write_qvf(
                tmp_path / "complex_bad",
                plan,
                molecule=_stub_molecule(),
                result=_stub_result(),
                method="rhf",
                basis="sto-3g",
                volume_data={"bad": (bad, np.zeros(3), np.eye(3))},
            )

    def test_label_slugifies_in_zip_paths(self, tmp_path):
        """Label-derived zip paths must not propagate path-traversal
        sequences or other unsafe characters. Pre-fix the writer only
        replaced spaces with underscores, so a label like
        '../etc/passwd' would have become a member at
        'volumes/../etc/passwd.dat' (still rejected later by the
        schema, but the bytes hit the zip first)."""
        plan = _plan(tmp_path)
        data = np.zeros((4, 4, 4), dtype=np.float32)
        path = write_qvf(
            tmp_path / "slug_qvf",
            plan,
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            volume_data={
                "../etc/passwd": (data, np.zeros(3), np.eye(3)),
                "ρ(prod) − ρ(react)": (data, np.zeros(3), np.eye(3)),
            },
        )
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            # No raw path traversal in any member name.
            assert not any(".." in n for n in names), names
            # No member name carries the literal Greek/minus glyphs.
            assert not any("ρ" in n or "−" in n for n in names), names
            # Each volume payload landed under volumes/ with an
            # ascii-safe stem.
            vol_dats = [
                n for n in names if n.startswith("volumes/") and n.endswith(".dat")
            ]
            assert len(vol_dats) == 2
            allowed = set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            )
            for n in vol_dats:
                stem = n[len("volumes/") : -len(".dat")]
                assert all(c in allowed for c in stem), f"unsafe char in {stem!r}"

    def test_writer_and_validator_size_caps_agree(self, tmp_path):
        """Regression: the writer's per-payload voxel cap and the
        validator's per-zip-member uncompressed-bytes cap must agree,
        so a write_qvf output cannot subsequently fail validate_qvf as
        "too large". Pre-fix the validator capped members at 128 MiB
        while the writer accepted up to ~4 GiB float32 payloads — a
        300³ float64 grid (~216 MiB) passed write_qvf and then failed
        validate_qvf, contradicting the "write_qvf never returns an
        invalid path" gate."""
        from vibeqc.output.formats.qvf import (
            _MAX_MEMBER_UNCOMPRESSED_BYTES,
            _MAX_VOXELS,
        )

        # Validator cap ≥ writer's largest emittable single payload
        # (float64 worst case).
        assert _MAX_MEMBER_UNCOMPRESSED_BYTES >= _MAX_VOXELS * 8

    def test_bad_plan_type(self, tmp_path):
        with pytest.raises(TypeError, match="OutputPlan"):
            write_qvf(tmp_path / "bp", None, molecule=_stub_molecule())


# ---------------------------------------------------------------------------
# Reserved kinds + bands segments
# ---------------------------------------------------------------------------


class TestReservedKinds:
    def test_volume_elf(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        data = np.random.randn(6, 6, 6).astype(np.float32)
        path = write_qvf(
            tmp_path / "elf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            elf_data={"ELF": (data, np.zeros(3), np.eye(3) * 0.5)},
        )
        with zipfile.ZipFile(path, "r") as zf:
            assert any(
                s["kind"] == "volume.elf" for s in _read_manifest(zf)["sections"]
            )
        assert validate_qvf(path)["valid"]

    def test_spectra_raman(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "raman",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            raman_data={"frequencies": [100, 200], "intensities": [0.1, 0.5]},
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = [
                s
                for s in _read_manifest(zf)["sections"]
                if s["kind"] == "spectra.raman"
            ][0]
            assert "spectrum" in s["members"]
        assert validate_qvf(path)["valid"]

    def test_structure_symmetry(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "sym",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            symmetry_data={"space_group_number": 1},
        )
        assert validate_qvf(path)["valid"]

    def test_scf_history(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "hist",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            scf_history_data=[{"iter": 0, "energy_eh": -75.0}],
        )
        assert validate_qvf(path)["valid"]

    def test_scf_history_from_result_round_trip(self, tmp_path):
        """The helper-extracted history lands in the archive verbatim."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        result = _stub_result()
        history = scf_history_from_result(result)
        assert [r["iter"] for r in history] == [1, 2, 3]
        # delta_e omitted on the first iteration, present thereafter.
        assert "delta_e" not in history[0]
        assert history[2]["delta_e"] == pytest.approx(-0.083)
        path = write_qvf(
            tmp_path / "hist_auto",
            plan,
            molecule=mol,
            result=result,
            method="rhf",
            basis="sto-3g",
            scf_history_data=history,
        )
        assert validate_qvf(path)["valid"]
        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            sec = next(s for s in manifest["sections"] if s["kind"] == "scf_history")
            payload = json.loads(zf.read(sec["members"]["iterations"]["path"]))
            iters = payload["iterations"]
            assert len(iters) == 3
            assert iters[-1]["energy_eh"] == pytest.approx(-75.983)

    def test_scf_history_from_result_dict_trace(self):
        """Periodic solvers carry dict-shaped trace steps."""
        result = type(
            "PResult",
            (),
            {
                "n_iter": 2,
                "scf_trace": (
                    {"iter": 1, "energy": -10.0, "delta_e": -10.0},
                    {"iter": 2, "energy": -10.5, "delta_e": -0.5},
                ),
            },
        )()
        history = scf_history_from_result(result)
        assert history == [
            {"iter": 1, "energy_eh": -10.0},
            {"iter": 2, "energy_eh": -10.5, "delta_e": -0.5},
        ]

    def test_scf_history_from_result_no_trace(self):
        """No trace → None (writer skips the section)."""
        result = type("R", (), {"converged": True, "energy": -1.0})()
        assert scf_history_from_result(result) is None

    def test_bonds(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "bonds",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            bonds_data=[(0, 1, 0.95), (0, 2, 0.95)],
        )
        assert validate_qvf(path)["valid"]


class TestBandsSegments:
    def test_segment_count(self, tmp_path):
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "bs",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            band_structure=_stub_bands(),
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = [s for s in _read_manifest(zf)["sections"] if s["kind"] == "bands"][0]
            kp = json.loads(zf.read(s["members"]["kpath"]["path"]))
            assert len(kp["segments"]) == 2

    def test_bands_with_projections(self, tmp_path):
        """Bands section with optional projections (fat bands) writes
        and validates."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        bs = _stub_bands()
        n_k, n_b = bs.energies.shape
        # Synthetic projections: 3 channels (atom-l resolved).
        n_ch = 3
        bs.projections = np.random.rand(n_k, n_b, n_ch).astype(np.float64)
        bs.channels = ["O-2s", "O-2p", "H-1s"]

        path = write_qvf(
            tmp_path / "fat_bands",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            band_structure=bs,
        )
        assert validate_qvf(path)["valid"]

        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            sec = [s for s in manifest["sections"] if s["kind"] == "bands"][0]
            assert "projections" in sec["members"]
            pm = sec["members"]["projections"]
            assert pm["dtype"] == "float64"
            assert pm["shape"] == [n_k, n_b, n_ch]
            # Round-trip the projections data.
            raw = zf.read(pm["path"])
            proj_rt = np.frombuffer(raw, dtype=np.float64).reshape(n_k, n_b, n_ch)
            np.testing.assert_allclose(proj_rt, bs.projections)
            # Channels in the kpath metadata.
            kp = json.loads(zf.read(sec["members"]["kpath"]["path"]))
            assert kp["channels"] == ["O-2s", "O-2p", "H-1s"]

    def test_bands_without_projections_no_channels_key(self, tmp_path):
        """Without projections, kpath JSON omits the channels key."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "plain_bands",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            band_structure=_stub_bands(),
        )
        with zipfile.ZipFile(path, "r") as zf:
            sec = [s for s in _read_manifest(zf)["sections"] if s["kind"] == "bands"][0]
            assert "projections" not in sec["members"]
            kp = json.loads(zf.read(sec["members"]["kpath"]["path"]))
            assert "channels" not in kp


class TestGridHelpers:
    @pytest.mark.slow
    def test_density_helper(self, tmp_path):
        from vibeqc import Atom, Molecule
        from vibeqc._vibeqc_core import BasisSet
        from vibeqc.output.formats.qvf import qvf_density_data
        from vibeqc.runner import run_job

        mol = Molecule(
            [
                Atom(8, (0, 0, 0.1173)),
                Atom(1, (0, 1.4315, -0.9314)),
                Atom(1, (0, -1.4315, -0.9314)),
            ],
            charge=0,
            multiplicity=1,
        )
        result = run_job(
            mol,
            basis="sto-3g",
            method="rhf",
            output=tmp_path / "gt",
            citations=False,
            write_population_file=False,
        )
        vol = qvf_density_data(result, BasisSet(mol, "sto-3g"), mol, spacing=0.5)
        assert vol["Electron density"][0].ndim == 3
        assert vol["Electron density"][0].shape[0] > 5

    @pytest.mark.slow
    def test_mo_helper(self, tmp_path):
        from vibeqc import Atom, Molecule
        from vibeqc._vibeqc_core import BasisSet
        from vibeqc.output.formats.qvf import qvf_mo_data
        from vibeqc.runner import run_job

        mol = Molecule(
            [
                Atom(8, (0, 0, 0.1173)),
                Atom(1, (0, 1.4315, -0.9314)),
                Atom(1, (0, -1.4315, -0.9314)),
            ],
            charge=0,
            multiplicity=1,
        )
        result = run_job(
            mol,
            basis="sto-3g",
            method="rhf",
            output=tmp_path / "mt",
            citations=False,
            write_population_file=False,
        )
        mo = qvf_mo_data(result, BasisSet(mol, "sto-3g"), mol, [4], spacing=0.5)
        assert len(mo) == 1
        assert mo[0]["data"].ndim == 3


# ---------------------------------------------------------------------------
# Unit-correctness regression tests
# ---------------------------------------------------------------------------


class TestGridUnits:
    @pytest.mark.slow
    def test_density_grid_origin_is_in_bohr(self, tmp_path):
        """Grid origin in a QVF from qvf_density_data must be in bohr,
        not angstrom.  A water molecule with 4 bohr padding should
        have origin coordinates on the order of -5 to -6 bohr (not -3 Å)."""
        from vibeqc import Atom, Molecule
        from vibeqc._vibeqc_core import BasisSet
        from vibeqc.output.formats.qvf import qvf_density_data, write_qvf
        from vibeqc.output.plan import OutputPlan
        from vibeqc.runner import run_job

        mol = Molecule(
            [
                Atom(8, (0, 0, 0.1173)),
                Atom(1, (0, 1.4315, -0.9314)),
                Atom(1, (0, -1.4315, -0.9314)),
            ],
            charge=0,
            multiplicity=1,
        )
        result = run_job(
            mol,
            basis="sto-3g",
            method="rhf",
            output=tmp_path / "gu",
            citations=False,
            write_population_file=False,
        )
        basis_obj = BasisSet(mol, "sto-3g")
        vol_data = qvf_density_data(result, basis_obj, mol, spacing=0.5, padding=4.0)
        plan = OutputPlan.from_run_job_kwargs(
            output=tmp_path / "gu_qvf",
            method="rhf",
            basis="sto-3g",
            functional=None,
        )
        path = write_qvf(
            tmp_path / "gu_qvf",
            plan,
            molecule=mol,
            result=result,
            method="rhf",
            basis="sto-3g",
            volume_data=vol_data,
        )
        with zipfile.ZipFile(path, "r") as zf:
            s = [
                s
                for s in _read_manifest(zf)["sections"]
                if s["kind"] == "volume.density"
            ][0]
            g = json.loads(zf.read(s["members"]["grid"]["path"]))
            # With 4 bohr padding around an O atom at (0,0,0.1173),
            # origin should be < -3.5 in at least one axis.
            # If the value were in Å it would be ~ -2, which is wrong.
            assert g["origin"][0] < -3.5, (
                f"origin[0]={g['origin'][0]:.3f} is too shallow for "
                f"4-bohr padding; suspect Å conversion"
            )
            # voxel_vectors are per-voxel steps in bohr; the full grid
            # extent is recovered from shape × step by the consumer.
            step_x = abs(g["voxel_vectors"][0][0])
            assert step_x == pytest.approx(0.5)
            assert step_x * (g["shape"][0] - 1) > 7.0


# ---------------------------------------------------------------------------
# Regression — DOS sections coexisting (no shared-path collision).
#
# Both dos.total and dos.projected used to write "dos/energies.bin",
# producing a duplicate zip member when an archive carried both. The
# energies-grid member path is per-section, so each must own its path.
# (Found via examples/vibe_view/showcase_qvf_all_sections.py.)
# ---------------------------------------------------------------------------


class TestDOSCoexistence:
    def _dos_inputs(self):
        energies = np.linspace(-25.0, 15.0, 128)
        dos = np.exp(-0.5 * (energies / 3.0) ** 2)
        pdos = np.vstack([dos, dos * 0.5])
        dos_data = {"energies": energies, "dos": dos, "n_spin": 1}
        pdos_data = {
            "energies": energies,
            "projections": pdos,
            "n_spin": 1,
            "channels": [
                {"atom_index": 0, "symbol": "O", "l": 0, "label": "O 2s"},
                {"atom_index": 0, "symbol": "O", "l": 1, "label": "O 2p"},
            ],
        }
        return dos_data, pdos_data

    def test_total_and_projected_in_one_archive(self, tmp_path, recwarn):
        mol = _stub_molecule()
        plan = _plan(tmp_path, job_kind="periodic_scf")
        dos_data, pdos_data = self._dos_inputs()
        path = write_qvf(
            tmp_path / "dos_both",
            plan,
            molecule=mol,
            method="rhf",
            basis="sto-3g",
            dos_data=dos_data,
            pdos_data=pdos_data,
        )
        # write_qvf already runs the validation gate; assert it explicitly.
        assert validate_qvf(path)["valid"]
        # No duplicate zip member names (the collision symptom).
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            assert len(names) == len(set(names)), "duplicate zip member"
            manifest = _read_manifest(zf)
        kinds = {s["kind"] for s in manifest["sections"]}
        assert {"dos.total", "dos.projected"} <= kinds
        # The two sections must reference distinct energies members.
        paths = {
            s["kind"]: s["members"]["energies"]["path"]
            for s in manifest["sections"]
            if s["kind"] in {"dos.total", "dos.projected"}
        }
        assert paths["dos.total"] != paths["dos.projected"]


# ---------------------------------------------------------------------------
# basis.ao section
# ---------------------------------------------------------------------------


class TestBasisAOSection:
    def test_single_contracted_ao(self, tmp_path):
        """A contracted AO section carries ao_metadata with
        is_primitive=False, is_contracted=True."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        data = np.random.randn(8, 8, 8).astype(np.float32)
        ao_meta = {
            "atom_index": 0,
            "atom_symbol": "O",
            "shell_index": 0,
            "primitive_index": 0,
            "angular_momentum": [0, 0],
            "shell_type": "s",
            "exponent": 3.5,
            "coefficient": 0.42,
            "is_primitive": False,
            "is_contracted": True,
            "ao_index": 0,
        }
        ao_entry = {
            "label": "O s (contracted)  sh=0",
            "data": data,
            "origin": np.zeros(3, dtype=np.float64),
            "span": np.eye(3, dtype=np.float64) * 0.3,
            "ao_metadata": ao_meta,
            "section_id": "ao_O_s_s0_contracted",
        }
        path = write_qvf(
            tmp_path / "basis_ao_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            ao_data=[ao_entry],
        )
        report = validate_qvf(path)
        assert report["valid"], f"Validation errors: {report.get('errors', [])}"
        with zipfile.ZipFile(path, "r") as zf:
            secs = [
                s for s in _read_manifest(zf)["sections"] if s["kind"] == "basis.ao"
            ]
            assert len(secs) == 1
            s = secs[0]
            assert s["id"] == "ao_O_s_s0_contracted"
            assert s["label"] == "O s (contracted)  sh=0"
            # ao_metadata round-trip
            meta = s["ao_metadata"]
            assert meta["atom_index"] == 0
            assert meta["atom_symbol"] == "O"
            assert meta["shell_type"] == "s"
            assert meta["angular_momentum"] == [0, 0]
            assert meta["exponent"] == pytest.approx(3.5)
            assert meta["coefficient"] == pytest.approx(0.42)
            assert meta["is_primitive"] is False
            assert meta["is_contracted"] is True
            assert meta["ao_index"] == 0
            # grid + data members
            dm = s["members"]["data"]
            assert dm["dtype"] == "float32"
            assert dm["format"] == "binary"
            assert dm["shape"] == [8, 8, 8]
            gm = s["members"]["grid"]
            assert gm["format"] == "json"
            grid = json.loads(zf.read(gm["path"]))
            assert grid["shape"] == [8, 8, 8]
            assert "voxel_vectors" in grid
            raw = zf.read(dm["path"])
            vol = np.frombuffer(raw, dtype=np.float32).reshape(8, 8, 8)
            np.testing.assert_allclose(vol.ravel(), data.ravel())

    def test_single_primitive_ao(self, tmp_path):
        """A primitive AO section carries is_primitive=True,
        is_contracted=False."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        data = np.random.randn(6, 6, 6).astype(np.float64)
        ao_meta = {
            "atom_index": 1,
            "atom_symbol": "H",
            "shell_index": 2,
            "primitive_index": 1,
            "angular_momentum": [1, 0],
            "shell_type": "p",
            "exponent": 0.12,
            "coefficient": 0.15,
            "is_primitive": True,
            "is_contracted": False,
            "ao_index": 5,
            "basis_label": "pob-TZVP",
        }
        ao_entry = {
            "label": "H p  sh=2  p1",
            "data": data,
            "origin": np.array([-1.0, -2.0, -3.0], dtype=np.float64),
            "span": np.eye(3, dtype=np.float64) * 0.15,
            "ao_metadata": ao_meta,
            "section_id": "ao_H_p_s2_p1",
        }
        path = write_qvf(
            tmp_path / "basis_ao_prim_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            volume_dtype="float64",
            ao_data=[ao_entry],
        )
        report = validate_qvf(path)
        assert report["valid"], f"Validation errors: {report.get('errors', [])}"
        with zipfile.ZipFile(path, "r") as zf:
            s = [s for s in _read_manifest(zf)["sections"] if s["kind"] == "basis.ao"][
                0
            ]
            meta = s["ao_metadata"]
            assert meta["atom_symbol"] == "H"
            assert meta["is_primitive"] is True
            assert meta["is_contracted"] is False
            assert meta["primitive_index"] == 1
            assert meta["basis_label"] == "pob-TZVP"
            assert meta["angular_momentum"] == [1, 0]
            assert s["members"]["data"]["dtype"] == "float64"

    def test_multiple_mixed_ao_sections(self, tmp_path):
        """A QVF can carry both contracted and primitive AO sections
        simultaneously."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        entries = []
        for i in range(3):
            data = np.random.randn(6, 6, 6).astype(np.float32)
            entries.append(
                {
                    "label": f"AO_{i}",
                    "data": data,
                    "origin": np.zeros(3, dtype=np.float64),
                    "span": np.eye(3, dtype=np.float64) * 0.2,
                    "ao_metadata": {
                        "atom_index": 0,
                        "atom_symbol": "O",
                        "shell_index": i,
                        "primitive_index": 0,
                        "angular_momentum": [i, 0],
                        "shell_type": ["s", "p", "d"][i],
                        "exponent": float(i + 1),
                        "coefficient": 1.0,
                        "is_primitive": False,
                        "is_contracted": True,
                        "ao_index": i,
                    },
                    "section_id": f"ao_test_{i}",
                }
            )
        path = write_qvf(
            tmp_path / "basis_ao_multi_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            ao_data=entries,
        )
        report = validate_qvf(path)
        assert report["valid"], f"Validation errors: {report.get('errors', [])}"
        with zipfile.ZipFile(path, "r") as zf:
            secs = [
                s for s in _read_manifest(zf)["sections"] if s["kind"] == "basis.ao"
            ]
            assert len(secs) == 3
            ids = {s["id"] for s in secs}
            assert ids == {"ao_test_0", "ao_test_1", "ao_test_2"}

    def test_viewer_defaults_per_ao(self, tmp_path):
        """Per-section viewer_defaults hints propagate into the
        manifest and are preserved by the schema."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        data = np.random.randn(5, 5, 5).astype(np.float32)
        ao_entry = {
            "label": "test",
            "data": data,
            "origin": np.zeros(3),
            "span": np.eye(3) * 0.3,
            "ao_metadata": {
                "atom_index": 0,
                "atom_symbol": "O",
                "shell_index": 0,
                "primitive_index": 0,
                "angular_momentum": [0, 0],
                "shell_type": "s",
                "exponent": 3.0,
                "coefficient": 1.0,
                "is_primitive": False,
                "is_contracted": True,
                "ao_index": 0,
            },
            "section_id": "ao_test_hints",
        }
        path = write_qvf(
            tmp_path / "basis_ao_hints_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            ao_data=[ao_entry],
            viewer_defaults={
                "ao_test_hints": {"isovalue": 0.02, "colormap": "RdBu", "opacity": 0.5},
            },
        )
        with zipfile.ZipFile(path, "r") as zf:
            m = _read_manifest(zf)
            vd = m.get("viewer_defaults", {})
            assert "ao_test_hints" in vd
            assert vd["ao_test_hints"]["isovalue"] == 0.02
            assert vd["ao_test_hints"]["colormap"] == "RdBu"
            assert vd["ao_test_hints"]["opacity"] == 0.5

    def test_empty_ao_data_is_noop(self, tmp_path):
        """Empty ao_data list produces no basis.ao sections and a
        valid archive."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)
        path = write_qvf(
            tmp_path / "basis_ao_empty_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            ao_data=[],
        )
        report = validate_qvf(path)
        assert report["valid"], f"Validation errors: {report.get('errors', [])}"
        with zipfile.ZipFile(path, "r") as zf:
            secs = [
                s for s in _read_manifest(zf)["sections"] if s["kind"] == "basis.ao"
            ]
            assert len(secs) == 0


# ---------------------------------------------------------------------------
# Pure-Python AO helpers (no native build needed)
# ---------------------------------------------------------------------------


class TestAOHelpers:
    @pytest.mark.parametrize(
        "l, expected",
        [(0, "s"), (1, "p"), (2, "d"), (3, "f"), (4, "g"), (5, "h"), (6, "l6")],
    )
    def test_l_to_shell_type(self, l, expected):
        assert _l_to_shell_type(l) == expected

    @pytest.mark.parametrize(
        "exponent, expected",
        [
            (0.05, 0.005),
            (0.099, 0.005),
            (0.1, 0.02),
            (0.5, 0.02),
            (0.99, 0.02),
            (1.0, 0.05),
            (5.0, 0.05),
            (9.99, 0.05),
            (10.0, 0.10),
            (100.0, 0.10),
        ],
    )
    def test_ao_isovalue_default(self, exponent, expected):
        assert _ao_isovalue_default(exponent) == pytest.approx(expected)

    def test_ao_label_contracted(self):
        label = _ao_label("O", shell_idx=2, l=1, contracted=True)
        assert "O" in label
        assert "p" in label
        assert "contracted" in label
        assert "sh=2" in label

    def test_ao_label_primitive(self):
        label = _ao_label("H", shell_idx=0, l=0, prim_idx=1, contracted=False)
        assert "H" in label
        assert "s" in label
        assert "sh=0" in label
        assert "p1" in label
        assert "contracted" not in label

    def test_ao_section_id_contracted(self):
        sid = _ao_section_id("O", shell_idx=3, l=2, contracted=True)
        assert sid == "ao_O_d_s3_contracted"

    def test_ao_section_id_primitive(self):
        sid = _ao_section_id("H", shell_idx=1, l=0, prim_idx=0, contracted=False)
        assert sid == "ao_H_s_s1_p0"


# ---------------------------------------------------------------------------
# M24 — localized wavefunction.gto companion section
# ---------------------------------------------------------------------------


class TestWavefunctionLocalized:
    def test_localized_section_emitted(self, tmp_path):
        """When wf_localized_data is present, a second wavefunction.gto
        section with id 'wf_localized' is written."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)

        # Minimal localized wf data — same shape as wf_data but with
        # orbital_kind set to "localized".
        wf_loc = {
            "basis": [{"center": 0, "l": 0, "exponents": [0.5], "coefficients": [1.0]}],
            "mo_metadata": {
                "spin": "restricted",
                "orbital_kind": "localized",
                "energies": [-0.5],
                "occupations": [2.0],
            },
            "mo_coefficients": np.array([[1.0]]),
        }

        path = write_qvf(
            tmp_path / "wf_loc_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            wf_localized_data=wf_loc,
        )
        assert validate_qvf(path)["valid"]

        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            wf_sections = [
                s for s in manifest["sections"] if s["kind"] == "wavefunction.gto"
            ]
            assert len(wf_sections) == 1  # only the localized one
            sec = wf_sections[0]
            assert sec["id"] == "wf_localized"
            assert "mo_metadata" in sec["members"]
            # Verify orbital_kind is preserved in the metadata
            meta = json.loads(zf.read(sec["members"]["mo_metadata"]["path"]))
            assert meta["orbital_kind"] == "localized"

    def test_both_canonical_and_localized_coexist(self, tmp_path):
        """Canonical wf_data and wf_localized_data can coexist in one archive."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)

        basis = [{"center": 0, "l": 0, "exponents": [0.5], "coefficients": [1.0]}]
        wf_canon = {
            "basis": basis,
            "mo_metadata": {
                "spin": "restricted",
                "orbital_kind": "canonical",
                "energies": [-0.5],
                "occupations": [2.0],
            },
            "mo_coefficients": np.array([[1.0]]),
        }
        wf_loc = {
            "basis": basis,
            "mo_metadata": {
                "spin": "restricted",
                "orbital_kind": "localized",
                "energies": [-0.48],
                "occupations": [2.0],
            },
            "mo_coefficients": np.array([[0.98]]),
        }

        path = write_qvf(
            tmp_path / "wf_both_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            wf_data=wf_canon,
            wf_localized_data=wf_loc,
        )
        assert validate_qvf(path)["valid"]

        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            wf_sections = [
                s for s in manifest["sections"] if s["kind"] == "wavefunction.gto"
            ]
            assert len(wf_sections) == 2
            ids = {s["id"] for s in wf_sections}
            assert ids == {"wf", "wf_localized"}
            # Verify the two sections have different coefficients
            for sec in wf_sections:
                meta = json.loads(zf.read(sec["members"]["mo_metadata"]["path"]))
                if sec["id"] == "wf":
                    assert meta["orbital_kind"] == "canonical"
                else:
                    assert meta["orbital_kind"] == "localized"


# ---------------------------------------------------------------------------
# M31 — NTO (Natural Transition Orbitals) writer dispatch
# ---------------------------------------------------------------------------


class TestNTOSections:
    def test_nto_single_state(self, tmp_path):
        """nto_data with one state emits a hole + electron wavefunction.gto pair."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)

        basis = [{"center": 0, "l": 0, "exponents": [0.5], "coefficients": [1.0]}]
        wf_hole = {
            "basis": basis,
            "mo_metadata": {
                "spin": "restricted",
                "energies": [-0.5],
                "occupations": [0.8],
            },
            "mo_coefficients": np.array([[0.9]]),
        }
        wf_elec = {
            "basis": basis,
            "mo_metadata": {
                "spin": "restricted",
                "energies": [0.1],
                "occupations": [0.8],
            },
            "mo_coefficients": np.array([[0.4]]),
        }

        path = write_qvf(
            tmp_path / "nto1_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            nto_data=[
                {
                    "hole": wf_hole,
                    "electron": wf_elec,
                    "state_index": 1,
                    "excitation_energy_ev": 4.5,
                }
            ],
        )
        assert validate_qvf(path)["valid"]

        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            wf_sections = [
                s for s in manifest["sections"] if s["kind"] == "wavefunction.gto"
            ]
            assert len(wf_sections) == 2
            ids = {s["id"] for s in wf_sections}
            assert ids == {"wf_nto_S1_hole", "wf_nto_S1_electron"}
            for sec in wf_sections:
                meta = json.loads(zf.read(sec["members"]["mo_metadata"]["path"]))
                assert meta["orbital_kind"] == "natural"
                assert meta["occupation_semantics"] == "transition_weight"
                assert meta["excitation_energy_ev"] == pytest.approx(4.5)
                assert meta["excitation_index"] == 1

    def test_nto_multi_state(self, tmp_path):
        """nto_data with two states emits four wavefunction.gto sections."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)

        basis = [{"center": 0, "l": 0, "exponents": [0.5], "coefficients": [1.0]}]

        def _make_wf(val):
            return {
                "basis": basis,
                "mo_metadata": {
                    "spin": "restricted",
                    "energies": [val],
                    "occupations": [0.5],
                },
                "mo_coefficients": np.array([[val]]),
            }

        path = write_qvf(
            tmp_path / "nto2_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            nto_data=[
                {
                    "hole": _make_wf(-0.5),
                    "electron": _make_wf(0.1),
                    "state_index": 1,
                    "excitation_energy_ev": 4.5,
                },
                {
                    "hole": _make_wf(-0.4),
                    "electron": _make_wf(0.2),
                    "state_index": 2,
                    "excitation_energy_ev": 5.2,
                },
            ],
        )
        assert validate_qvf(path)["valid"]

        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            wf_sections = [
                s for s in manifest["sections"] if s["kind"] == "wavefunction.gto"
            ]
            assert len(wf_sections) == 4
            ids = {s["id"] for s in wf_sections}
            assert ids == {
                "wf_nto_S1_hole",
                "wf_nto_S1_electron",
                "wf_nto_S2_hole",
                "wf_nto_S2_electron",
            }

    def test_nto_with_canonical_coexists(self, tmp_path):
        """Canonical wf + NTO sections coexist without path collisions."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)

        basis = [{"center": 0, "l": 0, "exponents": [0.5], "coefficients": [1.0]}]
        wf_canon = {
            "basis": basis,
            "mo_metadata": {
                "spin": "restricted",
                "orbital_kind": "canonical",
                "energies": [-0.5],
                "occupations": [2.0],
            },
            "mo_coefficients": np.array([[1.0]]),
        }
        wf_hole = {
            "basis": basis,
            "mo_metadata": {
                "spin": "restricted",
                "energies": [-0.5],
                "occupations": [0.8],
            },
            "mo_coefficients": np.array([[0.9]]),
        }
        wf_elec = {
            "basis": basis,
            "mo_metadata": {
                "spin": "restricted",
                "energies": [0.1],
                "occupations": [0.8],
            },
            "mo_coefficients": np.array([[0.4]]),
        }

        path = write_qvf(
            tmp_path / "nto3_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            wf_data=wf_canon,
            nto_data=[
                {
                    "hole": wf_hole,
                    "electron": wf_elec,
                    "state_index": 1,
                    "excitation_energy_ev": 4.5,
                }
            ],
        )
        assert validate_qvf(path)["valid"]

        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            wf_sections = [
                s for s in manifest["sections"] if s["kind"] == "wavefunction.gto"
            ]
            assert len(wf_sections) == 3  # canonical + hole + electron


# ---------------------------------------------------------------------------
# M25 — bond_orders section
# ---------------------------------------------------------------------------


class TestBondOrdersSection:
    def test_bond_orders_round_trip(self, tmp_path):
        """bond_orders section writes and validates."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)

        pairs = [
            {
                "i": 0,
                "j": 1,
                "order": 0.98,
                "distance_ang": 1.42,
                "symbol_i": "O",
                "symbol_j": "H",
            },
            {
                "i": 0,
                "j": 2,
                "order": 0.95,
                "distance_ang": 1.42,
                "symbol_i": "O",
                "symbol_j": "H",
            },
        ]

        path = write_qvf(
            tmp_path / "bo_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            bond_orders_data={"method": "mayer", "pairs": pairs},
        )
        assert validate_qvf(path)["valid"]

        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            bo_secs = [s for s in manifest["sections"] if s["kind"] == "bond_orders"]
            assert len(bo_secs) == 1
            sec = bo_secs[0]
            assert sec["id"] == "bond_orders"
            member = sec["members"]["bond_orders"]
            assert member["format"] == "json"
            payload = json.loads(zf.read(member["path"]))
            assert payload["method"] == "mayer"
            assert len(payload["pairs"]) == 2
            assert payload["pairs"][0]["i"] == 0
            assert payload["pairs"][0]["order"] == pytest.approx(0.98)
            assert payload["pairs"][0]["distance_ang"] == pytest.approx(1.42)

    def test_bond_orders_minimal(self, tmp_path):
        """Minimal pairs (only i, j, order) validate fine."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)

        path = write_qvf(
            tmp_path / "bo_min_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            bond_orders_data={
                "method": "wiberg",
                "pairs": [{"i": 0, "j": 1, "order": 0.88}],
            },
        )
        assert validate_qvf(path)["valid"]

        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            bo_sec = next(s for s in manifest["sections"] if s["kind"] == "bond_orders")
            payload = json.loads(zf.read(bo_sec["members"]["bond_orders"]["path"]))
            assert payload["method"] == "wiberg"
            assert payload["pairs"][0]["order"] == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# M27 — topology.qtaim section (dormant writer, compute module not yet)
# ---------------------------------------------------------------------------


class TestTopologyQTAIMSection:
    def test_qtaim_round_trip(self, tmp_path):
        """topology.qtaim section writes critical points and validates."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)

        cp = [
            {
                "type": "bcp",
                "position": [0.71, 0.0, 0.0],
                "rho": 0.26,
                "laplacian": -0.54,
                "ellipticity": 0.12,
                "atom_pair": [0, 1],
            },
            {
                "type": "rcp",
                "position": [0.0, 0.0, 0.5],
                "rho": 0.08,
                "laplacian": 0.12,
            },
        ]
        bond_paths = [
            {
                "atoms": [0, 1],
                "path": [[0.0, 0.0, 0.0], [0.1, 0.05, 0.0], [0.71, 0.0, 0.0]],
            }
        ]

        path = write_qvf(
            tmp_path / "qtaim_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            qtaim_data={
                "critical_points": cp,
                "bond_paths": bond_paths,
            },
        )
        assert validate_qvf(path)["valid"]

        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            secs = [s for s in manifest["sections"] if s["kind"] == "topology.qtaim"]
            assert len(secs) == 1
            sec = secs[0]
            assert sec["id"] == "qtaim"
            member = sec["members"]["critical_points"]
            assert member["format"] == "json"
            payload = json.loads(zf.read(member["path"]))
            assert len(payload["points"]) == 2
            assert payload["points"][0]["type"] == "bcp"
            assert payload["points"][0]["rho"] == pytest.approx(0.26)
            assert payload["points"][0]["atom_pair"] == [0, 1]
            assert len(payload["bond_paths"]) == 1
            assert payload["bond_paths"][0]["atoms"] == [0, 1]

    def test_qtaim_minimal_no_bond_paths(self, tmp_path):
        """QTAIM section without bond_paths validates fine."""
        mol = _stub_molecule()
        plan = _plan(tmp_path)

        path = write_qvf(
            tmp_path / "qtaim_min_qvf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            qtaim_data={
                "critical_points": [
                    {
                        "type": "ccp",
                        "position": [0, 0, 0],
                        "rho": 0.01,
                        "laplacian": 0.05,
                    },
                ],
            },
        )
        assert validate_qvf(path)["valid"]

        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            sec = next(s for s in manifest["sections"] if s["kind"] == "topology.qtaim")
            payload = json.loads(zf.read(sec["members"]["critical_points"]["path"]))
            assert "bond_paths" not in payload
            assert payload["points"][0]["type"] == "ccp"


# ---------------------------------------------------------------------------
# BUG 75 — unrestricted wavefunction.gto validation fixtures
# ---------------------------------------------------------------------------


class TestWavefunctionUnrestricted:
    """Unrestricted (UHF/UKS) wavefunction.gto uses mo_coefficients_alpha
    and mo_coefficients_beta, not a single mo_coefficients member."""

    @staticmethod
    def _basis_shells():
        return [
            {"center": 0, "l": 0, "exponents": [0.5], "coefficients": [[1.0]]},
            {"center": 1, "l": 0, "exponents": [0.4], "coefficients": [[1.0]]},
        ]

    @staticmethod
    def _stub_mol():
        return _stub_molecule()

    def test_unrestricted_wf_emits_alpha_beta_members(self, tmp_path):
        """UHF wavefunction.gto must write mo_coefficients_alpha.dat
        and mo_coefficients_beta.dat, not mo_coefficients.dat."""
        plan = _plan(tmp_path, method="uhf")
        shells = self._basis_shells()
        mol = self._stub_mol()
        wf_data = {
            "basis": shells,
            "structure_ref": "structure",
            "pure": True,
            "n_ao": 2,
            "mo_metadata": {
                "n_ao": 2,
                "spin": "unrestricted",
                "orbital_kind": "canonical",
                "alpha": {
                    "n_mo": 2,
                    "energies": [-0.5, 0.2],
                    "occupations": [1.0, 0.0],
                },
                "beta": {
                    "n_mo": 2,
                    "energies": [-0.4, 0.3],
                    "occupations": [1.0, 0.0],
                },
            },
            "mo_coefficients_alpha": np.array([[0.9, 0.1], [0.1, 0.9]]),
            "mo_coefficients_beta": np.array([[0.8, 0.2], [0.2, 0.8]]),
        }
        path = write_qvf(
            tmp_path / "uhf_wf",
            plan,
            molecule=mol,
            result=_stub_result(),
            method="uhf",
            basis="sto-3g",
            wf_data=wf_data,
            atomic=True,
        )
        assert validate_qvf(path)["valid"]

        with zipfile.ZipFile(path, "r") as zf:
            manifest = _read_manifest(zf)
            wf_sec = next(
                s for s in manifest["sections"] if s["kind"] == "wavefunction.gto"
            )
            members = wf_sec["members"]
            # Unrestricted must use alpha/beta members, not mo_coefficients.
            assert "mo_coefficients" not in members
            assert "mo_coefficients_alpha" in members
            assert "mo_coefficients_beta" in members
            # Verify alpha coefficient shape.
            alpha_raw = zf.read(members["mo_coefficients_alpha"]["path"])
            alpha_coeff = np.frombuffer(alpha_raw, dtype=np.float64)
            assert alpha_coeff.size == 4  # 2x2
            # Verify beta coefficient shape.
            beta_raw = zf.read(members["mo_coefficients_beta"]["path"])
            beta_coeff = np.frombuffer(beta_raw, dtype=np.float64)
            assert beta_coeff.size == 4
