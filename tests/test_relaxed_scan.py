"""Tests for the relaxed coordinate scan driver (molecular + periodic)."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import (
    Atom,
    Molecule,
    PeriodicSystem,
    relaxed_scan,
    relaxed_scan_2d,
)
from vibeqc.scan import (
    ScanResult,
    ScanResult2D,
    _measure_angle,
    _measure_bond,
    _measure_dihedral,
    _set_angle,
    _set_bond,
    _set_dihedral,
)
from vibeqc.progress import ProgressLogger


def _h2o() -> Molecule:
    """Bent H2O near equilibrium (bohr)."""
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [1.81, 0.0, 0.0]),
            Atom(1, [-0.46, 1.75, 0.0]),
        ],
        charge=0,
        multiplicity=1,
    )


def _h2_in_box() -> PeriodicSystem:
    """H2 in a small cubic box — Γ-only RHF/STO-3G is < 30 s on a laptop."""
    L = np.eye(3) * 8.0
    atoms = [Atom(1, [4.0, 4.0, 4.0]), Atom(1, [5.4, 4.0, 4.0])]
    return PeriodicSystem(3, L, atoms, charge=0, multiplicity=1)


# ── geometry mutator unit tests ──────────────────────────────────────────


class TestGeometryMutators:
    def test_set_bond_hits_target_exactly(self):
        p = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [3.0, 1.0, 0.0]])
        p2 = _set_bond(p, 0, 1, target=2.0)
        assert _measure_bond(p2, 0, 1) == pytest.approx(2.0, abs=1e-12)
        # Atom 0 untouched, atom 2 untouched.
        assert np.allclose(p2[0], p[0])
        assert np.allclose(p2[2], p[2])

    def test_set_bond_preserves_direction(self):
        p = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        p2 = _set_bond(p, 0, 1, target=10.0)
        # |v| was 5; now 10; should still point along (3, 4, 0) / 5.
        assert np.allclose(p2[1], np.array([6.0, 8.0, 0.0]))

    def test_set_angle_hits_target_exactly(self):
        p = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        target = math.radians(120.0)
        p2 = _set_angle(p, 0, 1, 2, target=target)
        assert _measure_angle(p2, 0, 1, 2) == pytest.approx(target, abs=1e-10)
        # Atoms 0 and 1 unchanged.
        assert np.allclose(p2[0], p[0])
        assert np.allclose(p2[1], p[1])

    def test_set_dihedral_hits_target_exactly(self):
        # Non-collinear i-j-k skeleton so the i-j-k and j-k-l planes
        # are well-defined.
        p = np.array(
            [
                [0.0, 1.0, 0.0],   # i
                [0.0, 0.0, 0.0],   # j
                [1.0, 0.0, 0.0],   # k
                [1.0, 1.0, 0.5],   # l
            ]
        )
        for target_deg in (30.0, 90.0, 150.0):
            target = math.radians(target_deg)
            p2 = _set_dihedral(p, 0, 1, 2, 3, target=target)
            got = _measure_dihedral(p2, 0, 1, 2, 3)
            diff = (got - target + math.pi) % (2 * math.pi) - math.pi
            assert abs(diff) < 1e-10, (
                f"target={target_deg}°: got {math.degrees(got):.4f}°, "
                f"diff {math.degrees(diff):.4e}°"
            )


# ── molecular relaxed scan ───────────────────────────────────────────────


class TestMolecularScan:
    def test_progress_uses_caller_logger_without_stdout(self, monkeypatch, capsys):
        import vibeqc.scan as scan_module

        stream = io.StringIO()
        logger = ProgressLogger(stream=stream, verbose=2)
        seen = {}

        def fake_step(
            seed,
            _basis,
            _method,
            _functional,
            _frozen,
            _max_iter,
            _conv_tol,
            progress,
            _kwargs,
        ):
            seen["progress"] = progress
            return seed, -75.0, True

        monkeypatch.setattr(scan_module, "_scan_step_molecular", fake_step)
        relaxed_scan(
            _h2o(),
            "sto-3g",
            coordinate=("bond", 0, 1),
            values=[1.8],
            progress=logger,
        )

        assert capsys.readouterr().out == ""
        assert seen["progress"] is logger
        assert "scan point 1/1" in stream.getvalue()

    def test_h2o_oh_stretch_minimum_at_known_bond(self):
        """O-H stretch should bottom out near the STO-3G equilibrium."""
        h2o = _h2o()
        bond_values = np.linspace(1.6, 2.4, 5)  # bohr
        result = relaxed_scan(
            h2o,
            "sto-3g",
            coordinate=("bond", 0, 1),
            values=bond_values,
            method="RHF",
            max_iter=20,
            conv_tol_grad=5e-4,
        )
        assert isinstance(result, ScanResult)
        assert result.values.shape == (5,)
        assert result.energies.shape == (5,)
        assert len(result.geometries) == 5
        assert all(result.converged_flags)
        # Minimum-energy point should be the second one (1.8 bohr),
        # which sits closest to the STO-3G O-H equilibrium ~1.79.
        i_min = int(np.argmin(result.energies))
        assert i_min == 1, (
            f"min at i={i_min} (value={result.values[i_min]:.2f}); "
            f"energies={result.energies}"
        )

    def test_relaxed_scan_holds_constraint_exactly(self):
        """At each scan point the constrained bond hits its target."""
        h2o = _h2o()
        targets = np.array([1.6, 1.9, 2.2])
        result = relaxed_scan(
            h2o,
            "sto-3g",
            coordinate=("bond", 0, 1),
            values=targets,
            method="RHF",
            max_iter=15,
        )
        for i, target in enumerate(targets):
            mol = result.geometries[i]
            r = float(
                np.linalg.norm(
                    np.array(list(mol.atoms[1].xyz))
                    - np.array(list(mol.atoms[0].xyz))
                )
            )
            assert r == pytest.approx(float(target), abs=1e-6), (
                f"point {i}: bond drifted to {r:.6f}, expected {target:.6f}"
            )

    def test_scan_writes_qvf(self, tmp_path):
        """``ScanResult.write_qvf`` produces a file that passes the
        producer-side QVF validator (which ``write_qvf`` runs before
        returning)."""
        h2o = _h2o()
        result = relaxed_scan(
            h2o,
            "sto-3g",
            coordinate=("bond", 0, 1),
            values=[1.6, 1.8, 2.0, 2.2],
            method="RHF",
            max_iter=10,
        )
        qvf_path = result.write_qvf(tmp_path / "oh_stretch")
        assert qvf_path.exists()
        assert qvf_path.suffix == ".qvf"
        assert qvf_path.stat().st_size > 0

    def _scan_metadata(self, qvf_path):
        """Helper: read the reaction.path section metadata JSON."""
        import json
        import zipfile

        with zipfile.ZipFile(qvf_path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            rxn = next(
                s for s in mf["sections"] if s["kind"] == "reaction.path"
            )
            return json.loads(zf.read(rxn["members"]["metadata"]["path"]))

    def test_scan_qvf_carries_coordinate_label_unit(self, tmp_path):
        """W0: the reaction.path metadata carries a human-readable
        coordinate label + unit so the viewer can label the x-axis."""
        h2o = _h2o()
        result = relaxed_scan(
            h2o,
            "sto-3g",
            coordinate=("bond", 0, 1),
            values=[1.6, 1.8, 2.0],
            method="RHF",
            max_iter=10,
        )
        meta = self._scan_metadata(result.write_qvf(tmp_path / "lbl"))
        assert meta["reaction_coordinate_label"] == "bond 0-1"
        assert meta["reaction_coordinate_unit"] == "bohr"

    def test_transition_state_frames_tag_waypoint(self, tmp_path):
        """W3: a caller-verified TS frame is tagged transition_state,
        and the generic 'scan max' point is suppressed at that frame
        (no duplicate waypoint)."""
        h2o = _h2o()
        result = relaxed_scan(
            h2o,
            "sto-3g",
            coordinate=("bond", 0, 1),
            values=[1.6, 1.8, 2.0, 2.2, 2.4],
            method="RHF",
            max_iter=10,
        )
        # Pick an interior frame and claim it as a verified TS.
        i_max = int(np.argmax(result.energies))
        ts_frame = i_max if 0 < i_max < 4 else 2
        meta = self._scan_metadata(
            result.write_qvf(tmp_path / "ts", transition_state_frames=[ts_frame])
        )
        wps = meta["waypoints"]
        ts = [w for w in wps if w["kind"] == "transition_state"]
        assert len(ts) == 1
        assert ts[0]["frame_index"] == ts_frame
        # No generic 'point' waypoint sitting on the same frame.
        assert not any(
            w["kind"] == "point" and w["frame_index"] == ts_frame for w in wps
        )

    def test_transition_state_frames_out_of_range_rejected(self, tmp_path):
        h2o = _h2o()
        result = relaxed_scan(
            h2o,
            "sto-3g",
            coordinate=("bond", 0, 1),
            values=[1.6, 1.8, 2.0],
            method="RHF",
            max_iter=10,
        )
        with pytest.raises(ValueError, match="out of range"):
            result.write_qvf(tmp_path / "bad", transition_state_frames=[99])

    def test_rejects_unknown_coordinate_kind(self):
        h2o = _h2o()
        with pytest.raises(ValueError, match="unknown coordinate kind"):
            relaxed_scan(
                h2o,
                "sto-3g",
                coordinate=("rumple", 0, 1),  # type: ignore[arg-type]
                values=[1.6, 1.8],
                method="RHF",
            )

    def test_rejects_out_of_range_atom_index(self):
        h2o = _h2o()
        with pytest.raises(ValueError, match="out of range"):
            relaxed_scan(
                h2o,
                "sto-3g",
                coordinate=("bond", 0, 99),
                values=[1.6, 1.8],
                method="RHF",
            )


# ── per-frame volumes (W1) ───────────────────────────────────────────────


class TestPerFrameVolumes:
    def _read_meta_and_members(self, qvf_path):
        import json
        import zipfile

        with zipfile.ZipFile(qvf_path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            rxn = next(s for s in mf["sections"] if s["kind"] == "reaction.path")
            meta = json.loads(zf.read(rxn["members"]["metadata"]["path"]))
            return meta, rxn["members"]

    def test_low_level_frame_volumes_round_trip(self, tmp_path):
        """write_reaction_path_qvf accepts precomputed per-frame volumes,
        the archive validates, and the 4D member + grid round-trip."""
        from vibeqc.output.formats.qvf import (
            validate_qvf,
            write_reaction_path_qvf,
        )

        frames = [_h2o(), _h2o(), _h2o()]
        vol = np.linspace(0, 1, 3 * 4 * 4 * 4, dtype=np.float32).reshape(3, 4, 4, 4)
        grid = {
            "origin": [-2.0, -2.0, -2.0],
            "voxel_vectors": [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
            "shape": [4, 4, 4],
        }
        wps = [{"frame_index": 0, "label": "start", "kind": "reactant"}]
        qvf_path = write_reaction_path_qvf(
            tmp_path / "vols",
            frames=frames,
            energies=[0.0, -0.5, -1.0],
            waypoints=wps,
            frame_volumes=vol,
            volume_grid=grid,
            volume_frame_index=[0, 1, 2],
            volume_label="Electron density",
            method="RHF",
            basis="sto-3g",
        )
        report = validate_qvf(qvf_path)
        assert report["valid"], report["errors"]
        meta, members = self._read_meta_and_members(qvf_path)
        assert "frame_volumes" in members
        assert members["frame_volumes"]["shape"] == [3, 4, 4, 4]
        assert "volume_grid" in members
        assert meta["volume_frame_index"] == [0, 1, 2]
        assert meta["volume_label"] == "Electron density"

    def test_frame_volumes_grid_mismatch_rejected(self, tmp_path):
        from vibeqc.output.formats.qvf import write_reaction_path_qvf

        frames = [_h2o(), _h2o()]
        vol = np.zeros((2, 4, 4, 4), dtype=np.float32)
        grid = {
            "origin": [0, 0, 0],
            "voxel_vectors": [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
            "shape": [5, 5, 5],  # disagrees with vol's 4×4×4
        }
        wps = [{"frame_index": 0, "label": "start", "kind": "reactant"}]
        with pytest.raises(ValueError, match="grid dims"):
            write_reaction_path_qvf(
                tmp_path / "bad",
                frames=frames,
                energies=[0.0, -1.0],
                waypoints=wps,
                frame_volumes=vol,
                volume_grid=grid,
                method="RHF",
                basis="sto-3g",
            )

    def test_scan_emit_volumes_every_molecular(self, tmp_path):
        """A molecular scan with emit_volumes_every attaches a per-frame
        density grid that validates + round-trips."""
        from vibeqc.output.formats.qvf import validate_qvf

        h2 = Molecule(
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.4, 0.0, 0.0])],
            charge=0,
            multiplicity=1,
        )
        result = relaxed_scan(
            h2,
            "sto-3g",
            coordinate=("bond", 0, 1),
            values=[1.2, 1.4, 1.6],
            method="RHF",
            max_iter=10,
        )
        qvf_path = result.write_qvf(
            tmp_path / "h2_vols",
            emit_volumes_every=1,
            volume_spacing=0.6,  # coarse grid keeps the test fast
        )
        report = validate_qvf(qvf_path)
        assert report["valid"], report["errors"]
        meta, members = self._read_meta_and_members(qvf_path)
        assert members["frame_volumes"]["shape"][0] == 3  # 3 emitted frames
        assert meta["volume_frame_index"] == [0, 1, 2]

    def test_scan_emit_volumes_periodic_raises(self, tmp_path):
        sys0 = _h2_in_box()
        result = relaxed_scan(
            sys0,
            "sto-3g",
            coordinate=("bond", 0, 1),
            values=[1.2, 1.4],
            method="RHF",
            kpoints=(1, 1, 1),
            max_iter=5,
            cutoff_bohr=4.0,
        )
        with pytest.raises(NotImplementedError, match="molecular scans"):
            result.write_qvf(tmp_path / "per_vols", emit_volumes_every=1)


# ── 2D relaxed scan (W2) ─────────────────────────────────────────────────


class TestScan2D:
    def test_relaxed_scan_2d_returns_grid(self):
        h2o = _h2o()
        res = relaxed_scan_2d(
            h2o,
            "sto-3g",
            ("bond", 0, 1),
            [1.7, 1.9],
            ("bond", 0, 2),
            [1.7, 1.9, 2.1],
            method="RHF",
            max_iter=10,
        )
        assert isinstance(res, ScanResult2D)
        assert res.energies.shape == (2, 3)
        assert res.converged_flags.shape == (2, 3)
        assert len(res.geometries) == 2
        assert len(res.geometries[0]) == 3
        assert np.all(np.isfinite(res.energies))

    def test_scan_2d_writes_scan_surface_qvf(self, tmp_path):
        import json
        import zipfile

        from vibeqc.output.formats.qvf import validate_qvf

        h2o = _h2o()
        res = relaxed_scan_2d(
            h2o,
            "sto-3g",
            ("bond", 0, 1),
            [1.7, 1.9],
            ("angle", 1, 0, 2),
            [1.8, 2.0],
            method="RHF",
            max_iter=10,
        )
        qvf_path = res.write_qvf(tmp_path / "surf")
        report = validate_qvf(qvf_path)
        assert report["valid"], report["errors"]
        with zipfile.ZipFile(qvf_path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            sec = next(s for s in mf["sections"] if s["kind"] == "scan.surface")
            assert sec["members"]["energies"]["shape"] == [2, 2]
            assert sec["members"]["axis_a"]["shape"] == [2]
            assert sec["members"]["axis_b"]["shape"] == [2]
            assert "geometries" in sec["members"]
            meta = json.loads(zf.read(sec["members"]["metadata"]["path"]))
            assert meta["coordinate_a_label"] == "bond 0-1"
            assert meta["coordinate_a_unit"] == "bohr"
            assert meta["coordinate_b_label"].startswith("angle")
            assert meta["coordinate_b_unit"] == "rad"
            assert meta["shape"] == [2, 2]

    def test_scan_2d_qvf_without_geometries(self, tmp_path):
        import json
        import zipfile

        from vibeqc.output.formats.qvf import validate_qvf

        h2o = _h2o()
        res = relaxed_scan_2d(
            h2o,
            "sto-3g",
            ("bond", 0, 1),
            [1.7, 1.9],
            ("bond", 0, 2),
            [1.7, 1.9],
            method="RHF",
            max_iter=10,
        )
        qvf_path = res.write_qvf(tmp_path / "surf_nogeo", include_geometries=False)
        assert validate_qvf(qvf_path)["valid"]
        with zipfile.ZipFile(qvf_path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            sec = next(s for s in mf["sections"] if s["kind"] == "scan.surface")
            assert "geometries" not in sec["members"]


# ── periodic relaxed scan ────────────────────────────────────────────────


class TestPeriodicScan:
    @pytest.mark.parametrize(
        "kpoints_factory",
        [
            lambda sysp: vq.monkhorst_pack(sysp, [1, 1, 1]),
            lambda sysp: vq.KPoints.gamma(sysp),
        ],
        ids=["native-bloch-kmesh", "kpoints-object"],
    )
    def test_periodic_scan_accepts_materialized_kmesh(
        self, monkeypatch, kpoints_factory
    ):
        """Periodic relaxed scans must preserve object-style BIPOLE k-meshes."""
        import vibeqc.scan as scan_mod

        sys0 = _h2_in_box()
        user_kpoints = kpoints_factory(sys0)
        expected = vq.as_bloch_kmesh(user_kpoints)
        captured = []

        def fake_scan_step_periodic(
            system,
            basis,
            method,
            functional,
            kmesh,
            freeze_indices,
            max_iter,
            conv_tol_grad,
            extra_kwargs,
        ):
            captured.append(kmesh)
            return system, -1.0, True

        monkeypatch.setattr(
            scan_mod,
            "_scan_step_periodic",
            fake_scan_step_periodic,
        )

        result = relaxed_scan(
            sys0,
            "sto-3g",
            coordinate=("bond", 0, 1),
            values=[1.2, 1.4],
            method="RHF",
            kpoints=user_kpoints,
            max_iter=1,
        )

        assert isinstance(result, ScanResult)
        assert len(captured) == 2
        for kmesh in captured:
            np.testing.assert_allclose(
                np.asarray(kmesh.kpoints, dtype=float),
                np.asarray(expected.kpoints, dtype=float),
            )
            np.testing.assert_allclose(
                np.asarray(kmesh.weights, dtype=float),
                np.asarray(expected.weights, dtype=float),
            )
        if isinstance(user_kpoints, vq.BlochKMesh):
            assert all(kmesh is user_kpoints for kmesh in captured)

    def test_h2_box_bond_scan_returns_valid_curve(self):
        """Periodic RHF/STO-3G H2 in an 8 bohr box — Γ-only."""
        sys0 = _h2_in_box()
        targets = np.array([1.2, 1.4, 1.6])
        result = relaxed_scan(
            sys0,
            "sto-3g",
            coordinate=("bond", 0, 1),
            values=targets,
            method="RHF",
            kpoints=(1, 1, 1),
            max_iter=5,
            cutoff_bohr=4.0,
        )
        assert isinstance(result, ScanResult)
        assert result.values.shape == (3,)
        assert result.energies.shape == (3,)
        assert len(result.geometries) == 3
        # All energies finite + monotonically rising past dissociation onset.
        assert np.all(np.isfinite(result.energies))
        # Equilibrium for H2 is ~1.4 bohr (HF/STO-3G). The 1.4 point
        # should be the minimum among the three.
        assert int(np.argmin(result.energies)) == 1

    def test_periodic_scan_emits_v1_qvf_with_lattice(self, tmp_path):
        """A periodic relaxed scan must round-trip through write_qvf as
        a QVF v2 archive — i.e. the PeriodicSystem frames carry their
        lattice + dim through to the reaction.path section. Before
        QVF Inc C, ScanResult.write_qvf converted PeriodicSystem to
        unit_cell_molecule() and silently dropped the lattice — this
        test pins the fix."""
        import json
        import zipfile

        from vibeqc.output.formats.qvf import (
            QVF_FORMAT_VERSION,
            validate_qvf,
        )

        sys0 = _h2_in_box()
        result = relaxed_scan(
            sys0,
            "sto-3g",
            coordinate=("bond", 0, 1),
            values=np.array([1.2, 1.4, 1.6]),
            method="RHF",
            kpoints=(1, 1, 1),
            max_iter=5,
            cutoff_bohr=4.0,
        )
        qvf_path = result.write_qvf(tmp_path / "periodic_scan")
        report = validate_qvf(qvf_path)
        assert report["valid"], report["errors"]
        with zipfile.ZipFile(qvf_path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            # Ruling 2026-07-10: periodic archives stay v1; `lattice` marks them.
            assert mf["qvf_version"] == QVF_FORMAT_VERSION
            rxn = next(
                s for s in mf["sections"] if s["kind"] == "reaction.path"
            )
            assert "lattice" in rxn["members"]
            lat_member = rxn["members"]["lattice"]
            assert lat_member["dtype"] == "float64"
            assert lat_member["shape"] == [3, 3]
            meta = json.loads(zf.read(rxn["members"]["metadata"]["path"]))
            # The h2-in-box fixture is dim=3, so the metadata should
            # carry a scalar `dim` of 3.
            assert meta["dim"] == 3
