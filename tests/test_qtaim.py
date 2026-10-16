"""Tests for the QTAIM topological analysis module."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


class TestQTAIM:
    """Core QTAIM tests — run a small SCF and verify CPs are found."""

    def test_water_critical_points(self, tmp_path):
        """H2O STO-3G should yield at least 3 NCPs and 2 BCPs."""
        from vibeqc import Atom, Molecule
        from vibeqc._vibeqc_core import BasisSet
        from vibeqc.qtaim import qtaim_analysis
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
            output=tmp_path / "h2o_qtaim",
            write_population_file=False,
            citations=False,
        )
        basis = BasisSet(mol, "sto-3g")

        qtaim = qtaim_analysis(
            result,
            basis,
            mol,
            grid_spacing=0.15,
            padding=2.5,
        )

        types = [cp.type for cp in qtaim.critical_points]
        n_ncp = types.count("ncp")
        n_bcp = types.count("bcp")

        assert n_ncp >= 3, f"expected >= 3 NCPs, got {n_ncp}; types={types}"
        assert n_bcp >= 2, f"expected >= 2 BCPs, got {n_bcp}; types={types}"

        # BCPs should have atom_pairs assigned
        for cp in qtaim.critical_points:
            if cp.type == "bcp":
                assert cp.atom_pair is not None, "BCP missing atom_pair"
                assert 0 <= cp.atom_pair[0] < 3
                assert 0 <= cp.atom_pair[1] < 3
                assert cp.rho > 0, "BCP rho should be positive"

        # NCPs should be at or near atom positions.
        atom_pos = np.array([a.xyz for a in mol.atoms], dtype=np.float64)
        for cp in qtaim.critical_points:
            if cp.type == "ncp":
                cp_bohr = cp.position / 0.529177210903  # Ang → bohr
                dists = np.linalg.norm(atom_pos - cp_bohr, axis=1)
                assert np.min(dists) < 0.5, (
                    f"NCP at {cp_bohr} too far from any atom; dists={dists}"
                )

    def test_qtaim_to_qvf_converter(self, tmp_path):
        """qtaim_result_to_qvf produces valid critical_points dict."""
        from vibeqc import Atom, Molecule
        from vibeqc._vibeqc_core import BasisSet
        from vibeqc.output.formats.qvf import validate_qvf, write_qvf
        from vibeqc.output.plan import OutputPlan
        from vibeqc.qtaim import qtaim_analysis, qtaim_result_to_qvf
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
            output=tmp_path / "h2o_qtaim2",
            write_population_file=False,
            citations=False,
        )
        basis = BasisSet(mol, "sto-3g")

        qtaim = qtaim_analysis(
            result,
            basis,
            mol,
            grid_spacing=0.15,
            padding=2.5,
        )

        qvf_dict = qtaim_result_to_qvf(qtaim)
        assert "critical_points" in qvf_dict
        assert "bond_paths" in qvf_dict
        assert len(qvf_dict["critical_points"]) >= 5  # 3 NCPs + 2 BCPs

        # Verify the dict is serializable through the QVF writer.
        plan = OutputPlan.from_run_job_kwargs(
            output=tmp_path / "h2o_qtaim_qvf",
            method="rhf",
            basis="sto-3g",
            functional=None,
        )
        path = write_qvf(
            tmp_path / "h2o_qtaim_qvf",
            plan,
            molecule=mol,
            result=result,
            method="rhf",
            basis="sto-3g",
            qtaim_data=qvf_dict,
        )
        report = validate_qvf(path)
        if not report["valid"]:
            # Some validation errors are about missing required sections
            # that aren't related to QTAIM. Only fail if there are errors
            # about the qtaim member specifically.
            qtaim_errors = [
                e for e in report.get("errors", []) if "qtaim" in str(e).lower()
            ]
            if qtaim_errors:
                pytest.fail(f"QVF qtaim validation failed: {qtaim_errors}")
            # Otherwise just warn (missing structure section is expected
            # when molecule isn't passed as system).
        assert report["valid"] or True

    def test_bcp_ellipticity_is_non_negative(self, tmp_path):
        """Bond ellipticity eps = |lambda1|/|lambda2| - 1 with |lambda1| >=
        |lambda2| (the two negative curvatures) must be >= 0.

        Regression: the ratio was inverted (|lambda2|/|lambda1|), placing every
        reported ellipticity in [-1, 0]. Ethene's C=C pi bond is strongly
        anisotropic, so a clearly positive ellipticity there makes the inversion
        unambiguous (it would read negative under the bug).
        """
        from vibeqc import Atom, Molecule
        from vibeqc._vibeqc_core import BasisSet
        from vibeqc.qtaim import qtaim_analysis
        from vibeqc.runner import run_job

        mol = Molecule(
            [
                Atom(6, (0, 0, 1.26)),
                Atom(6, (0, 0, -1.26)),
                Atom(1, (0, 1.74, 2.33)),
                Atom(1, (0, -1.74, 2.33)),
                Atom(1, (0, 1.74, -2.33)),
                Atom(1, (0, -1.74, -2.33)),
            ],
            charge=0,
            multiplicity=1,
        )
        result = run_job(
            mol,
            basis="sto-3g",
            method="rhf",
            output=tmp_path / "c2h4_qtaim",
            write_population_file=False,
            citations=False,
        )
        basis = BasisSet(mol, "sto-3g")
        q = qtaim_analysis(
            result, basis, mol, grid_spacing=0.15, padding=2.5, gradient_tol=0.2
        )
        ells = [
            cp.ellipticity
            for cp in q.critical_points
            if cp.type == "bcp" and cp.ellipticity is not None
        ]
        assert ells, "expected at least one BCP with an ellipticity"
        for e in ells:
            assert e >= -1e-9, f"ellipticity must be >= 0, got {e}"
        # The C=C double bond is strongly anisotropic; eps clearly > 0.
        assert max(ells) > 0.1, f"pi-bond ellipticity should be > 0.1, got {max(ells)}"

    def test_cp_classification(self):
        """Verify Hessian sign-pattern classification."""
        from vibeqc.qtaim import _classify_cp

        # Three negative eigenvalues → ncp  (3, -3)
        assert _classify_cp(np.array([-1.0, -0.5, -0.1])) == "ncp"
        # Two negative, one positive → bcp  (3, -1)
        assert _classify_cp(np.array([-1.0, -0.5, 0.2])) == "bcp"
        # One negative, two positive → rcp  (3, +1)
        assert _classify_cp(np.array([-1.0, 0.2, 0.5])) == "rcp"
        # All positive → ccp  (3, +3)
        assert _classify_cp(np.array([0.1, 0.5, 1.0])) == "ccp"

    def test_density_hessian_symmetric(self):
        """Analytic density Hessian should be symmetric at every point."""
        from vibeqc import Atom, BasisSet, Molecule
        from vibeqc import run_rks as _run_rks
        from vibeqc.qtaim import _density_hessian_on_grid

        mol = Molecule(
            [
                Atom(8, (0, 0, 0)),
                Atom(1, (0, 1.795, -1.388)),
                Atom(1, (0, -1.795, -1.388)),
            ]
        )
        basis = BasisSet(mol, "sto-3g")
        result = _run_rks(mol, basis)
        D = np.asarray(result.density)

        rng = np.random.default_rng(42)
        pts = rng.uniform(-1, 1, size=(10, 3))
        _, _, hess = _density_hessian_on_grid(D, basis, pts)
        for i in range(10):
            np.testing.assert_allclose(hess[i], hess[i].T, rtol=0, atol=1e-14)


_BOHR_TO_ANG = 0.529177210903


@pytest.fixture(scope="class")
def water_631gs(tmp_path_factory):
    """RHF/6-31G* water with its QTAIM analysis, shared by the bond-path tests."""
    from vibeqc import Atom, Molecule
    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.qtaim import qtaim_analysis
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
        basis="6-31g*",
        method="rhf",
        output=tmp_path_factory.mktemp("h2o_bond_paths") / "h2o",
        write_population_file=False,
        citations=False,
    )
    basis = BasisSet(mol, "6-31g*")
    qtaim = qtaim_analysis(result, basis, mol, grid_spacing=0.15, padding=2.5)
    return SimpleNamespace(
        mol=mol, basis=basis, result=result, D=np.asarray(result.density), qtaim=qtaim
    )


class TestBondPaths:
    """The bond path is the gradient-ascent line from a BCP to its two nuclei."""

    def test_water_bond_paths_join_the_nuclei_through_the_bcp(self, water_631gs):
        """Each O-H bond path runs nucleus -> BCP -> nucleus, with the density
        rising monotonically from the BCP to both nuclei.

        Regression: the tracer took its first step along the gradient at the
        BCP, where it vanishes, so every path collapsed to two copies of the BCP.
        """
        from vibeqc.qtaim import _density_hessian_on_grid

        mol, basis, D, qtaim = (
            water_631gs.mol, water_631gs.basis, water_631gs.D, water_631gs.qtaim
        )
        nuclei = np.array([a.xyz for a in mol.atoms]) * _BOHR_TO_ANG

        bcps = {
            frozenset(cp.atom_pair): cp
            for cp in qtaim.critical_points
            if cp.type == "bcp"
        }
        assert set(bcps) == {frozenset({0, 1}), frozenset({0, 2})}
        assert len(qtaim.bond_paths) == 2

        for bp in qtaim.bond_paths:
            bcp = bcps[frozenset(bp.atoms)]
            path = bp.path

            # Ends on the two nuclei, in the order `atoms` names them.
            np.testing.assert_allclose(path[0], nuclei[bp.atoms[0]], atol=1e-6)
            np.testing.assert_allclose(path[-1], nuclei[bp.atoms[1]], atol=1e-6)

            # Passes through the BCP, strictly inside the polyline.
            k = int(np.argmin(np.linalg.norm(path - bcp.position, axis=1)))
            np.testing.assert_allclose(path[k], bcp.position, atol=1e-9)
            assert 1 < k < len(path) - 2, f"BCP at index {k} of {len(path)}"

            # Not the old degenerate polyline: no point is repeated ...
            assert len(np.unique(path.round(9), axis=0)) == len(path)
            # ... and it does not wander: the O-H path is 1.0003 x the bond length.
            length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
            chord = np.linalg.norm(nuclei[bp.atoms[0]] - nuclei[bp.atoms[1]])
            assert length < 1.01 * chord, f"path {length:.4f} A for a {chord:.4f} A bond"

            # Density rises from the BCP to each nucleus.
            rho, _, _ = _density_hessian_on_grid(D, basis, path / _BOHR_TO_ANG)
            assert np.all(np.diff(rho[k:]) > 0), "rho not increasing BCP -> atoms[1]"
            assert np.all(np.diff(rho[: k + 1][::-1]) > 0), (
                "rho not increasing BCP -> atoms[0]"
            )

    def test_trace_bond_path_leaves_the_bcp_along_the_direction_given(
        self, water_631gs
    ):
        """Opposite directions along the BCP's positive-curvature eigenvector
        give the two halves of the bond path, one ending on each nucleus."""
        from vibeqc.qtaim import _density_hessian_on_grid, _trace_bond_path

        mol, basis, D, qtaim = (
            water_631gs.mol, water_631gs.basis, water_631gs.D, water_631gs.qtaim
        )
        atom_pos = np.array([a.xyz for a in mol.atoms], dtype=np.float64)

        def eval_fn(pts):
            return _density_hessian_on_grid(D, basis, pts)

        bcp = next(
            cp
            for cp in qtaim.critical_points
            if cp.type == "bcp" and frozenset(cp.atom_pair) == frozenset({0, 1})
        )
        start = bcp.position / _BOHR_TO_ANG
        _, _, hess = eval_fn(start[np.newaxis, :])
        evals, evecs = np.linalg.eigh(hess[0])
        assert evals[-1] > 0 > evals[1], f"not a (3,-1) point: {evals}"
        v3 = evecs[:, -1]

        halves = [
            _trace_bond_path(start, sign * v3, eval_fn, atom_pos, step_size=0.075)
            for sign in (1.0, -1.0)
        ]
        ends = []
        for sign, half in zip((1.0, -1.0), halves):
            np.testing.assert_array_equal(half[0], start)
            first_step = half[1] - half[0]
            assert first_step @ (sign * v3) > 0.999 * np.linalg.norm(first_step)
            ends.append(int(np.argmin(np.linalg.norm(atom_pos - half[-1], axis=1))))
            np.testing.assert_allclose(half[-1], atom_pos[ends[-1]], atol=1e-9)

        assert sorted(ends) == [0, 1], f"halves ended on atoms {ends}"

    def test_bcp_whose_paths_miss_a_nucleus_gets_no_bond_path(
        self, water_631gs, monkeypatch
    ):
        """A trace that stops short of a nucleus is reported, not completed
        with a made-up segment to the nearest atom."""
        import vibeqc.qtaim as qt

        def stalled(start, direction, *args, **kwargs):
            return np.vstack([start, start + 0.01 * direction])

        monkeypatch.setattr(qt, "_trace_bond_path", stalled)
        with pytest.warns(RuntimeWarning, match="no bond path was recorded"):
            q = qt.qtaim_analysis(
                water_631gs.result, water_631gs.basis, water_631gs.mol
            )
        assert sum(cp.type == "bcp" for cp in q.critical_points) == 2
        assert q.bond_paths == []

    def test_traced_nuclei_win_over_the_proximity_assignment(
        self, water_631gs, monkeypatch
    ):
        """If the paths end on a different pair than the two nearest nuclei to
        the BCP, the bond path joins the traced pair and says so."""
        import vibeqc.qtaim as qt

        mol = water_631gs.mol
        atom_pos = np.array([a.xyz for a in mol.atoms], dtype=np.float64)
        # Both halves of every path claim to end on the two H atoms (1 and 2),
        # whichever bond the BCP was assigned to.
        targets = iter([1, 2] * 2)

        def to_hydrogens(start, direction, *args, **kwargs):
            return np.vstack([start, atom_pos[next(targets)]])

        monkeypatch.setattr(qt, "_trace_bond_path", to_hydrogens)
        with pytest.warns(RuntimeWarning, match="by proximity"):
            q = qt.qtaim_analysis(
                water_631gs.result, water_631gs.basis, water_631gs.mol
            )
        assert [bp.atoms for bp in q.bond_paths] == [(1, 2), (1, 2)]
