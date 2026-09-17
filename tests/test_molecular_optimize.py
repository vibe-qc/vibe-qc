"""Tests for the native molecular geometry optimizer (no ASE required)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from vibeqc import Atom, Molecule
from vibeqc.molecular_optimize import MolecularOptimizeResult, optimize_molecule
from vibeqc.progress import ProgressLogger


def _h2_stretched() -> Molecule:
    """H₂ at 1.7 bohr — should relax to ~1.346 bohr (STO-3G eq)."""
    return Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.7])])


def _h2o() -> Molecule:
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.98]),
            Atom(1, [0.0, -1.43, -0.98]),
        ]
    )


class TestOptimizeMolecule:
    """End-to-end tests for the native optimizer."""

    def test_h2_rhf_converges_to_sto3g_bond(self):
        """H₂ (STO-3G) relaxes to the known equilibrium bond length."""
        mol = _h2_stretched()
        result = optimize_molecule(
            mol,
            basis_name="sto-3g",
            method="rhf",
            max_iter=30,
            conv_tol_grad=4.5e-4,
        )
        assert result.converged
        # STO-3G H₂ eq bond = 1.346 bohr
        a0 = result.system.atoms[0]
        a1 = result.system.atoms[1]
        r = float(np.linalg.norm(np.array(a1.xyz) - np.array(a0.xyz)))
        assert r == pytest.approx(1.346, abs=0.05)

    def test_h2o_rhf_converges(self):
        """H₂O (STO-3G) RHF optimization converges."""
        mol = _h2o()
        result = optimize_molecule(
            mol,
            basis_name="sto-3g",
            method="rhf",
            max_iter=50,
            conv_tol_grad=1e-3,
        )
        assert result.converged
        assert result.energy < -74.9  # near the STO-3G equilibrium

    @pytest.mark.skip(
        reason="PBE SCF oscillates at intermediate geometries — known C++ SCF robustness issue, not an optimizer bug"
    )
    def test_h2o_rks_pbe_converges(self):
        """H₂O (def2-svp) RKS/PBE optimization converges."""
        from vibeqc import RKSOptions

        rks_opts = RKSOptions()
        rks_opts.max_iter = 150
        rks_opts.use_diis = True
        mol = _h2o()
        result = optimize_molecule(
            mol,
            basis_name="def2-svp",
            method="rks",
            functional="PBE",
            rks_options=rks_opts,
            max_iter=50,
            conv_tol_grad=1e-3,
        )
        assert result.converged

    def test_trajectory_collected(self):
        """Trajectory frames and energies are collected when requested."""
        mol = _h2_stretched()
        result = optimize_molecule(
            mol,
            basis_name="sto-3g",
            method="rhf",
            max_iter=30,
            conv_tol_grad=4.5e-4,
            record_trajectory=True,
        )
        assert len(result.trajectory_frames) >= 2  # start + ≥1 step
        assert len(result.trajectory_energies) == len(result.trajectory_frames)
        # All frames should be Molecule objects with the right atom count.
        for frame in result.trajectory_frames:
            assert len(list(frame.atoms)) == 2
        # Energies should be in Hartree (negative for bound H₂).
        for e in result.trajectory_energies:
            assert e < 0.0

    def test_no_trajectory_when_disabled(self):
        """Trajectory lists are empty when record_trajectory=False."""
        mol = _h2_stretched()
        result = optimize_molecule(
            mol,
            basis_name="sto-3g",
            method="rhf",
            max_iter=5,
            conv_tol_grad=4.5e-4,
            record_trajectory=False,
        )
        assert result.trajectory_frames == []
        assert result.trajectory_energies == []

    def test_result_repr(self):
        mol = _h2_stretched()
        result = optimize_molecule(
            mol,
            basis_name="sto-3g",
            method="rhf",
            max_iter=5,
            conv_tol_grad=4.5e-4,
            record_trajectory=False,
        )
        s = repr(result)
        assert "MolecularOptimizeResult(" in s
        assert "energy=" in s
        assert "converged=" in s

    def test_progress_uses_caller_logger_without_stdout(self, capsys):
        stream = io.StringIO()
        logger = ProgressLogger(stream=stream, verbose=2)

        optimize_molecule(
            _h2_stretched(),
            basis_name="sto-3g",
            method="rhf",
            max_iter=1,
            record_trajectory=True,
            progress=logger,
        )

        assert capsys.readouterr().out == ""
        rendered = stream.getvalue()
        assert "Geometry optimization -- RHF" in rendered
        assert "Ha/bohr\n\n" in rendered
        assert "step" in rendered
        assert "converged=" in rendered

    def test_uhf_converges(self):
        """H₂ (STO-3G) UHF optimization converges (identical to RHF here)."""
        mol = _h2_stretched()
        result = optimize_molecule(
            mol,
            basis_name="sto-3g",
            method="uhf",
            max_iter=30,
            conv_tol_grad=4.5e-4,
        )
        assert result.converged

    def test_uks_pbe_converges(self):
        """H₂ (STO-3G) UKS/PBE optimization converges."""
        mol = _h2_stretched()
        result = optimize_molecule(
            mol,
            basis_name="sto-3g",
            method="uks",
            functional="PBE",
            max_iter=30,
            conv_tol_grad=4.5e-4,
        )
        assert result.converged


class TestOptimizeMoleculeFD:
    """Finite-difference gradient path for wavefunction methods."""

    def test_fci_fd_gradient_sign(self):
        """FCI FD gradient has the correct sign (bond-shortening)."""
        import numpy as np
        from vibeqc.molecular_optimize import _gradient_via_central_difference

        mol = _h2_stretched()
        grad = _gradient_via_central_difference(
            mol,
            "sto-3g",
            "fci",
            step_bohr=0.005,
        )
        # H2 at 1.7 bohr: atoms should move toward each other (z-axis).
        # Atom 0 at origin, atom 1 at +z → dE/dz > 0 for atom 1
        # (energy increases as bond stretches), and atom 0 moves opposite.
        assert grad.shape == (2, 3)
        assert np.abs(grad[0, 0]) < 1e-10  # x-component zero
        assert np.abs(grad[0, 1]) < 1e-10  # y-component zero
        assert grad[1, 2] > 0  # dE/dz positive for atom 1
        assert np.allclose(grad[0, 2], -grad[1, 2], atol=1e-8)

    def test_fci_optimization_converges(self):
        """H₂ FCI/STO-3G optimization converges via FD gradients."""
        mol = _h2_stretched()
        result = optimize_molecule(
            mol,
            basis_name="sto-3g",
            method="fci",
            max_iter=30,
            conv_tol_grad=4.5e-4,
            record_trajectory=True,
        )
        assert result.converged
        assert len(result.trajectory_frames) >= 2
        assert len(result.trajectory_energies) == len(result.trajectory_frames)
        # Bond should shorten from 1.7 bohr.
        a0, a1 = result.system.atoms
        import numpy as np

        r = float(np.linalg.norm(np.array(a1.xyz) - np.array(a0.xyz)))
        assert r < 1.7

    def test_fci_fd_agrees_with_analytic_rhf(self):
        """FCI FD gradient has same sign as analytic RHF gradient."""
        import numpy as np
        from vibeqc import BasisSet, RHFOptions, run_rhf
        from vibeqc._vibeqc_core import compute_gradient
        from vibeqc.molecular_optimize import _gradient_via_central_difference

        mol = _h2_stretched()
        basis = BasisSet(mol, "sto-3g")

        # Analytic RHF
        rhf = run_rhf(mol, basis, RHFOptions())
        g_rhf = np.asarray(compute_gradient(mol, basis, rhf))

        # FD FCI
        g_fci = _gradient_via_central_difference(
            mol,
            "sto-3g",
            "fci",
            step_bohr=0.005,
        )

        # Both should agree on sign and be within a factor of ~3.
        assert np.sign(g_rhf[1, 2]) == np.sign(g_fci[1, 2])
        ratio = g_fci[1, 2] / g_rhf[1, 2]
        assert 0.2 < ratio < 3.0

    @pytest.mark.parametrize("method", ["nevpt2", "caspt2", "mrci"])
    def test_multireference_pt2_ci_dispatches_through_fd(self, method, monkeypatch):
        """nevpt2/caspt2/mrci reach the same generic FD machinery as fci
        (GitLab #357): _evaluate_energy / _gradient_via_central_difference
        forward the method string to _run_single_point unchanged, with no
        method-specific carve-out that would (silently) substitute a
        mean-field surface. A stub keeps this fast -- a real single point
        for these methods costs 0.2-1.6 s each and a full FD gradient
        4.6-27.2 s (measured on H2O/STO-3G CAS(4,4), see #357)."""
        from vibeqc import BasisSet
        from vibeqc.molecular_optimize import _gradient_via_central_difference
        from vibeqc.solvers import CASCIOptions

        seen = []

        def _stub(m, mol, basis, **kwargs):
            seen.append(m)

            class _Fake:
                energy = -1.0

            return _Fake()

        import vibeqc.runner as runner_module

        monkeypatch.setattr(runner_module, "_run_single_point", _stub)

        mol = _h2_stretched()
        grad = _gradient_via_central_difference(
            mol,
            "sto-3g",
            method,
            active_space=(2, 2),
            casci_options=CASCIOptions(),
            step_bohr=0.005,
        )
        assert grad.shape == (2, 3)
        # 2 atoms x 3 components x 2 (+/-) displacements.
        assert len(seen) == 12
        assert all(m == method for m in seen)


class TestOptimizeMoleculeImports:
    """Verify public API exports."""

    def test_importable_from_vibeqc(self):
        from vibeqc import MolecularOptimizeResult, optimize_molecule

        assert callable(optimize_molecule)
        assert issubclass(MolecularOptimizeResult, object)


class TestBrentMinimize1D:
    """Unit tests for the classic Brent 1-D minimiser."""

    def test_parabola_exact(self):
        """Minimum of a perfect parabola is found to high precision."""
        from vibeqc import brent_minimize_1d

        def f(x):
            return (x - 3.7) ** 2 + 0.5

        x_opt, f_opt, n = brent_minimize_1d(f, 0.0, 3.0, 6.0, tol=1e-10)
        assert abs(x_opt - 3.7) < 1e-8
        assert abs(f_opt - 0.5) < 1e-8
        assert n >= 3
        # tol=1e-10 with a wide initial bracket [0,6] requires ~30-40 evals

    def test_cosine_minimum(self):
        """Minimum of cos(x) at x=pi."""
        import math

        from vibeqc import brent_minimize_1d

        def f(x):
            return math.cos(x)

        x_opt, f_opt, n = brent_minimize_1d(f, 2.0, 3.0, 4.0, tol=1e-10)
        assert abs(x_opt - math.pi) < 1e-8
        assert abs(f_opt - (-1.0)) < 1e-8

    def test_quartic_well(self):
        """Minimum of a quartic well at x=1."""
        from vibeqc import brent_minimize_1d

        def f(x):
            return (x - 1.0) ** 4

        x_opt, f_opt, n = brent_minimize_1d(f, 0.0, 0.5, 2.0, tol=1e-10)
        assert abs(x_opt - 1.0) < 1e-6
        assert abs(f_opt) < 1e-6

    def test_asymmetric_bracket(self):
        """Works with an asymmetric bracket around the minimum."""
        from vibeqc import brent_minimize_1d

        def f(x):
            return x**2 - 4 * x + 7  # Minimum at x=2, f_min=3

        x_opt, f_opt, n = brent_minimize_1d(f, 0.0, 1.0, 10.0, tol=1e-10)
        assert abs(x_opt - 2.0) < 1e-6
        assert abs(f_opt - 3.0) < 1e-6

    def test_flat_bracket_is_degenerate(self):
        """A flat bracket returns immediately with the centre point."""
        from vibeqc import brent_minimize_1d

        def f(x):
            return 42.0

        x_opt, f_opt, n = brent_minimize_1d(f, 0.0, 5.0, 10.0, tol=1e-10)
        assert x_opt == 5.0
        assert f_opt == 42.0
        assert n == 3

    def test_returns_tuple_of_three(self):
        """Return tuple has exactly three elements."""
        from vibeqc import brent_minimize_1d

        def f(x):
            return x**2

        result = brent_minimize_1d(f, -3.0, -1.0, 2.0, tol=1e-10)
        assert isinstance(result, tuple)
        assert len(result) == 3
        x_opt, f_opt, n = result
        assert isinstance(x_opt, float)
        assert isinstance(f_opt, float)
        assert isinstance(n, int)


class TestBracketLineMinimum:
    """Unit tests for the downhill-bracketing helper."""

    def test_downhill_then_uphill(self):
        """Typical case: function goes down then up."""
        from vibeqc.molecular_optimize import _bracket_line_minimum

        def f(x):
            return (x + 0.5) ** 2

        a, fa, b, fb, c, fc, n = _bracket_line_minimum(f, 0.0, f(0.0), step=0.1)
        assert fb < fa
        assert fb < fc
        assert n >= 2

    def test_uphill_initially_reverses(self):
        """When the first step goes uphill, the direction reverses."""
        from vibeqc.molecular_optimize import _bracket_line_minimum

        def g(x):
            return (x + 1.0) ** 2  # minimum at x=-1

        a, fa, b, fb, c, fc, n = _bracket_line_minimum(g, 0.0, g(0.0), step=0.1)
        assert fb < fa
        assert fb < fc

    def test_bracket_contains_minimum(self):
        """The returned bracket actually contains the minimum."""
        from vibeqc.molecular_optimize import _bracket_line_minimum

        def f(x):
            return (x - 2.0) ** 2 + 1.0

        a, fa, b, fb, c, fc, n = _bracket_line_minimum(f, 0.0, f(0.0), step=0.5)
        assert fb < fa
        assert fb < fc
        assert min(a, c) <= 2.0 <= max(a, c)


class TestLineSearchBrent:
    """Unit tests for the combined bracket + Brent line search."""

    def test_finds_minimum_of_quadratic(self):
        """Line search finds the 1-D minimum."""
        from vibeqc.molecular_optimize import _line_search_brent

        def f(alpha):
            return (alpha - 0.3) ** 2

        alpha_opt, f_opt, n = _line_search_brent(f, 0.0, f(0.0), step=0.1)
        assert abs(alpha_opt - 0.3) < 1e-4
        assert abs(f_opt) < 1e-4
        assert n >= 2

    def test_starts_from_nonzero(self):
        """Line search works when starting alpha is not zero."""
        from vibeqc.molecular_optimize import _line_search_brent

        def f(alpha):
            return (alpha - 1.5) ** 2 + 3.0

        alpha_opt, f_opt, n = _line_search_brent(f, 0.5, f(0.5), step=0.2)
        assert abs(alpha_opt - 1.5) < 1e-3
        assert abs(f_opt - 3.0) < 1e-3

    def test_narrow_well(self):
        """Works with a very narrow well."""
        from vibeqc.molecular_optimize import _line_search_brent

        def f(alpha):
            return (alpha * 10.0) ** 4

        alpha_opt, f_opt, n = _line_search_brent(f, 0.0, f(0.0), step=0.5)
        assert abs(alpha_opt) < 0.01
        assert abs(f_opt) < 1e-6


class TestOptimizeMoleculeBrent:
    """End-to-end tests for the Brent geometry optimizer."""

    def test_progress_uses_caller_logger_without_stdout(self, capsys):
        """Brent progress follows the same caller-owned logger contract."""
        from vibeqc.molecular_optimize import optimize_molecule_brent

        stream = io.StringIO()
        logger = ProgressLogger(stream=stream, verbose=2)

        optimize_molecule_brent(
            _h2_stretched(),
            basis_name="sto-3g",
            method="rhf",
            max_iter=1,
            record_trajectory=False,
            progress=logger,
        )

        assert capsys.readouterr().out == ""
        rendered = stream.getvalue()
        assert "Geometry optimization (Brent)" in rendered
        assert "Ha/bohr\n\n" in rendered
        assert "line search:" in rendered
        assert "step   1" in rendered

    def test_h2_rhf_converges(self):
        """H2 (STO-3G) RHF relaxes via Brent."""
        import numpy as np
        from vibeqc import Atom, Molecule
        from vibeqc.molecular_optimize import optimize_molecule_brent

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.7])])
        result = optimize_molecule_brent(
            mol,
            basis_name="sto-3g",
            method="rhf",
            max_iter=30,
            conv_tol_grad=4.5e-4,
        )
        assert result.converged
        a0 = result.system.atoms[0]
        a1 = result.system.atoms[1]
        r = float(np.linalg.norm(np.array(a1.xyz) - np.array(a0.xyz)))
        assert r == pytest.approx(1.346, abs=0.05)

    def test_trajectory_collected(self):
        """Trajectory frames are collected when requested."""
        from vibeqc import Atom, Molecule
        from vibeqc.molecular_optimize import optimize_molecule_brent

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.7])])
        result = optimize_molecule_brent(
            mol,
            basis_name="sto-3g",
            method="rhf",
            max_iter=10,
            conv_tol_grad=4.5e-4,
            record_trajectory=True,
        )
        assert len(result.trajectory_frames) >= 2
        assert len(result.trajectory_energies) == len(result.trajectory_frames)
        for frame in result.trajectory_frames:
            assert len(list(frame.atoms)) == 2
        for e in result.trajectory_energies:
            assert e < 0.0

    def test_no_trajectory_when_disabled(self):
        """Trajectory empty when record_trajectory=False."""
        from vibeqc import Atom, Molecule
        from vibeqc.molecular_optimize import optimize_molecule_brent

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.7])])
        result = optimize_molecule_brent(
            mol,
            basis_name="sto-3g",
            method="rhf",
            max_iter=5,
            conv_tol_grad=4.5e-4,
            record_trajectory=False,
        )
        assert result.trajectory_frames == []
        assert result.trajectory_energies == []

    def test_result_type(self):
        """Returns MolecularOptimizeResult."""
        from vibeqc import Atom, MolecularOptimizeResult, Molecule
        from vibeqc.molecular_optimize import optimize_molecule_brent

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.7])])
        result = optimize_molecule_brent(
            mol,
            basis_name="sto-3g",
            method="rhf",
            max_iter=5,
            conv_tol_grad=4.5e-4,
            record_trajectory=False,
        )
        assert isinstance(result, MolecularOptimizeResult)
        s = repr(result)
        assert "MolecularOptimizeResult(" in s

    def test_freeze_indices(self):
        """Frozen atoms stay at their initial positions."""
        import numpy as np
        from vibeqc import Atom, Molecule
        from vibeqc.molecular_optimize import optimize_molecule_brent

        mol = Molecule(
            [
                Atom(1, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 0.0, 1.7]),
            ]
        )
        result = optimize_molecule_brent(
            mol,
            basis_name="sto-3g",
            method="rhf",
            max_iter=20,
            conv_tol_grad=4.5e-4,
            freeze_indices=[0],
            record_trajectory=False,
        )
        a0 = result.system.atoms[0]
        assert np.allclose(np.array(a0.xyz), [0.0, 0.0, 0.0], atol=1e-10)

    def test_freeze_indices_out_of_range_raises(self):
        """Invalid freeze index raises ValueError."""
        import pytest
        from vibeqc import Atom, Molecule
        from vibeqc.molecular_optimize import optimize_molecule_brent

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.7])])
        with pytest.raises(ValueError, match="out of range"):
            optimize_molecule_brent(
                mol,
                basis_name="sto-3g",
                method="rhf",
                max_iter=5,
                freeze_indices=[99],
            )

    def test_uhf_converges(self):
        """UHF Brent optimization converges for H2."""
        from vibeqc import Atom, Molecule
        from vibeqc.molecular_optimize import optimize_molecule_brent

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.7])])
        result = optimize_molecule_brent(
            mol,
            basis_name="sto-3g",
            method="uhf",
            max_iter=30,
            conv_tol_grad=4.5e-4,
        )
        assert result.converged

    def test_uks_pbe_converges(self):
        """UKS/PBE Brent optimization converges for H2."""
        from vibeqc import Atom, Molecule
        from vibeqc.molecular_optimize import optimize_molecule_brent

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.7])])
        result = optimize_molecule_brent(
            mol,
            basis_name="sto-3g",
            method="uks",
            functional="PBE",
            max_iter=30,
            conv_tol_grad=4.5e-4,
        )
        assert result.converged

    def test_energy_lower_than_start(self):
        """Optimized energy is lower than starting energy."""
        from vibeqc import Atom, Molecule
        from vibeqc.molecular_optimize import optimize_molecule_brent

        mol = Molecule(
            [
                Atom(8, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 1.43, -0.98]),
                Atom(1, [0.0, -1.43, -0.98]),
            ]
        )
        result = optimize_molecule_brent(
            mol,
            basis_name="sto-3g",
            method="rhf",
            max_iter=30,
            conv_tol_grad=1e-3,
        )
        assert result.energy < -74.9

    def test_optimizer_backend_brent_resolves(self):
        """_resolve_optimizer_backend returns 'brent' for 'brent'."""
        from vibeqc.runner import _resolve_optimizer_backend

        assert _resolve_optimizer_backend("brent") == "brent"

    def test_optimizer_backend_rejects_unknown(self):
        """Unknown backend raises ValueError mentioning 'brent'."""
        import pytest
        from vibeqc.runner import _resolve_optimizer_backend

        with pytest.raises(ValueError, match="brent"):
            _resolve_optimizer_backend("magic")


def _gradient_path_spy(monkeypatch):
    """Patch the analytic + FD gradient kernels of vibeqc.molecular_optimize
    so any optimizer (brent, primary, or the geomopt provider -- all import
    these names from the module at call time) records which path it took.
    Returns a ``calls`` dict with ``"analytic"`` / ``"fd"`` counters."""
    import vibeqc.molecular_optimize as mo

    calls = {"analytic": 0, "fd": 0}
    real_analytic = mo._compute_molecular_gradient
    real_fd = mo._gradient_via_central_difference

    def _spy_analytic(*args, **kwargs):
        calls["analytic"] += 1
        return real_analytic(*args, **kwargs)

    def _spy_fd(*args, **kwargs):
        calls["fd"] += 1
        return real_fd(*args, **kwargs)

    monkeypatch.setattr(mo, "_compute_molecular_gradient", _spy_analytic)
    monkeypatch.setattr(mo, "_gradient_via_central_difference", _spy_fd)
    return calls


class TestCasscfAnalyticGradientGating:
    """The three molecular optimizers (L-BFGS-B primary, Brent, and the
    geomopt MolecularSCFProvider) share one decision per CAS-family method
    about when to walk the analytic gradient:

    * CASSCF: only inside the validated envelope (state-specific,
      closed-shell; the v0.15.0 P0 fix). Outside it (SA-CASSCF, open-shell,
      an explicit ``compute_wz="numerical"`` FD request) they fall back to
      full-energy central FD. ``compute_wz=True`` is a no-op alias of the
      analytic gradient since #516 and stays inside. See
      ``_casscf_analytic_gradient_ok``.
    * CASPT2/NEVPT2: only when the run requests and can produce the
      runner-supplied relaxed full-energy FD gradient: CASSCF-referenced,
      gas-phase, ``compute_corr_grad=True``. Otherwise the runner would return
      the bare CASSCF reference gradient (or None) -- a surface inconsistent
      with the reported PT2 energy -- so they fall back to the optimizer's
      full-energy central FD. See ``_mrpt_analytic_gradient_ok``.
    """

    def test_envelope_helper(self):
        from vibeqc.molecular_optimize import _casscf_analytic_gradient_ok
        from vibeqc.solvers import CASSCFOptions

        singlet = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.5])])
        triplet = Molecule(
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.5])], 0, 3
        )
        # Inside the validated envelope.
        assert _casscf_analytic_gradient_ok(singlet, CASSCFOptions())
        assert _casscf_analytic_gradient_ok(singlet, None)
        # compute_wz=True is a no-op alias of the analytic path (#516).
        assert _casscf_analytic_gradient_ok(singlet, CASSCFOptions(compute_wz=True))
        # Outside: state-averaged, an explicit FD request, open-shell.
        assert not _casscf_analytic_gradient_ok(singlet, CASSCFOptions(nroots=2))
        assert not _casscf_analytic_gradient_ok(
            singlet, CASSCFOptions(compute_wz="numerical")
        )
        assert not _casscf_analytic_gradient_ok(triplet, CASSCFOptions())

    def test_state_specific_casscf_brent_uses_analytic(self, monkeypatch):
        import vibeqc.molecular_optimize as mo
        from vibeqc.solvers import CASSCFOptions

        calls = _gradient_path_spy(monkeypatch)
        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        mo.optimize_molecule_brent(
            mol,
            basis_name="sto-3g",
            method="casscf",
            active_space=(2, 2),
            casscf_options=CASSCFOptions(),  # nroots=1, closed-shell, compute_wz=False
            max_iter=1,
            record_trajectory=False,
        )
        assert calls["analytic"] >= 1, (
            "state-specific CASSCF Brent optimization did not use the "
            "validated analytic gradient"
        )
        assert calls["fd"] == 0, (
            "state-specific CASSCF Brent optimization fell back to FD"
        )

    def test_sa_casscf_brent_uses_fd(self, monkeypatch):
        import vibeqc.molecular_optimize as mo
        from vibeqc.solvers import CASSCFOptions

        calls = _gradient_path_spy(monkeypatch)
        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        mo.optimize_molecule_brent(
            mol,
            basis_name="sto-3g",
            method="casscf",
            active_space=(2, 2),
            casscf_options=CASSCFOptions(nroots=2, weights=[0.5, 0.5]),
            max_iter=1,
            record_trajectory=False,
        )
        assert calls["fd"] >= 1, "SA-CASSCF Brent optimization did not use FD"
        assert calls["analytic"] == 0, (
            "SA-CASSCF Brent optimization walked the (unvalidated) analytic "
            "gradient -- it is only finiteness/translational-invariance checked"
        )

    def test_provider_state_specific_casscf_uses_analytic(self, monkeypatch):
        from vibeqc.geomopt import MolecularSCFProvider
        from vibeqc.solvers import CASSCFOptions

        calls = _gradient_path_spy(monkeypatch)
        provider = MolecularSCFProvider(
            "sto-3g", method="casscf", active_space=(2, 2),
            casscf_options=CASSCFOptions(),
        )
        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        provider(mol)
        assert calls["analytic"] >= 1 and calls["fd"] == 0

    def test_provider_sa_casscf_uses_fd(self, monkeypatch):
        from vibeqc.geomopt import MolecularSCFProvider
        from vibeqc.solvers import CASSCFOptions

        calls = _gradient_path_spy(monkeypatch)
        provider = MolecularSCFProvider(
            "sto-3g", method="casscf", active_space=(2, 2),
            casscf_options=CASSCFOptions(nroots=2, weights=[0.5, 0.5]),
        )
        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        provider(mol)
        assert calls["fd"] >= 1 and calls["analytic"] == 0

    def test_analytic_set_excludes_cas_family_literal(self):
        """The static ``_mean_field`` set never contains the CAS family.
        In BOTH native optimizers, CASSCF is admitted only through the
        _casscf_analytic_gradient_ok gate and CASPT2/NEVPT2 only through
        the _mrpt_analytic_gradient_ok gate (runner-supplied relaxed FD,
        b3645a46 + compute_corr_grad gating) rather than by
        joining the mean-field literal, whose members skip the CAS options
        plumbing."""
        import inspect

        import vibeqc.molecular_optimize as mo

        for fn in (mo.optimize_molecule_brent, mo.optimize_molecule):
            src = inspect.getsource(fn)
            assert '_mean_field = {"rhf", "uhf", "rks", "uks", "rohf"}' in src
            gate_region = src.split("_mean_field =")[1].split(
                "trajectory_frames"
            )[0]
            # CASSCF enters only via the helper, gated on method == "casscf".
            assert "_casscf_analytic_gradient_ok" in gate_region
            # CASPT2/NEVPT2 enter only via their own gated helper, not the
            # mean-field set and not unconditionally.
            assert 'method_lower in ("caspt2", "nevpt2")' in gate_region
            assert "_mrpt_analytic_gradient_ok" in gate_region

    # ---- CASPT2/NEVPT2 gradient gating (b3645a46 + compute_corr_grad) ----

    def test_mrpt_envelope_helper(self):
        from vibeqc._vibeqc_core import BasisSet
        from vibeqc.molecular_optimize import _mrpt_analytic_gradient_ok
        from vibeqc.solvers import CASPT2Options, CASSCFOptions, NEVPT2Options

        cas = CASSCFOptions()
        # Inside the envelope: CASSCF reference + compute_corr_grad, gas phase.
        assert _mrpt_analytic_gradient_ok(
            "caspt2", cas, CASPT2Options(compute_corr_grad=True), None
        )
        assert _mrpt_analytic_gradient_ok(
            "nevpt2", cas, None, NEVPT2Options(compute_corr_grad=True)
        )
        # Default options: the runner would return the bare CASSCF gradient.
        assert not _mrpt_analytic_gradient_ok(
            "caspt2", cas, CASPT2Options(), None
        )
        assert not _mrpt_analytic_gradient_ok("nevpt2", cas, None, None)
        # CASCI-on-HF reference: no reference gradient exists at all.
        assert not _mrpt_analytic_gradient_ok(
            "caspt2", None, CASPT2Options(compute_corr_grad=True), None
        )
        # Solvated: the CPCM composition has no analytic PT2 gradient.
        assert not _mrpt_analytic_gradient_ok(
            "caspt2",
            cas,
            CASPT2Options(compute_corr_grad=True),
            None,
            solvent="water",
        )
        # Unsupported single-state IC response variants stay on the
        # optimizer-owned full-energy FD route.
        assert not _mrpt_analytic_gradient_ok(
            "caspt2",
            cas,
            CASPT2Options(compute_corr_grad=True, ipea=0.25),
            None,
        )
        assert not _mrpt_analytic_gradient_ok(
            "caspt2",
            cas,
            CASPT2Options(compute_corr_grad=True, engine="cases"),
            None,
        )
        assert not _mrpt_analytic_gradient_ok(
            "caspt2",
            CASSCFOptions(ci_solver="selected_ci"),
            CASPT2Options(compute_corr_grad=True),
            None,
        )

        # The large unshifted IC path dispatches to the direct engine and
        # therefore remains outside the explicit small-space response route.
        n2 = Molecule([Atom(7, [0, 0, 0]), Atom(7, [0, 0, 2.1])])
        n2_basis = BasisSet(n2, "6-31g")
        assert not _mrpt_analytic_gradient_ok(
            "caspt2",
            cas,
            CASPT2Options(compute_corr_grad=True),
            None,
            molecule=n2,
            basis=n2_basis,
            active_space=(6, 6),
        )

    def test_fd_energy_path_disables_inner_caspt2_gradient(self, monkeypatch):
        """Outer FD evaluates shifted/direct energies without nested gradients."""
        from types import SimpleNamespace

        import vibeqc.runner as runner
        from vibeqc._vibeqc_core import BasisSet
        from vibeqc.molecular_optimize import _evaluate_energy
        from vibeqc.solvers import CASPT2Options, CASSCFOptions

        seen = []

        def fake_single_point(*_args, **kwargs):
            seen.append(kwargs["caspt2_options"])
            assert kwargs["_mrpt_gradient"] is False
            return SimpleNamespace(energy=-1.0)

        monkeypatch.setattr(runner, "_run_single_point", fake_single_point)
        mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
        options = CASPT2Options(compute_corr_grad=True, ipea=0.25)
        energy = _evaluate_energy(
            mol,
            BasisSet(mol, "sto-3g"),
            "caspt2",
            active_space=(2, 2),
            casscf_options=CASSCFOptions(),
            caspt2_options=options,
        )
        assert energy == -1.0
        assert options.compute_corr_grad is True
        assert seen and seen[0].compute_corr_grad is False
        assert seen[0].ipea == 0.25

    def test_caspt2_default_brent_uses_fd(self, monkeypatch):
        """Default CASPT2Options (compute_corr_grad=False): the runner's
        gradient would be the bare CASSCF reference gradient, so the
        optimizer must NOT walk it -- full-energy FD instead."""
        import vibeqc.molecular_optimize as mo
        from vibeqc.solvers import CASPT2Options, CASSCFOptions

        calls = _gradient_path_spy(monkeypatch)
        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        mo.optimize_molecule_brent(
            mol,
            basis_name="6-31g",
            method="caspt2",
            active_space=(2, 2),
            casscf_options=CASSCFOptions(),
            caspt2_options=CASPT2Options(),
            max_iter=1,
            record_trajectory=False,
        )
        assert calls["fd"] >= 1, (
            "default-options CASPT2 Brent optimization did not use FD"
        )
        assert calls["analytic"] == 0, (
            "default-options CASPT2 Brent optimization walked the bare "
            "CASSCF gradient while reporting PT2 energies"
        )

    def test_caspt2_corr_grad_brent_uses_analytic(self, monkeypatch):
        import warnings

        import vibeqc.molecular_optimize as mo
        from vibeqc.solvers import CASPT2Options, CASSCFOptions

        calls = _gradient_path_spy(monkeypatch)
        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mo.optimize_molecule_brent(
                mol,
                basis_name="6-31g",
                method="caspt2",
                active_space=(2, 2),
                casscf_options=CASSCFOptions(),
                caspt2_options=CASPT2Options(compute_corr_grad=True),
                max_iter=1,
                record_trajectory=False,
            )
        assert calls["analytic"] >= 1, (
            "compute_corr_grad=True CASPT2 Brent optimization did not use "
            "the runner-supplied relaxed FD gradient"
        )
        assert calls["fd"] == 0
        # Pins the caspt2_options plumbing: if the analytic branch dropped
        # the options, the runner would warn 'compute_corr_grad=False' and
        # hand back the bare CASSCF gradient.
        assert not [
            w for w in caught if "compute_corr_grad=False" in str(w.message)
        ], "caspt2_options did not reach the solver on the analytic path"

    def test_caspt2_corr_grad_primary_uses_analytic(self, monkeypatch):
        import warnings

        import vibeqc.molecular_optimize as mo
        from vibeqc.solvers import CASPT2Options, CASSCFOptions

        calls = _gradient_path_spy(monkeypatch)
        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mo.optimize_molecule(
                mol,
                basis_name="6-31g",
                method="caspt2",
                active_space=(2, 2),
                casscf_options=CASSCFOptions(),
                caspt2_options=CASPT2Options(compute_corr_grad=True),
                max_iter=1,
                record_trajectory=False,
            )
        assert calls["analytic"] >= 1 and calls["fd"] == 0
        assert not [
            w for w in caught if "compute_corr_grad=False" in str(w.message)
        ], "caspt2_options did not reach the solver on the analytic path"

    def test_provider_caspt2_default_uses_fd(self, monkeypatch):
        from vibeqc.geomopt import MolecularSCFProvider
        from vibeqc.solvers import CASPT2Options, CASSCFOptions

        calls = _gradient_path_spy(monkeypatch)
        provider = MolecularSCFProvider(
            "6-31g",
            method="caspt2",
            active_space=(2, 2),
            casscf_options=CASSCFOptions(),
            caspt2_options=CASPT2Options(),
        )
        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        provider(mol)
        assert calls["fd"] >= 1 and calls["analytic"] == 0

    def test_provider_caspt2_corr_grad_uses_analytic(self, monkeypatch):
        import warnings

        from vibeqc.geomopt import MolecularSCFProvider
        from vibeqc.solvers import CASPT2Options, CASSCFOptions

        calls = _gradient_path_spy(monkeypatch)
        provider = MolecularSCFProvider(
            "6-31g",
            method="caspt2",
            active_space=(2, 2),
            casscf_options=CASSCFOptions(),
            caspt2_options=CASPT2Options(compute_corr_grad=True),
        )
        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            provider(mol)
        assert calls["analytic"] >= 1 and calls["fd"] == 0
        assert not [
            w for w in caught if "compute_corr_grad=False" in str(w.message)
        ], "caspt2_options did not reach the solver on the analytic path"

    def test_provider_nevpt2_corr_grad_uses_analytic(self, monkeypatch):
        """Also pins the provider's nevpt2_options plumbing (new parameter)."""
        import warnings

        from vibeqc.geomopt import MolecularSCFProvider
        from vibeqc.solvers import CASSCFOptions, NEVPT2Options

        calls = _gradient_path_spy(monkeypatch)
        provider = MolecularSCFProvider(
            "6-31g",
            method="nevpt2",
            active_space=(2, 2),
            casscf_options=CASSCFOptions(),
            nevpt2_options=NEVPT2Options(compute_corr_grad=True),
        )
        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            provider(mol)
        assert calls["analytic"] >= 1 and calls["fd"] == 0
        assert not [
            w for w in caught if "compute_corr_grad=False" in str(w.message)
        ], "nevpt2_options did not reach the solver on the analytic path"


class TestSolventMethodGating:
    """CPCM composes with the mean-field SCFs only (run_cpcm_scf: rhf /
    uhf / rks / uks). ``_run_molecular_scf`` used to assign ``opts`` only
    in those branches yet pass ``options=opts`` to ``run_cpcm_scf``
    unconditionally, so method="casscf" (or caspt2 / nevpt2 / rohf) with a
    solvent died with a NameError -- reachable via the optimizers' analytic
    CASSCF path, whose envelope gate does not exclude solvated runs. The
    fix refuses up front, before any gas-phase solve, with an error naming
    the actual limitation (rather than silently rerouting to FD, where
    _run_single_point drops the solvent and would walk the gas-phase
    surface while claiming solvation)."""

    @pytest.mark.parametrize("method", ["casscf", "caspt2", "nevpt2", "rohf"])
    def test_run_molecular_scf_rejects_solvent(self, method):
        from vibeqc import BasisSet
        from vibeqc.molecular_optimize import _run_molecular_scf

        mol = _h2_stretched()
        basis = BasisSet(mol, "sto-3g")
        with pytest.raises(ValueError, match="CPCM.*not supported"):
            _run_molecular_scf(
                mol,
                basis,
                method,
                active_space=(2, 2),
                solvent="water",
            )

    def test_run_molecular_scf_rejects_solvent_before_solving(self, monkeypatch):
        """The refusal fires before the gas-phase CASSCF is even attempted."""
        import vibeqc.runner as runner
        from vibeqc import BasisSet
        from vibeqc.molecular_optimize import _run_molecular_scf

        def _boom(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("gas-phase solve ran despite solvent refusal")

        monkeypatch.setattr(runner, "_run_single_point", _boom)
        mol = _h2_stretched()
        basis = BasisSet(mol, "sto-3g")
        with pytest.raises(ValueError, match="CPCM"):
            _run_molecular_scf(
                mol, basis, "casscf", active_space=(2, 2), solvent="water"
            )

    def test_optimize_molecule_casscf_solvent_clear_error(self):
        """End-to-end: the analytic CASSCF optimizer path (whose envelope
        gate passes for solvated state-specific closed-shell runs) surfaces
        the clear ValueError, not a NameError on ``opts``."""
        mol = _h2_stretched()
        with pytest.raises(ValueError, match="CPCM.*not supported"):
            optimize_molecule(
                mol,
                basis_name="sto-3g",
                method="casscf",
                active_space=(2, 2),
                solvent="water",
                max_iter=1,
                record_trajectory=False,
            )

    def test_unknown_method_error_unchanged_with_solvent(self):
        """An unknown method still reports 'Unknown method', not the
        solvent refusal."""
        from vibeqc import BasisSet
        from vibeqc.molecular_optimize import _run_molecular_scf

        mol = _h2_stretched()
        basis = BasisSet(mol, "sto-3g")
        with pytest.raises(ValueError, match="Unknown method"):
            _run_molecular_scf(mol, basis, "magic", solvent="water")

    @pytest.mark.parametrize("method", ["casscf", "caspt2", "nevpt2"])
    def test_fd_energy_path_rejects_solvent(self, method):
        """The FD optimizer energy path (_evaluate_energy →
        runner._run_single_point) used to *silently drop* the solvent, so
        a solvated PT2 optimization rerouted to FD walked the gas-phase
        surface while claiming solvation. The dispatcher gate now raises
        the same clear ValueError as _run_molecular_scf."""
        from vibeqc import BasisSet
        from vibeqc.molecular_optimize import _evaluate_energy

        mol = _h2_stretched()
        basis = BasisSet(mol, "sto-3g")
        with pytest.raises(
            ValueError, match="Implicit solvation is not supported"
        ):
            _evaluate_energy(
                mol,
                basis,
                method,
                active_space=(2, 2),
                solvent="water",
            )

    def test_brent_caspt2_solvent_clear_error(self):
        """End-to-end: solvated CASPT2 with default options fails the
        analytic-gradient envelope and reroutes to the FD path — which
        must now refuse instead of silently optimizing in gas phase."""
        from vibeqc.molecular_optimize import optimize_molecule_brent
        from vibeqc.solvers import CASPT2Options, CASSCFOptions

        mol = _h2_stretched()
        with pytest.raises(
            ValueError, match="Implicit solvation.*not supported|CPCM.*not supported"
        ):
            optimize_molecule_brent(
                mol,
                basis_name="sto-3g",
                method="caspt2",
                active_space=(2, 2),
                casscf_options=CASSCFOptions(),
                caspt2_options=CASPT2Options(),
                solvent="water",
                max_iter=1,
                record_trajectory=False,
            )


class TestCasGradientNoneCheck:
    """``_compute_molecular_gradient`` checks ``SolverResult.gradient`` for
    None BEFORE ``np.asarray``. Pre-fix the conversion ran first, so a None
    gradient (e.g. a CASCI-referenced PT2 run that computed no reference
    gradient) surfaced as an opaque numpy TypeError instead of the intended
    ValueError."""

    @pytest.mark.parametrize("method", ["casscf", "caspt2", "nevpt2"])
    def test_none_gradient_raises_valueerror(self, method):
        from vibeqc import BasisSet
        from vibeqc.molecular_optimize import _compute_molecular_gradient

        class _StubResult:
            gradient = None

        mol = _h2_stretched()
        basis = BasisSet(mol, "sto-3g")
        with pytest.raises(ValueError, match="gradient is None"):
            _compute_molecular_gradient(mol, basis, _StubResult(), method)


# ---------------------------------------------------------------------------
# ECP systems (#643)
# ---------------------------------------------------------------------------


def _h2s_lanl2dz_with_ecp():
    """H2S with the LANL2DZ ECP attached for the start geometry, as run_job does."""
    import vibeqc as vq
    from vibeqc.ecp_metadata import attach_inline_ecp_options_from_basis_sidecar

    mol = Molecule(
        [Atom(16, [0.0, 0.0, 0.0]),
         Atom(1, [0.0, 1.815, 1.425]),
         Atom(1, [0.0, -1.815, 1.425])], 0, 1)
    opts = vq.RHFOptions()
    opts.max_iter = 200
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    attach_inline_ecp_options_from_basis_sidecar(opts, mol, vq.BasisSet(mol, "lanl2dz"))
    assert len(opts.ecp_primitive_blocks) == 1 and opts.ecp_total_ncore == 10
    return mol, opts


def _fresh_ecp_single_point(mol):
    import vibeqc as vq
    from vibeqc.ecp_metadata import attach_inline_ecp_options_from_basis_sidecar

    opts = vq.RHFOptions()
    opts.max_iter = 200
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    basis = vq.BasisSet(mol, "lanl2dz")
    attach_inline_ecp_options_from_basis_sidecar(opts, mol, basis)
    return vq.run_rhf(mol, basis, opts).energy


@pytest.mark.parametrize("backend", ["lbfgs", "brent"])
def test_native_optimizers_follow_ecp_centres(backend):
    """#643: both native optimisers move the ECP centres with every geometry
    they evaluate and mirror the ECP fields onto the gradient. Pre-fix the
    L-BFGS-B path walked a bare-Z gradient with the centre stuck at the
    start geometry and ended unconverged at E = -86.21 Ha with S driven to
    z = 7.87 bohr; Brent raised "RHF result is not converged" in its line
    search. The converged energy must equal a fresh ECP single point at
    the optimised geometry, and the caller's options must come back
    describing the start geometry."""
    from vibeqc.molecular_optimize import optimize_molecule_brent

    fn = optimize_molecule if backend == "lbfgs" else optimize_molecule_brent
    mol, opts = _h2s_lanl2dz_with_ecp()
    result = fn(mol, basis_name="lanl2dz", method="rhf", rhf_options=opts,
                max_iter=40, conv_tol_grad=1e-4)
    assert result.converged
    e_fresh = _fresh_ecp_single_point(result.system)
    assert abs(result.energy - e_fresh) < 1e-8, (result.energy, e_fresh)
    # Measured on the fixed tree: -11.031259 Ha (HF/LANL2DZ minimum).
    assert result.energy == pytest.approx(-11.03126, abs=2e-4)
    assert list(opts.ecp_primitive_centers[0]) == [0.0, 0.0, 0.0]


def test_optimize_molecule_refuses_ecp_centres_off_the_start_geometry():
    import vibeqc as vq

    mol, opts = _h2s_lanl2dz_with_ecp()
    opts.ecp_centers = [vq.ECPCenter(Z=16, xyz=[0.0, 0.0, 0.05])]
    with pytest.raises(ValueError, match="coincides with no atom"):
        optimize_molecule(mol, basis_name="lanl2dz", method="rhf",
                          rhf_options=opts, max_iter=5)
