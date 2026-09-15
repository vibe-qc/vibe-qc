"""Selected-CI as an active-space CASCI kernel (roadmap 25i, M11).

Oracle methodology: the dense determinant CASCI is the exact reference.

* Full-selection limit (thresholds ~ 0, unlimited target) must reproduce
  the dense CASCI roots to numerical precision, closed- and open-shell.
* Truncated runs are variational (>= the dense ground state) and improve
  monotonically with target_size.
* 1-/2-RDMs built from a truncated wavefunction must reproduce the
  variational energy exactly (the spill-aware generator products in
  _rdm.py: out-of-list components of E_pq|Psi> contribute to the 2-RDM
  through intermediate dots, and dropping them is a real, testable error).
* 3-/4-RDM builders reject truncated lists with a clear error.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    build_hamiltonian_mo,
    casci,
    energy_from_rdms,
    get_hf_orbital_provider,
    make_rdm1,
    make_rdm12,
    make_rdm123,
    make_rdm1234,
)
from vibeqc.solvers._casci import _frozen_core_dressing
from vibeqc.solvers._selected_ci import SelectedCIOptions, selected_casci

H2O = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -0.93]),
        Atom(1, [0.0, -1.43, -0.93]),
    ]
)
LIH = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])])
LI = Molecule([Atom(3, [0.0, 0.0, 0.0])], multiplicity=2)


def _ham(mol, basis_name, method="rhf"):
    b = BasisSet(mol, basis_name)
    c = get_hf_orbital_provider(mol, b, method=method)
    return build_hamiltonian_mo(mol, b, c)


def _exact_options(**kw):
    """Options that drive the selection to the exact (full-CI) limit."""
    base = dict(
        target_size=100000,
        max_iter=80,
        conv_tol_energy=1e-13,
        pt2_threshold=1e-14,
        max_det_per_iter=100000,
        significant_coeff=1e-9,
    )
    base.update(kw)
    return SelectedCIOptions(**base)


class TestSelectedCASCIOracle:
    def test_full_selection_limit_matches_dense_casci_closed_shell(self):
        H = _ham(H2O, "sto-3g")
        ref = casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, nroots=2,
        )
        sel = selected_casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, nroots=2,
            options=_exact_options(),
        )
        assert np.abs(np.asarray(sel.e_totals) - ref.e_totals).max() < 1e-10
        assert sel.n_det <= ref.n_det  # symmetry-decoupled dets never enter

    def test_full_selection_limit_matches_dense_casci_open_shell(self):
        # Li doublet, CAS(3,4): M_s = 1/2 sector, 24 determinants.
        H = _ham(LI, "sto-3g", method="uhf")
        ref = casci(
            H.h1e, H.h2e, 3, 4, 0,
            nuclear_repulsion=H.nuclear_repulsion, ms2=1,
        )
        sel = selected_casci(
            H.h1e, H.h2e, 3, 4, 0,
            nuclear_repulsion=H.nuclear_repulsion, ms2=1,
            options=_exact_options(),
        )
        assert abs(sel.e_total - ref.e_total) < 1e-10

    def test_truncation_is_variational_and_monotone(self):
        H = _ham(H2O, "sto-3g")
        ref = casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
        )
        prev = np.inf
        for ts in (4, 8, 16, 36):
            sel = selected_casci(
                H.h1e, H.h2e, 4, 4, 3,
                nuclear_repulsion=H.nuclear_repulsion, ms2=0,
                options=_exact_options(target_size=ts),
            )
            assert sel.e_total >= ref.e_total - 1e-12  # variational
            assert sel.e_total <= prev + 1e-12         # monotone in ts
            prev = sel.e_total
        assert abs(prev - ref.e_total) < 1e-10  # ts=36 covers the space

    def test_warm_start_from_det_guess(self):
        H = _ham(H2O, "sto-3g")
        opts = _exact_options()
        cold = selected_casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, options=opts,
        )
        warm = selected_casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, options=opts,
            det_guess=list(cold.determinants),
        )
        assert abs(warm.e_total - cold.e_total) < 1e-12
        # the guess space is retained (no re-discovery from scratch)
        assert set(cold.determinants) <= set(warm.determinants)

    def test_spin_completion_closes_under_swap(self):
        H = _ham(H2O, "sto-3g")
        sel = selected_casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
            options=_exact_options(target_size=20),
        )
        dets = set(sel.determinants)
        assert all((b, a) in dets for (a, b) in dets)

    def test_invalid_inputs_raise(self):
        H = _ham(H2O, "sto-3g")
        with pytest.raises(ValueError, match="nroots"):
            selected_casci(H.h1e, H.h2e, 4, 4, 3, ms2=0, nroots=0)
        with pytest.raises(ValueError, match="electron counts"):
            selected_casci(
                H.h1e, H.h2e, 4, 4, 3, ms2=0,
                det_guess=[((0,), (0,))],
            )


class TestTruncatedRDMs:
    """Spill-aware generator products: truncated-list 1-/2-RDMs are exact."""

    def _setup(self, target_size):
        H = _ham(H2O, "sto-3g")
        e_core, h1a = _frozen_core_dressing(H.h1e, H.h2e, 3, slice(3, 7))
        h2a = np.ascontiguousarray(H.h2e[3:7, 3:7, 3:7, 3:7])
        econst = e_core + H.nuclear_repulsion
        sel = selected_casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
            options=_exact_options(target_size=target_size),
        )
        return sel, h1a, h2a, econst

    def test_truncated_rdm_energy_matches_variational(self):
        # ndet=7-ish space: NOT closed under E_pq, so the spill terms in
        # the 2-RDM are nonzero and load-bearing here.
        sel, h1a, h2a, econst = self._setup(6)
        from math import comb

        assert sel.n_det < comb(4, 2) ** 2  # genuinely truncated
        dm1, dm2 = make_rdm12(sel.ci_coeffs, sel.determinants, 4)
        e_rdm = energy_from_rdms(h1a, h2a, dm1, dm2, econst)
        assert abs(e_rdm - sel.e_total) < 1e-12
        assert abs(np.trace(dm1) - 4.0) < 1e-12
        # rdm1 from the dedicated entry point agrees
        assert np.allclose(
            make_rdm1(sel.ci_coeffs, sel.determinants, 4), dm1, atol=1e-14
        )

    def test_full_cas_rdm_regression_unchanged(self):
        # The spill path must be a no-op for closed (full-CAS) lists.
        H = _ham(H2O, "sto-3g")
        e_core, h1a = _frozen_core_dressing(H.h1e, H.h2e, 3, slice(3, 7))
        h2a = np.ascontiguousarray(H.h2e[3:7, 3:7, 3:7, 3:7])
        ref = casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
        )
        dm1, dm2 = make_rdm12(ref.ci_coeffs, ref.determinants, 4)
        e_rdm = energy_from_rdms(
            h1a, h2a, dm1, dm2, e_core + H.nuclear_repulsion
        )
        assert abs(e_rdm - ref.e_total) < 1e-11

    def test_rdm3_rdm4_reject_truncated_lists(self):
        sel, *_ = self._setup(6)
        with pytest.raises(NotImplementedError, match="full CAS"):
            make_rdm123(sel.ci_coeffs, sel.determinants, 4)
        with pytest.raises(NotImplementedError, match="full CAS"):
            make_rdm1234(sel.ci_coeffs, sel.determinants, 4)


class TestSelectedCICASSCF:
    """casscf(ci_solver='selected_ci'): the large-active-space CASSCF route.

    H2O can reach different orbital basins even in the full-selection limit
    (#56), so CI parity and truncation errors are measured at fixed optimized
    orbitals. LiH also checks parity of the independently optimized SA roots.
    """

    @pytest.mark.parametrize("orbital_tilt", [0.0, 1e-12])
    def test_exact_limit_matches_determinant_casscf(self, orbital_tilt):
        from vibeqc.solvers import casscf
        from vibeqc.solvers._casscf import _rotate_integrals

        H = _ham(H2O, "sto-3g")
        # Equivalent starting orbitals can select different CASSCF stationary
        # points in this basin-rich system (#56). Exercise a tiny rotation as
        # well as the native HF orbitals, without changing the Hamiltonian.
        rotation = np.eye(H.h1e.shape[0])
        c, t = np.cos(orbital_tilt), np.sin(orbital_tilt)
        rotation[np.ix_([0, 3], [0, 3])] = [[c, -t], [t, c]]
        h1, h2 = _rotate_integrals(H.h1e, H.h2e, rotation)
        for backend in ("casci", "selected_ci"):
            result = casscf(
                h1, h2, n_active_elec=4, n_active_orb=4, n_core=3,
                nuclear_repulsion=H.nuclear_repulsion, ms2=0,
                ci_solver=backend,
                selected_ci_options=_exact_options() if backend == "selected_ci" else None,
            )
            assert result.converged
            assert result.grad_norm < 1e-6
            # Verify that the reported optimized basis belongs to the input
            # Hamiltonian, then compare both CI kernels in THAT same basis.
            rebuilt_h1, rebuilt_h2 = _rotate_integrals(h1, h2, result.mo_rotation)
            np.testing.assert_allclose(result.h1e_cas, rebuilt_h1, atol=1e-10, rtol=0)
            np.testing.assert_allclose(result.h2e_cas, rebuilt_h2, atol=1e-10, rtol=0)
            exact = casci(
                result.h1e_cas, result.h2e_cas, 4, 4, 3,
                nuclear_repulsion=H.nuclear_repulsion, ms2=0,
            )
            selected = selected_casci(
                result.h1e_cas, result.h2e_cas, 4, 4, 3,
                nuclear_repulsion=H.nuclear_repulsion, ms2=0,
                options=_exact_options(),
            )
            assert abs(result.e_total - exact.e_total) < 1e-10
            assert abs(selected.e_total - exact.e_total) < 1e-10

    def test_exact_limit_sa_matches_determinant_casscf(self):
        from vibeqc.solvers import casscf

        H = _ham(LIH, "sto-3g")
        ref = casscf(
            H.h1e, H.h2e, n_active_elec=2, n_active_orb=2, n_core=1,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, nroots=2,
        )
        sel = casscf(
            H.h1e, H.h2e, n_active_elec=2, n_active_orb=2, n_core=1,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, nroots=2,
            ci_solver="selected_ci", selected_ci_options=_exact_options(),
        )
        assert sel.converged
        assert abs(sel.e_total - ref.e_total) < 1e-10
        assert np.abs(np.asarray(sel.e_totals) - ref.e_totals).max() < 1e-10

    def test_truncated_is_variational_and_close(self):
        from vibeqc.solvers import casscf

        H = _ham(H2O, "sto-3g")
        sel = casscf(
            H.h1e, H.h2e, n_active_elec=4, n_active_orb=4, n_core=3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
            ci_solver="selected_ci",
            selected_ci_options=_exact_options(target_size=8),
        )
        assert sel.converged
        assert sel.grad_norm < 1e-6
        assert 0 < sel.cas.n_det < 36
        # The variational inequality compares subspaces of one Hamiltonian.
        # A separately optimized dense CASSCF can reach another orbital basin
        # and does not measure the selected-space truncation error (#56).
        ref = casci(
            sel.h1e_cas, sel.h2e_cas, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
        )
        gap = sel.e_total - ref.e_total
        assert -1e-10 <= gap < 1e-5, f"fixed-orbital truncation error: {gap} Ha"

    def test_invalid_backend_options_raise(self):
        from vibeqc.solvers import casscf

        H = _ham(LIH, "sto-3g")
        with pytest.raises(ValueError, match="ci_solver"):
            casscf(
                H.h1e, H.h2e, n_active_elec=2, n_active_orb=2, n_core=1,
                ms2=0, ci_solver="bogus",
            )


class TestSpinPureSelectedCI:
    """spin_pure + ci_solver="selected_ci" (M15a).

    Oracle: the dense ⟨S²⟩ filter (_ms_caspt2._spin_pure_roots).  At the
    full-selection limit the selected filter must reproduce it exactly;
    truncated runs must return spin-pure roots (αβ-swap closure blocks
    singlet–triplet mixing, so truncation contamination is tiny).  H2O
    CAS(4,4) SA2 is basin-rich across CI backends (#56), so full-selection
    energy parity uses the same optimized orbital basis. Numerical agreement
    of CI solves does not guarantee identical optimization trajectories.
    """

    def test_helper_full_limit_matches_dense_filter(self):
        # LiH CAS(2,2) M_s=0: 4 dets, the triplet interleaves between the
        # two singlets.  n_buf=4 > the 3-det natural seed, so this also
        # exercises the BFS seed augmentation.
        from vibeqc.solvers._ms_caspt2 import _spin_pure_roots
        from vibeqc.solvers._selected_ci import _spin_pure_selected_roots

        H = _ham(LIH, "sto-3g")
        _, e_dense, s2_dense, _ = _spin_pure_roots(
            H.h1e, H.h2e, 1, 2, 2, 0, 2,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        _, e_sel, s2_sel, buf = _spin_pure_selected_roots(
            H.h1e, H.h2e, 2, 2, 1,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
            nroots=2, options=_exact_options(),
        )
        assert buf.n_det == 4  # augmented to the full space
        assert np.abs(np.asarray(e_sel) - e_dense).max() < 1e-12
        assert max(abs(s) for s in s2_sel) < 1e-12

    def test_casscf_full_limit_matches_dense_spin_pure(self):
        from vibeqc.solvers import casscf
        from vibeqc.solvers._ms_caspt2 import _s2_expectation, _spin_pure_roots

        H = _ham(H2O, "sto-3g")
        sel = casscf(
            H.h1e, H.h2e, n_active_elec=4, n_active_orb=4, n_core=3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, nroots=2,
            spin_pure=True, ci_solver="selected_ci",
            selected_ci_options=_exact_options(),
        )
        assert sel.converged
        assert sel.grad_norm < 1e-6
        _, e_dense, _, _ = _spin_pure_roots(
            sel.h1e_cas, sel.h2e_cas, 3, 4, 4, 0, 2,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        assert np.abs(np.asarray(sel.e_totals) - e_dense).max() < 5e-9
        s2 = [
            _s2_expectation(sel.cas.ci_coeffs_all[:, k], sel.cas.determinants, 4, 0)
            for k in range(2)
        ]
        assert max(abs(s) for s in s2) < 1e-8

    def test_casscf_truncated_roots_stay_spin_pure(self):
        from vibeqc.solvers import casscf
        from vibeqc.solvers._ms_caspt2 import _s2_expectation

        H = _ham(H2O, "sto-3g")
        sel = casscf(
            H.h1e, H.h2e, n_active_elec=4, n_active_orb=4, n_core=3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, nroots=2,
            spin_pure=True, ci_solver="selected_ci",
            selected_ci_options=_exact_options(target_size=20),
        )
        assert sel.converged
        assert sel.cas.n_det < 36  # genuinely truncated
        s2 = [
            _s2_expectation(sel.cas.ci_coeffs_all[:, k], sel.cas.determinants, 4, 0)
            for k in range(2)
        ]
        # αβ-swap closure blocks the adjacent (triplet) sector; residual
        # contamination is ≥-2-sectors-away and tiny.
        assert max(abs(s) for s in s2) < 1e-6
        # Sanity window only: basin-rich system, no dense-energy pin.
        assert abs(sel.e_totals[0] - (-74.95)) < 0.05

    def test_runner_composition_matches_openmolcas(self, tmp_path):
        # End-to-end: run_job spin-pure selected SA2-CASSCF on LiH lands
        # in OpenMolcas's singlet-averaged RASSCF solution (same recorded
        # constants as the dense spin-pure pin in tests/test_ms_caspt2.py).
        from vibeqc.runner import run_job
        from vibeqc.solvers import CASSCFOptions

        om_sa2 = [-7.85516481, -7.72558848]
        res = run_job(
            LIH, basis="sto-3g", method="casscf", active_space=(2, 2),
            output=tmp_path / "spsel", citations=False,
            write_molden_file=False, write_xyz_file=False,
            write_population_file=False,
            casscf_options=CASSCFOptions(
                nroots=2, spin_pure=True, ci_solver="selected_ci",
                selected_ci_options=_exact_options(),
            ),
        )
        assert res.method == "casscf(2e,2o)_selci_sa2"
        assert res.converged
        assert np.abs(np.asarray(res.root_energies) - om_sa2).max() < 5e-6


def _cpp_available():
    try:
        from vibeqc._vibeqc_core import selected_ci_solve  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _cpp_available(), reason="C++ selected-CI not built")
class TestCppSelectedCIBackend:
    """C++ kernel (cpp/src/selected_ci.cpp) vs the Python oracle."""

    def test_cpp_full_limit_matches_dense(self, monkeypatch):
        monkeypatch.setenv("VIBEQC_SELECTED_CI_BACKEND", "cpp")
        H = _ham(H2O, "sto-3g")
        ref = casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, nroots=2,
        )
        sel = selected_casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, nroots=2,
            options=_exact_options(),
        )
        assert np.abs(np.asarray(sel.e_totals) - ref.e_totals).max() < 1e-10

    def test_cpp_full_limit_open_shell(self, monkeypatch):
        monkeypatch.setenv("VIBEQC_SELECTED_CI_BACKEND", "cpp")
        H = _ham(LI, "sto-3g", method="uhf")
        ref = casci(
            H.h1e, H.h2e, 3, 4, 0,
            nuclear_repulsion=H.nuclear_repulsion, ms2=1,
        )
        sel = selected_casci(
            H.h1e, H.h2e, 3, 4, 0,
            nuclear_repulsion=H.nuclear_repulsion, ms2=1,
            options=_exact_options(),
        )
        assert abs(sel.e_total - ref.e_total) < 1e-10

    def test_cpp_matches_python_at_fixed_thresholds(self, monkeypatch):
        # identical CIPSI criterion + deterministic tie-free selection on
        # this system: the two backends grow the same space and agree to
        # solver precision
        H = _ham(H2O, "sto-3g")
        opts = _exact_options(target_size=20)
        monkeypatch.setenv("VIBEQC_SELECTED_CI_BACKEND", "python")
        py = selected_casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, options=opts,
        )
        monkeypatch.setenv("VIBEQC_SELECTED_CI_BACKEND", "cpp")
        cp = selected_casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, options=opts,
        )
        assert abs(cp.e_total - py.e_total) < 1e-9

    def test_cpp_rdm12_matches_python_on_truncated_list(self, monkeypatch):
        # same truncated wavefunction through both RDM implementations:
        # the C++ pair-based Slater-Condon build vs the spill-aware
        # generator products
        from vibeqc._vibeqc_core import selected_ci_rdm12
        from vibeqc.solvers._selected_ci import _dets_to_masks

        monkeypatch.setenv("VIBEQC_SELECTED_CI_BACKEND", "python")
        monkeypatch.setenv("VIBEQC_RDM_BACKEND", "python")
        H = _ham(H2O, "sto-3g")
        sel = selected_casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
            options=_exact_options(target_size=6),
        )
        dm1_py, dm2_py = make_rdm12(sel.ci_coeffs, sel.determinants, 4)
        ma, mb = _dets_to_masks(sel.determinants)
        dm1_cpp, dm2_cpp = selected_ci_rdm12(
            ma, mb, np.ascontiguousarray(sel.ci_coeffs), 4
        )
        assert np.abs(np.asarray(dm1_cpp) - dm1_py).max() < 1e-13
        assert np.abs(np.asarray(dm2_cpp) - dm2_py).max() < 1e-13

    def test_cpp_casscf_exact_limit(self, monkeypatch):
        from vibeqc.solvers import casscf

        monkeypatch.setenv("VIBEQC_SELECTED_CI_BACKEND", "cpp")
        H = _ham(LIH, "sto-3g")
        ref = casscf(
            H.h1e, H.h2e, n_active_elec=2, n_active_orb=2, n_core=1,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
        )
        sel = casscf(
            H.h1e, H.h2e, n_active_elec=2, n_active_orb=2, n_core=1,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
            ci_solver="selected_ci", selected_ci_options=_exact_options(),
        )
        assert sel.converged
        assert abs(sel.e_total - ref.e_total) < 1e-9

    def test_stall_trigger_switches_and_converges(self):
        # The M18 auto-mode stall trigger: with stall_to_nr large enough
        # to fire on ANY progress rate, the switch happens at the first
        # eligible iteration and the optimization still lands on the
        # unique LiH minimum; with the trigger disabled (0), the
        # historical gradient-only switching is exactly preserved.
        from vibeqc.solvers import casscf

        H = _ham(LIH, "sto-3g")
        kw = dict(
            n_active_elec=2, n_active_orb=2, n_core=1,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
            ci_solver="selected_ci",
            selected_ci_options=_exact_options(),
        )
        forced = casscf(H.h1e, H.h2e, orbital_step="auto",
                        stall_to_nr=1e6, **kw)
        legacy = casscf(H.h1e, H.h2e, orbital_step="auto",
                        stall_to_nr=0.0, **kw)
        assert forced.converged and legacy.converged
        assert abs(forced.e_total - legacy.e_total) < 1e-9

    def test_cpp_spin_pure_casscf_full_limit(self, monkeypatch):
        # Spin-pure SA2 through the C++ kernel equals dense spin-pure CI
        # at the same optimized orbitals in the full-selection limit.
        from vibeqc.solvers import casscf
        from vibeqc.solvers._ms_caspt2 import _s2_expectation, _spin_pure_roots

        monkeypatch.setenv("VIBEQC_SELECTED_CI_BACKEND", "cpp")
        H = _ham(H2O, "sto-3g")
        sel = casscf(
            H.h1e, H.h2e, n_active_elec=4, n_active_orb=4, n_core=3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, nroots=2,
            spin_pure=True, ci_solver="selected_ci",
            selected_ci_options=_exact_options(),
        )
        assert sel.converged
        assert sel.grad_norm < 1e-6
        _, e_dense, _, _ = _spin_pure_roots(
            sel.h1e_cas, sel.h2e_cas, 3, 4, 4, 0, 2,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        assert np.abs(np.asarray(sel.e_totals) - e_dense).max() < 5e-9
        s2 = [
            _s2_expectation(sel.cas.ci_coeffs_all[:, k], sel.cas.determinants, 4, 0)
            for k in range(2)
        ]
        assert max(abs(s) for s in s2) < 1e-8

    def test_invalid_backend_env_raises(self, monkeypatch):
        monkeypatch.setenv("VIBEQC_SELECTED_CI_BACKEND", "bogus")
        H = _ham(LIH, "sto-3g")
        with pytest.raises(ValueError, match="VIBEQC_SELECTED_CI_BACKEND"):
            selected_casci(
                H.h1e, H.h2e, 2, 2, 1,
                nuclear_repulsion=H.nuclear_repulsion, ms2=0,
            )


N2 = Molecule([Atom(7, [0.0, 0.0, 0.0]), Atom(7, [0.0, 0.0, 2.074])])


@pytest.mark.skipif(not _cpp_available(), reason="C++ selected-CI not built")
class TestHeatBathWalk:
    """Heat-bath presorted candidate walks (M16).

    The walk's keep/drop predicate is the bit-identical float expression
    the brute-force prefilter applies per candidate, and the presorted
    magnitudes are computed by the same arithmetic as the matrix
    elements, so walk and brute prefilter accept exactly the same
    candidate set at any select_eps; only enumeration cost changes.
    Caveat pinned here with full-keep settings (max_new_per_cycle >=
    all scored): with a binding room cut, score TIES at the boundary
    are broken by hash-map iteration order, which legitimately differs
    between enumeration orders (same situation as python-vs-cpp).
    """

    def _n2_active(self):
        from vibeqc.solvers._casci import _frozen_core_dressing

        H = _ham(N2, "sto-3g")
        # CAS(10,8): C(8,5)^2 = 3136 determinants
        e_core, h1a = _frozen_core_dressing(H.h1e, H.h2e, 2, slice(2, 10))
        h2a = np.ascontiguousarray(H.h2e[2:10, 2:10, 2:10, 2:10])
        return np.ascontiguousarray(h1a), h2a

    def _solve_direct(self, h1a, h2a, eps, walk):
        from vibeqc._vibeqc_core import (
            SelectedCIOptionsCpp,
            selected_ci_solve,
        )

        eri_chem = np.ascontiguousarray(h2a.transpose(0, 2, 1, 3))
        o = SelectedCIOptionsCpp()
        o.nroots = 2
        o.max_cycles = 60
        o.target_size = 100000
        o.max_new_per_cycle = 100000  # full keep: no room-cut tie races
        o.conv_tol_energy = 1e-13
        o.pt2_threshold = 1e-12
        o.significant_coeff = 1e-7
        o.select_eps = eps
        o.use_heat_bath_walk = walk
        ga = np.empty(0, dtype=np.uint64)
        gb = np.empty(0, dtype=np.uint64)
        return selected_ci_solve(h1a, eri_chem, 8, 5, 5, o, ga, gb)

    @pytest.mark.parametrize("eps", [1e-4, 1e-3])
    def test_walk_equals_brute_prefilter(self, eps):
        h1a, h2a = self._n2_active()
        rw = self._solve_direct(h1a, h2a, eps, walk=True)
        rb = self._solve_direct(h1a, h2a, eps, walk=False)
        assert set(zip(rw.dets_a, rw.dets_b)) == set(zip(rb.dets_a, rb.dets_b))
        assert (
            max(
                abs(rw.eigenvalues[k] - rb.eigenvalues[k])
                for k in range(2)
            )
            < 1e-9
        )

    def test_eps_prunes_and_walk_stays_exact_about_it(self):
        # eps=1e-3 genuinely shrinks the selected space on this system
        # (the equivalence above would be vacuous otherwise), and the
        # eps cost on the variational energy stays at the documented
        # sub-mHa scale.
        h1a, h2a = self._n2_active()
        r0 = self._solve_direct(h1a, h2a, 0.0, walk=True)
        re = self._solve_direct(h1a, h2a, 1e-3, walk=True)
        assert len(re.dets_a) < len(r0.dets_a)
        assert abs(re.eigenvalues[0] - r0.eigenvalues[0]) < 1e-3
        assert re.eigenvalues[0] >= r0.eigenvalues[0] - 1e-12  # variational

    def test_pt2_deterministic_matches_dense_oracle(self):
        # Independent oracle: Eq. 4 of Sharma et al 2017 evaluated by
        # plain linear algebra over the FULL dense Hamiltonian
        # (perturber block H[C,V] @ c), vs the brute Python kernel vs
        # the C++ kernel, on a genuinely truncated 7-det space.
        from vibeqc.solvers._determinant import generate_determinants
        from vibeqc.solvers._selected_ci import (
            _dets_to_masks,
            _en_pt2_deterministic_python,
        )
        from vibeqc.solvers._slater_condon import (
            build_hamiltonian_matrix_unrestricted,
        )
        from vibeqc._vibeqc_core import (
            selected_ci_en_pt2_deterministic as cdet,
        )

        res, h1a, h2a, _ = TestTruncatedRDMs()._setup(6)
        dets_v = list(res.determinants)
        ci = np.asarray(res.ci_coeffs, float)
        e0 = res.e_total - res.e_core

        full = generate_determinants(4, 2, 2)
        h_full = build_hamiltonian_matrix_unrestricted(full, h1a, h2a)
        pos_v = [full.index(d) for d in dets_v]
        pos_c = [i for i in range(len(full)) if full[i] not in set(dets_v)]
        num = h_full[np.ix_(pos_c, pos_v)] @ ci
        haa = np.diag(h_full)[pos_c]
        mask = np.abs(e0 - haa) >= 1e-10
        e2_dense = float(np.sum(num[mask] ** 2 / (e0 - haa[mask])))
        assert e2_dense < -1e-7  # non-vacuous: the space truly couples out

        e2_py, n_py = _en_pt2_deterministic_python(
            dets_v, ci, e0, h1a, h2a, 4, 0.0
        )
        ma, mb = _dets_to_masks(dets_v)
        eri_chem = np.ascontiguousarray(h2a.transpose(0, 2, 1, 3))
        e2_cpp, n_cpp = cdet(h1a, eri_chem, 4, ma, mb, ci, e0, 0.0, True)
        assert abs(e2_py - e2_dense) < 1e-14
        assert abs(e2_cpp - e2_dense) < 1e-14
        assert n_py == n_cpp

        # eps2 pruning: walk == brute == python at the same threshold
        for eps2 in (1e-5, 1e-3):
            ew, nw = cdet(h1a, eri_chem, 4, ma, mb, ci, e0, eps2, True)
            eb, nb = cdet(h1a, eri_chem, 4, ma, mb, ci, e0, eps2, False)
            ep, np_ = _en_pt2_deterministic_python(
                dets_v, ci, e0, h1a, h2a, 4, eps2
            )
            # The presorted walk may discover perturbers in another order,
            # but it must retain the same set and reduce it bit-identically.
            assert nw == nb == np_
            assert ew.hex() == eb.hex()
            assert abs(ew - ep) < 1e-14

    def test_pt2_invariant_subspace_is_zero(self):
        # A selected space that exhausts the ground state's
        # symmetry-connected determinant block has EXACTLY zero coupling
        # to the complement: E2 = 0 identically (and the kernels agree
        # on it rather than producing numerical noise).
        from vibeqc.solvers import selected_ci_pt2

        H = _ham(H2O, "sto-3g")
        res = selected_casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
            options=_exact_options(target_size=10),
        )
        out = selected_ci_pt2(H.h1e, H.h2e, 4, 4, 3, result=res, eps2=0.0)
        assert abs(out[0]["e_pt2"]) < 1e-12

    def test_pt2_semistochastic_identities(self):
        from vibeqc.solvers._selected_ci import _dets_to_masks
        from vibeqc._vibeqc_core import (
            selected_ci_en_pt2_deterministic as cdet,
            selected_ci_en_pt2_stochastic as cstoch,
        )

        res, h1a, h2a, _ = TestTruncatedRDMs()._setup(6)
        ci = np.asarray(res.ci_coeffs, float)
        e0 = res.e_total - res.e_core
        ma, mb = _dets_to_masks(list(res.determinants))
        eri_chem = np.ascontiguousarray(h2a.transpose(0, 2, 1, 3))
        e2_det, _ = cdet(h1a, eri_chem, 4, ma, mb, ci, e0, 0.0, True)

        # eps2_loose == eps2: the batch difference vanishes identically
        # (zero stochastic noise; the Eq. 11 limit).
        mean0, sem0 = cstoch(
            h1a, eri_chem, 4, ma, mb, ci, e0, 1e-8, 1e-8, 20, 8, 42, True
        )
        assert mean0 == 0.0 and sem0 == 0.0

        # fixed seed => bitwise-reproducible estimate
        m1, s1 = cstoch(
            h1a, eri_chem, 4, ma, mb, ci, e0,
            0.0, float("inf"), 50, 6, 123, True,
        )
        m2, s2 = cstoch(
            h1a, eri_chem, 4, ma, mb, ci, e0,
            0.0, float("inf"), 50, 6, 123, True,
        )
        assert m1 == m2 and s1 == s2

        # pure stochastic (Eq. 10) is unbiased: with this fixed seed the
        # estimate sits well inside the error bar of the deterministic
        # value (the estimator's unbiasedness is verified analytically
        # in the kernel's docstring derivation; this pins the code).
        mean, sem = cstoch(
            h1a, eri_chem, 4, ma, mb, ci, e0,
            0.0, float("inf"), 2000, 12, 3, True,
        )
        assert abs(mean - e2_det) < 4.0 * sem

        # semistochastic combination (Eq. 11) reproduces the tight
        # deterministic answer within its (smaller) error bar.
        e2_loose, _ = cdet(h1a, eri_chem, 4, ma, mb, ci, e0, 1e-3, True)
        md, sd = cstoch(
            h1a, eri_chem, 4, ma, mb, ci, e0,
            0.0, 1e-3, 600, 6, 11, True,
        )
        assert abs((e2_loose + md) - e2_det) < 4.0 * sd

        # validation raises
        with pytest.raises(ValueError, match="n_samples"):
            cstoch(h1a, eri_chem, 4, ma, mb, ci, e0,
                   0.0, 1e-3, 1, 6, 0, True)
        with pytest.raises(ValueError, match="eps2_loose"):
            cstoch(h1a, eri_chem, 4, ma, mb, ci, e0,
                   1e-3, 1e-8, 10, 6, 0, True)

    def test_pt2_wrapper_multiroot_and_backends(self, monkeypatch):
        from vibeqc.solvers import selected_ci_pt2

        H = _ham(H2O, "sto-3g")
        res = selected_casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, nroots=2,
            options=_exact_options(target_size=12, max_det_per_iter=4),
        )
        monkeypatch.setenv("VIBEQC_SELECTED_CI_BACKEND", "cpp")
        out_c = selected_ci_pt2(H.h1e, H.h2e, 4, 4, 3, result=res, eps2=0.0)
        monkeypatch.setenv("VIBEQC_SELECTED_CI_BACKEND", "python")
        out_p = selected_ci_pt2(H.h1e, H.h2e, 4, 4, 3, result=res, eps2=0.0)
        assert len(out_c) == 2 == len(out_p)
        for k in range(2):
            assert abs(out_c[k]["e_pt2"] - out_p[k]["e_pt2"]) < 1e-13
            assert out_c[k]["n_perturbers"] == out_p[k]["n_perturbers"]
            assert (
                abs(
                    out_c[k]["e_total"]
                    - (res.e_totals[k] + out_c[k]["e_pt2"])
                )
                < 1e-12
            )
        # PT2 lowers each root (variational space misses correlation)
        assert all(o["e_pt2"] <= 1e-12 for o in out_c)
        # the python oracle rejects the stochastic mode
        with pytest.raises(NotImplementedError, match="C\\+\\+"):
            selected_ci_pt2(
                H.h1e, H.h2e, 4, 4, 3, result=res,
                eps2=1e-8, n_samples=4,
            )

    def test_python_prefilter_matches_cpp_walk(self, monkeypatch):
        # The Python kernel's brute prefilter is the oracle for the C++
        # walk: same predicate, same float expression, same candidate
        # sets, at an eps that genuinely prunes.
        H = _ham(N2, "sto-3g")
        opts = _exact_options(select_eps=1e-3)
        kw = dict(nuclear_repulsion=H.nuclear_repulsion, ms2=0, nroots=2,
                  options=opts)
        monkeypatch.setenv("VIBEQC_SELECTED_CI_BACKEND", "cpp")
        rc = selected_casci(H.h1e, H.h2e, 10, 8, 2, **kw)
        monkeypatch.setenv("VIBEQC_SELECTED_CI_BACKEND", "python")
        rp = selected_casci(H.h1e, H.h2e, 10, 8, 2, **kw)
        monkeypatch.setenv("VIBEQC_SELECTED_CI_BACKEND", "cpp")
        r0 = selected_casci(
            H.h1e, H.h2e, 10, 8, 2,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, nroots=2,
            options=_exact_options(),
        )
        assert rc.n_det < r0.n_det  # the eps actually prunes here
        assert set(rc.determinants) == set(rp.determinants)
        assert (
            np.abs(np.asarray(rc.e_totals) - rp.e_totals).max() < 1e-9
        )

    def test_beyond_direct_wall_cas14(self, monkeypatch):
        # N2/6-31G CAS(14,14): 11,778,624 determinants, 6x past the
        # direct determinant engine's 2M cap.  The selected CASCI must
        # run (auto-dispatches to C++ above the 1e5 full-space
        # threshold), descend monotonically with the selection budget,
        # and land in the recorded window (plateau ~-109.0000, recorded
        # 2026-06-11; HF-orbital reference basis).
        from math import comb

        monkeypatch.setenv("VIBEQC_SELECTED_CI_BACKEND", "auto")
        N2 = Molecule([Atom(7, [0.0, 0.0, 0.0]), Atom(7, [0.0, 0.0, 2.074])])
        H = _ham(N2, "6-31g")
        assert comb(14, 7) ** 2 > 2_000_000
        energies = []
        for ts, eps, sig in ((5000, 1e-8, 0.003), (20000, 1e-9, 0.001)):
            sel = selected_casci(
                H.h1e, H.h2e, 14, 14, 0,
                nuclear_repulsion=H.nuclear_repulsion, ms2=0,
                options=SelectedCIOptions(
                    target_size=ts, max_iter=40, conv_tol_energy=1e-9,
                    pt2_threshold=eps, max_det_per_iter=ts,
                    significant_coeff=sig,
                ),
            )
            energies.append(sel.e_total)
        assert energies[1] <= energies[0] + 1e-9  # monotone in the budget
        assert -109.01 < energies[1] < -108.99   # recorded plateau window

    @pytest.mark.slow
    def test_beyond_wall_cas14_selected_casscf_converges(self, monkeypatch):
        # The full beyond-wall CASSCF: N2/6-31G CAS(14,14) selected-CI
        # CASSCF with the frozen-selection Newton-CG converges in ~8
        # macro-iterations (~2 min; recorded 2026-06-11:
        # E = -109.07272740, |g| = 1.6e-5, 72.7 mHa below the HF-orbital
        # CASCI).  orbital_step="nr" pins the pure-NR path; since the
        # stall_to_nr trigger (M18) "auto" converges here too (recorded
        # 2026-06-11: 29 macro-iterations / 110 s to -109.07272812, the
        # same stationary point), where it previously plateaued in the
        # Super-CI phase above the gradient switch.
        from vibeqc.solvers import casscf

        monkeypatch.setenv("VIBEQC_SELECTED_CI_BACKEND", "auto")
        N2 = Molecule([Atom(7, [0.0, 0.0, 0.0]), Atom(7, [0.0, 0.0, 2.074])])
        H = _ham(N2, "6-31g")
        sel = casscf(
            H.h1e, H.h2e, n_active_elec=14, n_active_orb=14, n_core=0,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
            ci_solver="selected_ci",
            selected_ci_options=SelectedCIOptions(
                target_size=20000, max_iter=40, conv_tol_energy=1e-9,
                pt2_threshold=1e-9, max_det_per_iter=20000,
                significant_coeff=0.001,
            ),
            orbital_step="nr", max_macro=25, conv_tol_grad=1e-4,
        )
        assert sel.converged
        assert sel.grad_norm < 1e-4
        assert -109.080 < sel.e_total < -109.065  # recorded window
        assert sel.e_total < -109.00000692 - 0.05  # well below CASCI@HF


class TestRunJobSelectedCASSCF:
    def test_run_job_casscf_selected_backend(self, tmp_path):
        from vibeqc.runner import run_job
        from vibeqc.solvers import CASSCFOptions

        res = run_job(
            LIH,
            basis="sto-3g",
            method="casscf",
            active_space=(2, 2),
            casscf_options=CASSCFOptions(
                ci_solver="selected_ci",
                selected_ci_options=_exact_options(),
            ),
            output=tmp_path / "selci",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=True,
        )
        assert "_selci" in res.method
        assert res.converged
        # the CIPSI selection algorithm is cited when the CASSCF uses it
        bib = (tmp_path / "selci.bibtex").read_text()
        assert "huron_malrieu_cipsi_1973" in bib
        assert "holmes_tubman_umrigar_shci_2016" in bib
        # exact-limit selected backend == determinant backend through run_job
        ref = run_job(
            LIH,
            basis="sto-3g",
            method="casscf",
            active_space=(2, 2),
            output=tmp_path / "det",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
        )
        assert abs(res.energy - ref.energy) < 1e-10

    def test_run_job_selected_casscf_pt2_block(self, tmp_path):
        # The SHCI perturbative stage through run_job (M19a): truncated
        # N2 CAS(10,8) selected CASSCF + deterministic EN-PT2.  The
        # headline energy stays variational; the PT2 estimate surfaces
        # on SolverResult.selected_pt2, in the .out block, and the
        # Sharma-2017 citation reaches the references.
        from vibeqc.runner import run_job
        from vibeqc.solvers import (
            CASSCFOptions,
            SelectedCIOptions,
            SelectedCIPT2Options,
        )

        res = run_job(
            N2,
            basis="sto-3g",
            method="casscf",
            active_space=(8, 10),
            casscf_options=CASSCFOptions(
                orbital_step="nr",
                ci_solver="selected_ci",
                selected_ci_options=SelectedCIOptions(
                    target_size=200, max_iter=40,
                    conv_tol_energy=1e-12, pt2_threshold=1e-12,
                    max_det_per_iter=100, significant_coeff=1e-7,
                ),
                pt2=SelectedCIPT2Options(eps2=0.0),
            ),
            output=tmp_path / "n2pt2",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=True,
        )
        assert res.converged
        p = res.selected_pt2[0]
        assert p["e_pt2"] < -1e-5  # genuinely truncated: PT2 is negative
        assert p["n_perturbers"] > 0
        assert abs(p["e_total"] - (res.energy + p["e_pt2"])) < 1e-12
        out = (tmp_path / "n2pt2.out").read_text()
        assert "Epstein-Nesbet PT2 on the selected wavefunction" in out
        assert "E_var+PT2" in out
        bib = (tmp_path / "n2pt2.bibtex").read_text()
        assert "sharma_semistochastic_hci_2017" in bib

    def test_run_job_pt2_requires_selected_backend(self, tmp_path):
        from vibeqc.runner import run_job
        from vibeqc.solvers import CASSCFOptions, SelectedCIPT2Options

        with pytest.raises(ValueError, match="selected_ci"):
            run_job(
                LIH,
                basis="sto-3g",
                method="casscf",
                active_space=(2, 2),
                casscf_options=CASSCFOptions(
                    pt2=SelectedCIPT2Options(),
                ),
                output=tmp_path / "bad",
                write_molden_file=False,
                write_xyz_file=False,
                write_population_file=False,
                citations=False,
            )

    def test_mrci_composition_rejects_selected_backend(self, tmp_path):
        # MRCI re-solves the full CAS reference space downstream, so it
        # still requires the exact CI backend (unlike single-state
        # NEVPT2/CASPT2, which take a selected reference since M21-M23;
        # see tests/test_solvers_mrpt_parity.py::TestSelectedReferenceMRPT2).
        from vibeqc.runner import run_job
        from vibeqc.solvers import CASSCFOptions

        with pytest.raises(ValueError, match="exact CI"):
            run_job(
                LIH,
                basis="sto-3g",
                method="mrci",
                active_space=(2, 2),
                casscf_options=CASSCFOptions(ci_solver="selected_ci"),
                output=tmp_path / "bad",
                write_molden_file=False,
                write_xyz_file=False,
                write_population_file=False,
                citations=False,
            )

    def test_caspt2_composition_accepts_selected_backend(self, tmp_path):
        # Single-state CASPT2 now composes with the selected-CI CASSCF
        # reference (M22/M23): the job runs and the determinant-based
        # engine returns a PT2-lowered energy.
        from vibeqc.runner import run_job
        from vibeqc.solvers import CASSCFOptions

        res = run_job(
            LIH,
            basis="sto-3g",
            method="caspt2",
            active_space=(2, 2),
            casscf_options=CASSCFOptions(
                ci_solver="selected_ci",
                selected_ci_options=_exact_options(),
            ),
            output=tmp_path / "ok",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
        )
        assert res.method.endswith("_casscf")
        assert res.converged
