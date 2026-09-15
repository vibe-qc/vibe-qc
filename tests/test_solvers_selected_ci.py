"""End-to-end tests for the Selected-CI solver on small molecules."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    Hamiltonian,
    SelectedCIOptions,
    build_hamiltonian_mo,
    casci,
    get_hf_orbital_provider,
    solve_selected_ci,
)


@pytest.fixture
def h2_sto3g_ham() -> Hamiltonian:
    """H2 / STO-3G at R = 1.4 bohr, MO basis."""
    mol = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        charge=0,
        multiplicity=1,
    )
    basis = BasisSet(mol, "sto-3g")
    C = get_hf_orbital_provider(mol, basis)
    return build_hamiltonian_mo(mol, basis, C)


@pytest.fixture
def h4_sto3g_ham() -> Hamiltonian:
    """Rectangular H4 / STO-3G, R = 2.0 bohr side (strong correlation)."""
    r = 2.0
    mol = Molecule(
        [
            Atom(1, [0.0, 0.0, 0.0]),
            Atom(1, [r, 0.0, 0.0]),
            Atom(1, [0.0, r, 0.0]),
            Atom(1, [r, r, 0.0]),
        ],
        charge=0,
        multiplicity=1,
    )
    basis = BasisSet(mol, "sto-3g")
    C = get_hf_orbital_provider(mol, basis)
    return build_hamiltonian_mo(mol, basis, C)


def _n2_ccpvdz_cas66_ham() -> Hamiltonian:
    """N2 / cc-pVDZ CAS(6e, 6o) Hamiltonian (the GitLab #107 reproducer)."""
    ang = 1.8897259886
    mol = Molecule(
        [Atom(7, [0.0, 0.0, -0.55 * ang]), Atom(7, [0.0, 0.0, 0.55 * ang])],
        charge=0,
        multiplicity=1,
    )
    basis = BasisSet(mol, "cc-pvdz")
    C = get_hf_orbital_provider(mol, basis)
    H = build_hamiltonian_mo(mol, basis, C)
    return H.active_space(6, 6)


def _default_ladder_options(target_size: int) -> SelectedCIOptions:
    """Ladder rung options: variational energy only, the default basis."""
    return SelectedCIOptions(
        target_size=target_size,
        max_iter=20,
        conv_tol_energy=1e-8,
        do_pt2_correction=False,
        verbose=0,
    )


def test_default_ladder_varies_with_target_size():
    """GitLab #107: the default ladder must not flat-line (determinant view).

    The pre-fix default (``spin_restricted=True``) returned ndet=6 at
    target_size 6 and 16 at every larger rung.  This is the count-only view;
    it is *not* the guard -- the count moved once before while the energy
    stayed bit-identical (LEARNINGS L110).  The energy guard is
    :func:`test_default_ladder_is_energy_monotone_and_reaches_the_casci_oracle`.
    """
    H = _n2_ccpvdz_cas66_ham()
    results = {t: solve_selected_ci(H, _default_ladder_options(t)) for t in (6, 20, 50, 200)}

    ndets = {t: len(r.ci_labels) for t, r in results.items()}
    assert ndets[6] < ndets[20] < ndets[200]
    assert ndets[6] >= 6  # the small target is honoured
    for r in results.values():
        assert r.stop_reason in ("energy", "target_size", "space_exhausted")
        assert r.converged
    assert results[6].stop_reason == "target_size"


def test_default_ladder_is_energy_monotone_and_reaches_the_casci_oracle():
    """GitLab #107 guard: the default ladder's ENERGY must move toward CASCI.

    Under the pre-fix default (``spin_restricted=True``) the N2/cc-pVDZ
    CAS(6e,6o) ladder returned -108.9594672752 Ha at every target_size,
    +62.3 mHa above the CASCI oracle -109.0217750033 Ha, reported
    ``converged=True`` (measured 2026-09-03 at main 65201761f, the deck of
    the run_job tests below; verifier notes 11687 and 14004 on the issue).
    The unrestricted default is variational, monotone in the budget, and
    bit-identical to the oracle from target_size 100 on.

    The oracle is computed here, in the same test, from the same
    Hamiltonian, so the assertion is on the quantity the issue is about
    and not on a number pinned elsewhere.
    """
    H = _n2_ccpvdz_cas66_ham()
    ref = casci(H.h1e, H.h2e, 6, 6, 0, nuclear_repulsion=H.nuclear_repulsion, ms2=0)
    e_oracle = float(ref.e_total)

    energies = {}
    prev = float("inf")
    for t in (6, 20, 50, 100):
        r = solve_selected_ci(H, _default_ladder_options(t))
        e = float(r.energy)  # variational: no PT2 on these rungs
        assert e <= prev + 1e-10, (t, e, prev)  # monotone in the budget
        assert e >= e_oracle - 1e-10, (t, e, e_oracle)  # variational bound
        energies[t] = e
        prev = e

    # The ladder moves (the defect was a 0.0 Ha spread), and from t=50 on it
    # sits within 1 mHa of the CASCI value (the defect was +62.3 mHa).
    assert energies[6] - energies[100] > 1e-3, energies
    for t in (50, 100):
        assert abs(energies[t] - e_oracle) < 1e-3, (t, energies[t], e_oracle)


# N2/cc-pVDZ CAS(6e,6o) seniority-zero (DOCI) energy over all C(6,3) = 20
# closed-shell determinants, computed from the exact in-repo oracle
# (build_hamiltonian_matrix_unrestricted over ((occ), (occ)) SpinDets) at
# main f15db2d79, HF orbitals, OMP_NUM_THREADS=2; 36.7 mHa above the CASCI
# value -109.0217750033 Ha because every omitted configuration is open-shell.
N2_CAS66_DOCI_E = -108.9850687702
N2_CAS66_DOCI_NDET = 20


def test_spin_restricted_ladder_reports_exhaustion_not_convergence():
    """GitLab #107 / #639: the restricted basis climbs to its DOCI limit, then
    says it can go no further.

    ``spin_restricted=True`` walks closed-shell determinants over spatial
    orbitals coupled through the pair-move integral ``g_iiaa`` (#639).  On
    N2/cc-pVDZ CAS(6e,6o) that seniority-zero space holds C(6,3) = 20
    determinants, all coupled to the reference, so a run whose budget covers
    them reaches the DOCI energy and stops on ``target_size``; a budget the
    pool cannot fill is reported ``space_exhausted`` and NOT converged: the
    closed subspace cannot certify the open-shell configurations it omits
    (36.7 mHa here against CASCI).

    Before #639 the pair-move coupling was scored with the Brillouin formula
    and the ladder froze at ndet=6 and -108.9594672752 Ha at every budget.
    """
    H = _n2_ccpvdz_cas66_ham()

    def _run(t):
        opts = _default_ladder_options(t)
        opts.spin_restricted = True
        return solve_selected_ci(H, opts)

    r20 = _run(N2_CAS66_DOCI_NDET)
    assert len(r20.ci_labels) == N2_CAS66_DOCI_NDET
    assert r20.stop_reason == "target_size"
    assert r20.converged  # the budget was met; nothing to complain about
    assert r20.energy == pytest.approx(N2_CAS66_DOCI_E, abs=1e-9)

    r200 = _run(200)
    assert r200.stop_reason == "space_exhausted"
    assert r200.converged is False
    # The pool is the whole seniority-zero space: nothing left to admit.
    assert len(r200.ci_labels) == N2_CAS66_DOCI_NDET
    assert r200.energy == pytest.approx(r20.energy, abs=1e-12)


@pytest.mark.parametrize("spin_restricted", [False, True])
def test_space_exhausted_is_distinct_from_energy_convergence(spin_restricted):
    """GitLab #107: a run whose space cannot grow below target_size must
    say so, distinctly from an energy-converged run.

    H2/STO-3G: target_size=10 is unattainable in either basis (the S_z=0
    space holds 4 determinants, of which 2 couple to the ground state).
    With conv_tol_energy=0 the energy criterion can never fire, forcing the
    selection to run to exhaustion.  The exhausted *unrestricted* space is
    the exact answer (the FCI energy, pinned against the dense CASCI), so it
    is converged.  The exhausted *restricted* space happens to be exact too
    for two electrons (the singlet ground state is seniority-zero, so with
    the #639 pair-move coupling it is the FCI energy as well), but the solver
    cannot know that in general -- a closed-shell space cannot certify the
    open-shell configurations it omits -- so it is reported
    ``converged=False`` with the stop reason saying why.
    """
    mol = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        charge=0,
        multiplicity=1,
    )
    basis = BasisSet(mol, "sto-3g")
    C = get_hf_orbital_provider(mol, basis)
    H = build_hamiltonian_mo(mol, basis, C)
    opts = SelectedCIOptions(
        target_size=10,
        max_iter=10,
        conv_tol_energy=0.0,
        do_pt2_correction=False,
        spin_restricted=spin_restricted,
        verbose=0,
    )
    r = solve_selected_ci(H, opts)
    assert r.stop_reason == "space_exhausted"
    assert len(r.ci_labels) <= 2
    assert f"ndet={len(r.ci_labels)}" in r.method
    e_fci = float(
        casci(H.h1e, H.h2e, 2, 2, 0, nuclear_repulsion=H.nuclear_repulsion, ms2=0).e_total
    )
    if spin_restricted:
        assert r.converged is False
        # #639: both closed-shell determinants couple through g_0011 =
        # 0.18125791 Ha, so the 2-determinant space IS the FCI energy
        # (pre-fix it returned E_HF = -1.1167143251 Ha, 20.6 mHa too high).
        assert len(r.ci_labels) == 2
        assert r.energy == pytest.approx(e_fci, abs=1e-10)
    else:
        # A valid fixed point of the selection procedure AND the exact
        # answer in this space: the stop reason, not a bare converged=True,
        # tells the caller the requested target was never reached.
        assert r.converged
        assert len(r.ci_labels) == 2
        assert r.energy == pytest.approx(e_fci, abs=1e-10)


def test_restricted_closed_shell_space_carries_the_pair_move_coupling(h2_sto3g_ham):
    """GitLab #639 closure criterion: H2/STO-3G with both closed-shell
    determinants {sigma_g^2, sigma_u^2} in the restricted space returns the
    FCI energy -1.1372759436170652 Ha, not the HF energy.

    The two determinants differ by one pair (0 -> 1); the seniority-zero
    Hamiltonian couples them through g_0011 = (01|01) = K_01 = 0.18125791 Ha
    (Helgaker, Jorgensen & Olsen, Sec.1.4; Bytautas et al. 2011,
    doi:10.1063/1.3613706).  Pre-fix the restricted builder scored that pair
    move with the one-electron Brillouin formula (1.0e-16 Ha for HF
    orbitals), so the 2x2 matrix was diagonal and the lowest root was E_HF =
    -1.1167143250625702 Ha.  The unrestricted builder over the same two
    determinants as ((0,),(0,)) and ((1,),(1,)) is the exact oracle.
    """
    from vibeqc.solvers import (
        build_hamiltonian_matrix,
        build_hamiltonian_matrix_unrestricted,
        pair_excitation_matrix_element,
    )

    H = h2_sto3g_ham
    dets = [(0,), (1,)]
    Hr = build_hamiltonian_matrix(dets, H.h1e, H.h2e)
    Hu = build_hamiltonian_matrix_unrestricted([(d, d) for d in dets], H.h1e, H.h2e)
    assert Hr == pytest.approx(Hu, abs=1e-14)
    assert Hr[0, 1] == pytest.approx(0.18125791479311382, abs=1e-10)
    assert Hr[0, 1] == pair_excitation_matrix_element(0, 1, H.h2e)
    e_doci = float(np.linalg.eigvalsh(Hr)[0]) + H.nuclear_repulsion
    e_fci = float(
        casci(H.h1e, H.h2e, 2, 2, 0, nuclear_repulsion=H.nuclear_repulsion, ms2=0).e_total
    )
    assert e_doci == pytest.approx(-1.1372759436170652, abs=1e-10)
    assert e_doci == pytest.approx(e_fci, abs=1e-12)
    # ... and the solver route reaches it through the same coupling.
    opts = SelectedCIOptions(
        target_size=2, max_iter=5, conv_tol_energy=1e-10,
        do_pt2_correction=False, spin_restricted=True, verbose=0,
    )
    r = solve_selected_ci(H, opts)
    assert len(r.ci_labels) == 2
    assert r.energy == pytest.approx(e_fci, abs=1e-10)


class TestSelectedCI:
    def test_h2_improves_with_size(self, h2_sto3g_ham):
        """Energy should decrease (or stay same) as determinant space grows."""
        opts_small = SelectedCIOptions(
            target_size=2, max_iter=5, conv_tol_energy=1e-8, verbose=0
        )
        result_small = solve_selected_ci(h2_sto3g_ham, opts_small)
        assert result_small.converged

        opts_large = SelectedCIOptions(
            target_size=10, max_iter=10, conv_tol_energy=1e-8, verbose=0
        )
        result_large = solve_selected_ci(h2_sto3g_ham, opts_large)
        assert result_large.converged

        assert result_large.energy <= result_small.energy + 1e-10

    def test_h2_vs_hf(self, h2_sto3g_ham):
        """Selected-CI with target_size=1 = HF determinant."""
        opts = SelectedCIOptions(
            target_size=1,
            max_iter=1,
            conv_tol_energy=1e-10,
            do_pt2_correction=False,
            verbose=0,
        )
        result = solve_selected_ci(h2_sto3g_ham, opts)

        from vibeqc._vibeqc_core import RHFOptions, run_rhf

        mol = Molecule(
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
            charge=0,
            multiplicity=1,
        )
        basis = BasisSet(mol, "sto-3g")
        rhf_opts = RHFOptions()
        rhf_opts.conv_tol_energy = 1e-12
        rhf_opts.conv_tol_grad = 1e-10
        rhf_result = run_rhf(mol, basis, rhf_opts)

        assert result.energy == pytest.approx(rhf_result.energy, rel=1e-5)

    def test_h2_result_structure(self, h2_sto3g_ham):
        """Verify the result object carries expected fields."""
        opts = SelectedCIOptions(
            target_size=5, max_iter=5, conv_tol_energy=1e-6, verbose=0
        )
        result = solve_selected_ci(h2_sto3g_ham, opts)
        assert result.method.startswith("selected_ci")
        assert result.converged
        assert result.n_iter > 0
        assert len(result.energy_trace) > 0
        assert result.ci_coeffs is not None
        assert result.ci_labels is not None
        assert abs(result.ci_coeffs[0]) > 0.9

    def test_h4_multi_reference(self, h4_sto3g_ham):
        """Stretched H4 should show some multi-reference character
        (at least the reference coefficient is not exactly 1)."""
        opts = SelectedCIOptions(
            target_size=30,
            max_iter=12,
            conv_tol_energy=1e-5,
            do_pt2_correction=False,
            verbose=0,
        )
        result = solve_selected_ci(h4_sto3g_ham, opts)
        assert result.converged
        coeffs = np.abs(result.ci_coeffs)
        # The solver finds more than just the reference
        assert len(coeffs) >= 2
        # The leading coefficient is less than 0.9999 (others contribute)
        assert coeffs[0] < 0.9999

    def test_pt2_correction(self, h2_sto3g_ham):
        """PT2 correction should be present and lower the energy."""
        opts_with = SelectedCIOptions(
            target_size=5,
            max_iter=5,
            conv_tol_energy=1e-6,
            do_pt2_correction=True,
            verbose=0,
        )
        result_with = solve_selected_ci(h2_sto3g_ham, opts_with)

        opts_without = SelectedCIOptions(
            target_size=5,
            max_iter=5,
            conv_tol_energy=1e-6,
            do_pt2_correction=False,
            verbose=0,
        )
        result_without = solve_selected_ci(h2_sto3g_ham, opts_without)

        assert result_with.pt2_correction is not None
        assert result_with.energy <= result_without.energy + 1e-10

    def test_pt2_is_coherent_vs_dense_oracle(self):
        """The EN-PT2 uses coherent perturber numerators (2026-06-11 fix).

        Independent oracle: Eq. 4 of Sharma et al, JCTC 13, 1595 (2017)
        evaluated by plain linear algebra over the FULL dense
        Hamiltonian (perturber block H[C,V] @ c).  The pre-fix
        implementation summed (c_I H_aI)^2 per (generator, perturber)
        pair, neglecting cross-generator interference; on this system
        that overestimated |E_PT2| by 0.23 mHa.
        """
        from vibeqc.solvers._determinant import generate_determinants
        from vibeqc.solvers._slater_condon import (
            build_hamiltonian_matrix_unrestricted,
        )

        mol = Molecule(
            [
                Atom(8, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 1.43, -0.93]),
                Atom(1, [0.0, -1.43, -0.93]),
            ]
        )
        basis = BasisSet(mol, "sto-3g")
        C = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C)
        opts = SelectedCIOptions(
            target_size=12, max_iter=30, conv_tol_energy=1e-12,
            do_pt2_correction=True, spin_restricted=False, verbose=0,
        )
        r = solve_selected_ci(H, opts)
        dets_v = list(r.ci_labels)
        ci = np.asarray(r.ci_coeffs, float)
        e0_elec = r.energy - r.pt2_correction - H.nuclear_repulsion

        full = generate_determinants(H.norb, 5, 5)
        h_full = build_hamiltonian_matrix_unrestricted(full, H.h1e, H.h2e)
        pos_v = [full.index(d) for d in dets_v]
        pos_c = [
            i for i in range(len(full)) if full[i] not in set(dets_v)
        ]
        num = h_full[np.ix_(pos_c, pos_v)] @ ci
        haa = np.diag(h_full)[pos_c]
        mask = np.abs(e0_elec - haa) > 1e-12
        e2_dense = float(np.sum(num[mask] ** 2 / (e0_elec - haa[mask])))
        assert e2_dense < -1e-4  # non-vacuous coupling outside the space
        assert abs(r.pt2_correction - e2_dense) < 1e-13

    def test_oh_open_shell(self):
        """Selected-CI in unrestricted mode for OH radical."""
        mol = Molecule(
            [Atom(8, [0, 0, 0]), Atom(1, [0, 0, 1.834])],
            charge=0,
            multiplicity=2,
        )
        basis = BasisSet(mol, "sto-3g")
        C = get_hf_orbital_provider(mol, basis, method="uhf")
        H = build_hamiltonian_mo(mol, basis, C)

        opts = SelectedCIOptions(
            target_size=20,
            max_iter=10,
            spin_restricted=False,
            do_pt2_correction=False,
            verbose=0,
        )
        result = solve_selected_ci(H, opts)
        assert result.converged
        assert result.energy < 0  # bound state
        # Unrestricted determinant space should have both alpha and beta
        assert len(result.ci_labels) >= 2
        # First label should be a SpinDet (tuple of two tuples)
        first = result.ci_labels[0]
        assert isinstance(first, tuple) and len(first) == 2
        assert isinstance(first[0], tuple) and isinstance(first[1], tuple)


# ---------------------------------------------------------------------------
# Issue #107: the spin-restricted basis cannot reach a CAS with open-shell
# character, and must say so instead of reporting a clean convergence.
# ---------------------------------------------------------------------------


def test_spin_restricted_run_warns_that_the_cas_limit_is_unreachable(
    tmp_path, monkeypatch
):
    """A restricted run that cannot spend its budget must say why.

    `spin_restricted=True` (the closed-shell basis; the default is False
    since #107) walks closed-shell determinants over spatial orbitals,
    excited by whole pairs and coupled through g_iiaa (#639).  That subspace
    holds C(norb, nelec/2) determinants and cannot represent an open-shell
    configuration, so on a CAS whose solution has open-shell character it
    stops at the seniority-zero (DOCI) energy, not the CAS energy.

    Measured on N2/cc-pVDZ CAS(6e,6o), this deck: the restricted ladder
    fills its 20-determinant seniority-zero space at E = -108.9850687702 Ha
    (pre-#639 it froze at ndet=6, -108.9594672752 Ha), while the default
    reaches ndet=96 and E = -109.0217750033 Ha, the CASCI oracle.  With the
    default ``conv_tol_energy=1e-6`` the last admissions move the energy by
    2.5e-7 Ha, so the run stops on the ``energy`` criterion and is
    ``converged=True`` -- honestly: the energy converged within the space it
    was asked to search.  The warning must still fire, because 20 < 400
    determinants were spent, and it must say that the limit reached is the
    DOCI energy, not the CAS energy.  (The exhausted-pool variant, which
    forces ``conv_tol_energy=0`` and is reported ``converged=False`` with
    ``stop_reason="space_exhausted"``, is pinned at the solver level in
    ``test_spin_restricted_ladder_reports_exhaustion_not_convergence``.)
    """
    from vibeqc import run_job
    from vibeqc.solvers import SelectedCIOptions

    monkeypatch.chdir(tmp_path)
    n2 = Molecule(
        [Atom(7, [0.0, 0.0, 0.0]), Atom(7, [0.0, 0.0, 2.074])],
        charge=0,
        multiplicity=1,
    )
    r = run_job(
        n2,
        basis="cc-pvdz",
        method="selected_ci",
        active_space=(6, 6),
        selected_ci_options=SelectedCIOptions(target_size=400, spin_restricted=True),
        output="restricted",
    )
    out = (tmp_path / "restricted.out").read_text()

    # Behaviour first: the closed-shell ladder converged on energy at its
    # DOCI limit, 36.7 mHa above the CAS energy it cannot represent.
    assert r.converged is True
    assert r.stop_reason == "energy"
    assert "Converged:         True" in out
    assert "Stop reason:       energy" in out
    # ... then the warning that explains why that is not the CAS energy.
    assert "spin-restricted selected-CI stopped on 20 determinants" in out
    # The seniority-zero space of CAS(6e,6o) is C(6,3) = 20 determinants --
    # the number the user needs in order to see that the run spanned the
    # whole closed-shell subspace, not 20 of the 400-determinant CAS.
    assert "20 closed-shell (seniority-zero) determinants" in out
    assert "seniority-zero (DOCI) energy" in out
    assert "spin_restricted=False" in out
    # ... and the energy it stopped at is this deck's DOCI energy: the exact
    # seniority-zero oracle (the unrestricted builder over every closed-shell
    # determinant written as ((occ), (occ))) for the same 2.074 bohr geometry
    # and CAS(6e,6o); -108.991293194112 Ha, measured bit-identical.
    from vibeqc.solvers import (
        build_hamiltonian_matrix_unrestricted,
        build_hamiltonian_mo,
        generate_closed_shell_determinants,
        get_hf_orbital_provider,
    )

    basis = BasisSet(n2, "cc-pvdz")
    H = build_hamiltonian_mo(n2, basis, get_hf_orbital_provider(n2, basis)).active_space(6, 6)
    cs = generate_closed_shell_determinants(H.norb, 3)
    assert len(cs) == 20
    H_doci = build_hamiltonian_matrix_unrestricted([(d, d) for d in cs], H.h1e, H.h2e)
    e_doci = float(np.linalg.eigvalsh(H_doci)[0]) + H.nuclear_repulsion
    assert r.energy == pytest.approx(e_doci, abs=1e-9)
    assert e_doci == pytest.approx(-108.991293194112, abs=1e-8)


def test_default_run_does_not_warn_and_reaches_casci(tmp_path, monkeypatch):
    """The default basis reaches the answer, so nothing warns and the
    selected-CI total agrees with the CASCI route on the same deck.

    Guards against the warning degenerating into noise on every selected-CI
    run, and pins the SI-row shape: a default `run_job(method="selected_ci")`
    ladder rung at target_size 400 is the CAS energy, not +62 mHa above it.
    """
    from vibeqc import run_job
    from vibeqc.solvers import SelectedCIOptions

    monkeypatch.chdir(tmp_path)
    n2 = Molecule(
        [Atom(7, [0.0, 0.0, 0.0]), Atom(7, [0.0, 0.0, 2.074])],
        charge=0,
        multiplicity=1,
    )
    sci = run_job(
        n2,
        basis="cc-pvdz",
        method="selected_ci",
        active_space=(6, 6),
        selected_ci_options=SelectedCIOptions(target_size=400),
        output="default_sci",
    )
    cas = run_job(
        n2,
        basis="cc-pvdz",
        method="casci",
        active_space=(6, 6),
        output="default_cas",
    )
    out = (tmp_path / "default_sci.out").read_text()

    assert "spin-restricted selected-CI" not in out
    assert "Converged:         True" in out
    assert sci.converged
    assert abs(float(sci.energy) - float(cas.energy)) < 1e-5
