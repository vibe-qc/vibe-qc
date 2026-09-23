"""Initial-guess selection for SCF."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    InitialGuess,
    Molecule,
    RHFOptions,
    SpinlockMode,
    UHFOptions,
    UKSOptions,
    run_rhf,
    run_uhf,
    run_uks,
)
from vibeqc import _vibeqc_core as _core

from .conftest import GEOMETRIES, run_pyscf_rhf


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def _rhf_opts(guess: InitialGuess) -> RHFOptions:
    o = RHFOptions()
    o.conv_tol_energy = 1e-12
    o.conv_tol_grad = 1e-10
    o.initial_guess = guess
    return o


def _uhf_opts(guess: InitialGuess) -> UHFOptions:
    o = UHFOptions()
    o.conv_tol_energy = 1e-12
    o.conv_tol_grad = 1e-8
    o.max_iter = 300
    o.initial_guess = guess
    return o


def test_sad_and_hcore_give_same_rhf_energy_on_h2o():
    """SAD is an initial guess, not a new method — converged energy must
    match Hcore to machine precision."""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "sto-3g")
    r_sad = run_rhf(mol, basis, _rhf_opts(InitialGuess.SAD))
    r_hcore = run_rhf(mol, basis, _rhf_opts(InitialGuess.HCORE))
    assert r_sad.converged and r_hcore.converged
    assert abs(r_sad.energy - r_hcore.energy) < 1e-10


def test_sad_does_not_increase_rhf_iterations_on_h2o():
    """SAD should never be worse than Hcore on a well-behaved case."""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "sto-3g")
    r_sad = run_rhf(mol, basis, _rhf_opts(InitialGuess.SAD))
    r_hcore = run_rhf(mol, basis, _rhf_opts(InitialGuess.HCORE))
    assert r_sad.n_iter <= r_hcore.n_iter


def test_sad_fixes_oh_6_31gstar_uhf_false_minimum():
    """Documented failure mode of Hcore guess on OH/6-31G*: SCF converges to
    a local min ~0.16 Ha above the true minimum (PySCF: −75.3809309907).
    SAD should reach the correct minimum."""
    A2B = 1.0 / 0.529177210903
    atoms = [(8, [0.0, 0.0, 0.0]),
             (1, [0.97 * A2B, 0.0, 0.0])]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms], multiplicity=2)
    basis = BasisSet(mol, "6-31g*")
    r = run_uhf(mol, basis, _uhf_opts(InitialGuess.SAD))
    assert r.converged
    assert r.energy == pytest.approx(-75.3809309907, abs=1e-8)
    assert r.s_squared == pytest.approx(0.755, abs=1e-2)


def test_sad_uhf_preserves_rhf_match_on_closed_shell():
    """UHF with SAD guess on a closed-shell singlet must still recover RHF."""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "sto-3g")
    r_uhf = run_uhf(mol, basis, _uhf_opts(InitialGuess.SAD))
    r_rhf = run_rhf(mol, basis, _rhf_opts(InitialGuess.SAD))
    assert abs(r_uhf.energy - r_rhf.energy) < 1e-10
    assert abs(r_uhf.s_squared) < 1e-10


# ---- singlet spin symmetry of the open-shell guesses -------------------
#
# A spin-balanced (n_α == n_β) system must START spin-symmetric: the Hund
# split is for n_α ≠ n_β only (BUG 88), and a Hund-split singlet start
# (any atom with an open-shell Hund ground state — the O below) leaves
# convergence-limited α/β polarisation in the converged densities, breaks
# the UKS/RKS parity contract guarded by
# tests/test_rijcosx.py::test_rijcosx_singlet_uks_matches_rks, and defeats
# the spin-degenerate shared-exchange gate (1e-13 envelope) in uhf/uks.
# Deliberate broken-symmetry singlet starts go through ATOMSPIN.
# GitLab issue 19; regressed by the Hund-split commit 7a5ed0175.

def _singlet_h2o_sto3g():
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    return mol, BasisSet(mol, "sto-3g")


@pytest.mark.parametrize("guess", [InitialGuess.PATOM, InitialGuess.SAD])
def test_uhf_singlet_alpha_beta_density_parity(guess):
    """Default (PATOM) and SAD guesses keep a singlet UHF spin-degenerate:
    the converged α and β densities agree to the roundoff envelope."""
    mol, basis = _singlet_h2o_sto3g()
    r = run_uhf(mol, basis, _uhf_opts(guess))
    assert r.converged
    np.testing.assert_allclose(
        np.asarray(r.density_alpha), np.asarray(r.density_beta),
        rtol=0.0, atol=1e-12)


def test_uks_singlet_alpha_beta_density_parity_default_guess():
    """A singlet UKS with default options (PATOM guess) stays
    spin-degenerate — same contract as the RIJCOSX singlet parity test,
    on the plain (non-DF) path where the regression was root-caused."""
    mol, basis = _singlet_h2o_sto3g()
    o = UKSOptions()
    o.functional = "B3LYP"
    o.conv_tol_energy = 1e-9
    r = run_uks(mol, basis, o)
    assert r.converged
    np.testing.assert_allclose(
        np.asarray(r.density_alpha), np.asarray(r.density_beta),
        rtol=0.0, atol=1e-12)


# ---- SAP (Superposition of Atomic Potentials, Lehtola 2020) -----------


def test_sap_and_hcore_give_same_rhf_energy_on_h2o():
    """SAP is an initial guess — converged energy must match HCORE."""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "sto-3g")
    r_sap = run_rhf(mol, basis, _rhf_opts(InitialGuess.SAP))
    r_hcore = run_rhf(mol, basis, _rhf_opts(InitialGuess.HCORE))
    assert r_sap.converged and r_hcore.converged
    assert abs(r_sap.energy - r_hcore.energy) < 1e-10


def test_sap_and_sad_give_same_rhf_energy_on_h2o():
    """SAP and SAD are both initial guesses — same converged energy."""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "sto-3g")
    r_sap = run_rhf(mol, basis, _rhf_opts(InitialGuess.SAP))
    r_sad = run_rhf(mol, basis, _rhf_opts(InitialGuess.SAD))
    assert r_sap.converged and r_sad.converged
    assert abs(r_sap.energy - r_sad.energy) < 1e-10


def test_sap_converges_in_comparable_iterations_to_hcore_on_h2o():
    """SAP should converge in roughly the same iteration count as HCORE
    on small main-group systems. The DIIS-dominated regime hides most
    of SAP's advantage on H2O/STO-3G — we allow a small slack (±2)
    rather than asserting a strict improvement. SAP's quantitative win
    shows up on bigger atoms (Lehtola 2020 Tables I-II)."""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "sto-3g")
    r_sap = run_rhf(mol, basis, _rhf_opts(InitialGuess.SAP))
    r_hcore = run_rhf(mol, basis, _rhf_opts(InitialGuess.HCORE))
    assert r_sap.converged and r_hcore.converged
    assert r_sap.n_iter <= r_hcore.n_iter + 2


def test_sap_uhf_matches_rhf_on_closed_shell_h2o():
    """UHF with SAP guess on H2O singlet must recover RHF energy."""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "sto-3g")
    r_uhf = run_uhf(mol, basis, _uhf_opts(InitialGuess.SAP))
    r_rhf = run_rhf(mol, basis, _rhf_opts(InitialGuess.SAP))
    assert abs(r_uhf.energy - r_rhf.energy) < 1e-10
    assert abs(r_uhf.s_squared) < 1e-10


def test_sap_open_shell_guess_preserves_spin_electron_counts():
    """SAP orbitals must be occupied with the requested spin counts.

    Lehtola, Visscher, and Engel (JCP 152, 144105, 2020), Section II A,
    define one spin-independent SAP Hamiltonian whose orbitals seed the SCF.
    Occupying that common orbital set through ``n_alpha`` and ``n_beta`` must
    therefore give the exact per-spin populations (GitLab issue 666).
    """
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [1.8, 0.0, 0.0]),
        ],
        multiplicity=2,
    )
    basis = BasisSet(mol, "sto-3g")
    overlap = np.asarray(_core.compute_overlap(basis))
    hcore = np.asarray(_core.compute_kinetic(basis)) + np.asarray(
        _core.compute_nuclear(basis, mol)
    )
    jk = _core.make_direct_jk_builder(basis)

    density_alpha, density_beta = _core._guess_open_shell_density_with_jk(
        mol,
        basis,
        5,
        4,
        InitialGuess.SAP,
        overlap,
        hcore,
        jk,
        False,
    )

    n_alpha = float(np.trace(np.asarray(density_alpha) @ overlap))
    n_beta = float(np.trace(np.asarray(density_beta) @ overlap))
    assert n_alpha == pytest.approx(5.0, abs=1e-10)
    assert n_beta == pytest.approx(4.0, abs=1e-10)
    assert n_alpha + n_beta == pytest.approx(9.0, abs=1e-10)


def test_molecule_aware_auto_resolution_exposes_effective_guess():
    """The citation seam and production GuessEngine share one AUTO policy."""
    h2 = Molecule(
        [Atom(1, [0.0, 0.0, -0.7]), Atom(1, [0.0, 0.0, 0.7])]
    )
    hydrogen = Molecule([Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)
    iron = Molecule([Atom(26, [0.0, 0.0, 0.0])])

    resolve = _core._resolve_initial_guess_for_molecule
    assert resolve(h2, InitialGuess.AUTO, False, False) == InitialGuess.PATOM
    assert resolve(h2, InitialGuess.AUTO, False, True) == InitialGuess.SAD
    assert resolve(hydrogen, InitialGuess.AUTO, False, True) == InitialGuess.PATOM
    assert resolve(iron, InitialGuess.AUTO, False, False) == InitialGuess.SAD
    assert resolve(h2, InitialGuess.AUTO, True, False) == InitialGuess.SAD
    assert resolve(h2, InitialGuess.HCORE, False, False) == InitialGuess.HCORE


def test_auto_resolves_open_shell_d_block_to_patom():
    """Issue #273: an open-shell d/f-block complex needs the in-field step.

    The SAD construction superposes free-ATOM Hund densities, so every
    open-shell ligand arrives carrying its own free-atom moment -- one full
    unpaired electron per Cl in FeCl3, which in the molecule is a
    closed-shell chloride. That polarisation traps a symmetry-broken SCF
    solution 90.05 mHa above the ground state (measured on FeCl3 sextet /
    cc-pVDZ, internally unstable, S^2 = 8.776737). PATOM is the same Hund
    seed plus a per-spin in-field re-polarisation, and reaches the ORCA
    6.1.1 ground state; see tests/test_uhf_stability.py for the SCF pins.

    Closed-shell metal systems and open-shell main-group systems keep their
    established SAD / PATOM choices.
    """
    fecl3 = Molecule(
        [
            Atom(26, [0.0, 0.0, 0.0]),
            Atom(17, [4.0, 0.0, 0.0]),
            Atom(17, [-2.0, 3.5, 0.0]),
            Atom(17, [-2.0, -3.5, 0.0]),
        ],
        charge=0,
        multiplicity=6,
    )
    oh = Molecule(
        [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.83])], multiplicity=2
    )
    resolve = _core._resolve_initial_guess_for_molecule

    # Open-shell d block -> PATOM (the change).
    assert resolve(fecl3, InitialGuess.AUTO, False, True) == InitialGuess.PATOM
    # Closed-shell metal and open-shell main group are unchanged.
    assert resolve(fecl3, InitialGuess.AUTO, False, False) == InitialGuess.SAD
    assert resolve(oh, InitialGuess.AUTO, False, True) == InitialGuess.SAD
    # Periodic still resolves to SAD before the molecular rules run.
    assert resolve(fecl3, InitialGuess.AUTO, True, True) == InitialGuess.SAD
    # An explicit selector is never rewritten.
    assert resolve(fecl3, InitialGuess.SAD, False, True) == InitialGuess.SAD

    # A route without PATOM falls back to the advertised SAD construction.
    assert resolve(
        fecl3, InitialGuess.AUTO, False, True,
        [InitialGuess.HCORE, InitialGuess.SAD],
    ) == InitialGuess.SAD


def test_auto_keeps_atomic_spins_on_sad_for_a_metal_complex():
    """Issue #273: an ATOMSPIN seed is defined on the SAD density.

    The open-shell d/f rule above must not take an antiferromagnetically
    seeded metal complex to PATOM, which would then reject its own seed.
    """
    from vibeqc.guess import select_initial_guess

    fe2 = Molecule(
        [Atom(26, [0.0, 0.0, 0.0]), Atom(26, [0.0, 0.0, 4.5])],
        charge=0,
        multiplicity=3,
    )
    seeded = select_initial_guess(
        fe2, InitialGuess.AUTO, is_open_shell=True, atomic_spins=[1, -1],
    )
    assert seeded.effective == InitialGuess.SAD
    assert seeded.transport == InitialGuess.SAD

    unseeded = select_initial_guess(fe2, InitialGuess.AUTO, is_open_shell=True)
    assert unseeded.effective == InitialGuess.PATOM


def test_raw_open_shell_auto_matches_patom_for_isolated_atom():
    """The raw builder applies the isolated-atom AUTO policy end to end.

    Oxygen/STO-3G is deliberately used instead of hydrogen: its PATOM
    re-polarisation differs from the unrefined SAD density, so this catches
    both a stale AUTO resolution inside ``build_open_shell`` and failure to
    forward the JK builder needed by the effective PATOM guess.
    """
    mol = Molecule([Atom(8, [0.0, 0.0, 0.0])], multiplicity=3)
    basis = BasisSet(mol, "sto-3g")
    overlap = np.asarray(_core.compute_overlap(basis))
    hcore = np.asarray(_core.compute_kinetic(basis)) + np.asarray(
        _core.compute_nuclear(basis, mol)
    )
    jk = _core.make_direct_jk_builder(basis)

    def build(kind: InitialGuess) -> tuple[np.ndarray, np.ndarray]:
        densities = _core._guess_open_shell_density_with_jk(
            mol,
            basis,
            5,
            3,
            kind,
            overlap,
            hcore,
            jk,
            False,
        )
        return tuple(np.asarray(density) for density in densities)

    auto = build(InitialGuess.AUTO)
    patom = build(InitialGuess.PATOM)
    sad = build(InitialGuess.SAD)

    assert _core._resolve_initial_guess_for_molecule(
        mol, InitialGuess.AUTO, False, True
    ) == InitialGuess.PATOM
    for actual, expected in zip(auto, patom):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    # Negative control: this fixture genuinely distinguishes the two policies.
    assert not np.allclose(sad[1], patom[1], rtol=0.0, atol=1e-8)


@pytest.mark.parametrize(
    ("value", "is_periodic", "is_open_shell", "expected"),
    [
        ("core", False, False, InitialGuess.HCORE),
        ("InitialGuess.HUCKEL", False, False, InitialGuess.HUECKEL),
        (InitialGuess.AUTO, True, False, InitialGuess.SAD),
        ("auto", False, True, InitialGuess.SAD),
    ],
)
def test_python_guess_resolver_uses_canonical_coercion_and_native_auto_policy(
    value, is_periodic, is_open_shell, expected
):
    """Every Python route shares aliases and AUTO policy with GuessEngine."""
    from vibeqc.guess import resolve_initial_guess

    mol = Molecule(
        [Atom(1, [0.0, 0.0, -0.7]), Atom(1, [0.0, 0.0, 0.7])]
    )
    assert resolve_initial_guess(
        mol,
        value,
        is_periodic=is_periodic,
        is_open_shell=is_open_shell,
    ) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, InitialGuess.SAD),
        ("core", InitialGuess.HCORE),
        (InitialGuess.SAP, InitialGuess.SAP),
        ("auto", InitialGuess.SAD),
    ],
)
def test_periodic_driver_guess_gate_returns_effective_kind(value, expected):
    """Direct drivers get one normalized, capability-checked selector."""
    from vibeqc.guess import _coerce_periodic_driver_guess

    supported = {
        InitialGuess.HCORE,
        InitialGuess.SAD,
        InitialGuess.SAP,
    }
    assert _coerce_periodic_driver_guess(
        value, driver="test-route", supported=supported
    ) == expected


def test_periodic_driver_guess_gate_rejects_recognized_unavailable_kind():
    """A pinned but unavailable guess must not silently become Hcore."""
    from vibeqc.guess import _coerce_periodic_driver_guess

    with pytest.raises(
        NotImplementedError,
        match=r"HUECKEL.*(supported|capabilit)",
    ):
        _coerce_periodic_driver_guess(
            "huckel",
            driver="test-route",
            supported={InitialGuess.HCORE, InitialGuess.SAP},
        )


def test_periodic_driver_guess_restart_canonicalizes_requested_selector():
    """Restart cannot conceal a malformed or unavailable selector."""
    from vibeqc.guess import _coerce_periodic_driver_guess

    supported = {InitialGuess.HCORE, InitialGuess.READ}
    assert _coerce_periodic_driver_guess(
        InitialGuess.HCORE,
        driver="test-route",
        supported=supported,
        restart_supplied=True,
    ) == InitialGuess.READ
    with pytest.raises(NotImplementedError, match="FRAGMO"):
        _coerce_periodic_driver_guess(
            InitialGuess.FRAGMO, driver="test-route", supported=supported,
            restart_supplied=True,
        )
    with pytest.raises(ValueError, match="unknown initial_guess='bogus'"):
        _coerce_periodic_driver_guess(
            "bogus",
            driver="test-route",
            supported=supported,
            restart_supplied=True,
        )


@pytest.mark.parametrize("guess", [InitialGuess.SAP, "huckel"])
def test_periodic_fock_guess_dispatch_returns_hermitian_fock(guess):
    """SAP and HUECKEL share the periodic Fock-mode dispatch point."""
    from vibeqc.guess import periodic_fock_guess_k

    vq, sysp, basis = _periodic_h2_sto3g()
    fock_k = periodic_fock_guess_k(
        sysp,
        basis,
        [np.zeros(3)],
        guess,
        lattice_opts=vq.LatticeSumOptions(),
    )
    assert fock_k is not None and len(fock_k) == 1
    assert fock_k[0].shape == (basis.nbasis, basis.nbasis)
    np.testing.assert_allclose(fock_k[0], fock_k[0].conj().T, atol=1e-12)


def test_periodic_fock_guess_dispatch_returns_none_for_density_mode():
    """Artifact dispatch returns None for SAD; it does not substitute Hcore."""
    from vibeqc.guess import periodic_fock_guess_k

    _vq, sysp, basis = _periodic_h2_sto3g()
    assert periodic_fock_guess_k(
        sysp, basis, [np.zeros(3)], InitialGuess.SAD
    ) is None


def test_sap_periodic_gamma_reaches_same_basin_as_sad():
    """Periodic SAP is wired into the Γ RHF driver (F_SAP = T + V_SAP via the
    lattice-summed SAP potential). The guess does not change the converged
    minimum, so SAP and SAD reach the same Γ-RHF energy. (He and Ne single-atom
    cells converge cleanly; tighter ionic cells are a separate SCF-convergence
    matter, not a guess one.)"""
    import vibeqc as vq
    import numpy as np

    def energy(guess, Z, a):
        sysp = vq.PeriodicSystem(3, a * np.eye(3), [vq.Atom(Z, [0.0, 0.0, 0.0])],
                                 charge=0, multiplicity=1)
        basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
        opts = vq.PeriodicRHFOptions()
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        r = vq.run_rhf_periodic_gamma(sysp, basis, opts)
        assert r.converged, f"{guess.name} did not converge"
        return r.energy

    for Z, a in [(2, 10.0), (10, 12.0)]:  # He, Ne
        e_sap = energy(vq.InitialGuess.SAP, Z, a)
        e_sad = energy(vq.InitialGuess.SAD, Z, a)
        assert abs(e_sap - e_sad) < 1e-7, f"Z={Z}: SAP {e_sap} vs SAD {e_sad}"


# ---- SAD transition-metal d-shell occupation -------------------------
#
# Regression for the 3d-row mis-occupation bug (2026-06-01): SAD assigned
# atomic occupations via a single global eigenvalue sort, which the
# 3d/4s/4p near-degeneracy makes unreliable — every 3d metal came out
# wrong (Ni 3d10 4s0 instead of 3d8 4s2; Sc/Ti spilled their d electrons
# into an empty 4p). Occupations are now pinned per angular-momentum
# channel from the neutral-atom ground-state configuration
# (cpp/src/guess.cpp ground_config_per_l / configuration_occupations).


def _sad_l_populations(Z: int, basis_name: str = "sto-3g"):
    """Mulliken (s, p, d) populations of the isolated-atom SAD density.

    The atomic SAD density is spherical, so Mulliken populations grouped
    by the AO angular momentum recover the per-l electron totals exactly.
    """
    mult = 1 if Z % 2 == 0 else 2  # only needs correct parity for the ctor
    mol = Molecule([Atom(Z, [0.0, 0.0, 0.0])], multiplicity=mult)
    basis = BasisSet(mol, basis_name)
    D = np.asarray(_core.sad_density(mol, basis))
    S = np.asarray(_core.compute_overlap(basis))
    ao_l = np.array([int(sh.l) for sh in basis.shells()
                     for _ in range(2 * int(sh.l) + 1)])
    diag = np.diag(D @ S)
    return (diag[ao_l == 0].sum(), diag[ao_l == 1].sum(),
            diag[ao_l == 2].sum())


@pytest.mark.parametrize(
    "Z, sym, exp_d",
    [
        (21, "Sc", 1),   # under-fill: 3d1 4s2 (bug gave 3d0 + spurious 4p)
        (22, "Ti", 2),   # under-fill: 3d2 4s2 (bug gave 3d0 + spurious 4p)
        (24, "Cr", 5),   # anomaly:    3d5 4s1
        (28, "Ni", 8),   # over-fill:  3d8 4s2 (bug gave 3d10 4s0)
        (29, "Cu", 10),  # anomaly:    3d10 4s1
    ],
)
def test_sad_3d_metal_d_population(Z, sym, exp_d):
    """3d-row SAD must reproduce the experimental d-shell occupation, and
    still conserve the electron count (trace D·S == Z)."""
    s, p, d = _sad_l_populations(Z)
    assert d == pytest.approx(exp_d, abs=1e-6), f"{sym}: 3d population"
    assert s + p + d == pytest.approx(Z, abs=1e-6), f"{sym}: electron count"


def test_sad_sc_ti_have_no_spurious_4p_occupation():
    """Sc/Ti previously dumped their 3d electrons into the empty 4p shell
    (p > 12). Per-l pinning keeps the Ar-core p-count at exactly 12."""
    for Z in (21, 22):
        _s, p, _d = _sad_l_populations(Z)
        assert p == pytest.approx(12.0, abs=1e-6)


def test_sad_main_group_occupation_unchanged():
    """Main-group SAD occupations must be unaffected by the per-l pin
    (their eigenvalue order already matched the ground configuration)."""
    for Z, exp_s, exp_p in [(6, 4, 2), (8, 4, 4), (10, 4, 6)]:  # C, O, Ne
        s, p, d = _sad_l_populations(Z)
        assert s == pytest.approx(exp_s, abs=1e-6)
        assert p == pytest.approx(exp_p, abs=1e-6)
        assert d == pytest.approx(0.0, abs=1e-6)


# ---- PATOM / HUECKEL / MINAO -----------------------------------------
#
# Three further molecular initial guesses, all density-mode:
#   * PATOM   — SAD density + in-field re-polarisation (ORCA PAtom).
#   * HUECKEL — parameter-free GWH / extended Hückel (Hoffmann 1963 +
#               Wolfsberg-Helmholz 1952 + Lehtola 2019).
#   * MINAO   — project free-atom densities built in the ANO-RCC minimal
#               basis onto the target basis (Knizia 2013 / ANO-RCC).
#
# The defining contract for any initial guess: it changes the SCF *path*,
# not the converged minimum. So each must reach the same total energy as
# HCORE to numerical precision.

_NEW_GUESSES = [InitialGuess.PATOM, InitialGuess.HUECKEL, InitialGuess.MINAO]


@pytest.mark.parametrize("guess", _NEW_GUESSES)
@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31g*"])
def test_new_guess_matches_hcore_rhf_h2o(guess, basis_name):
    """PATOM/HUECKEL/MINAO are initial guesses, not new methods — the
    converged RHF energy must match HCORE to machine precision."""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, basis_name)
    r = run_rhf(mol, basis, _rhf_opts(guess))
    r_hcore = run_rhf(mol, basis, _rhf_opts(InitialGuess.HCORE))
    assert r.converged and r_hcore.converged
    assert abs(r.energy - r_hcore.energy) < 1e-9


@pytest.mark.parametrize("guess", _NEW_GUESSES)
def test_new_guess_uhf_matches_rhf_on_closed_shell_h2o(guess):
    """UHF with each new guess on a closed-shell singlet must recover RHF
    (the symmetric per-spin start stays spin-restricted)."""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "sto-3g")
    r_uhf = run_uhf(mol, basis, _uhf_opts(guess))
    r_rhf = run_rhf(mol, basis, _rhf_opts(guess))
    assert abs(r_uhf.energy - r_rhf.energy) < 1e-9
    assert abs(r_uhf.s_squared) < 1e-8


def test_patom_reaches_oh_uhf_true_minimum():
    """PATOM is SAD-based, so like SAD it should reach the correct OH/6-31G*
    UHF minimum rather than the Hcore false minimum (PySCF: −75.3809309907)."""
    A2B = 1.0 / 0.529177210903
    atoms = [(8, [0.0, 0.0, 0.0]), (1, [0.97 * A2B, 0.0, 0.0])]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms], multiplicity=2)
    basis = BasisSet(mol, "6-31g*")
    r = run_uhf(mol, basis, _uhf_opts(InitialGuess.PATOM))
    assert r.converged
    assert r.energy == pytest.approx(-75.3809309907, abs=1e-7)
    assert r.s_squared == pytest.approx(0.755, abs=1e-2)


def _periodic_h2_sto3g():
    """Tiny 30-bohr H₂ cell + STO-3G used by the periodic-guess guards."""
    import vibeqc as vq
    import numpy as np
    lat = 30.0 * np.eye(3)
    sysp = vq.PeriodicSystem(3, lat, [vq.Atom(1, [0.0, 0.0, 0.0]),
                                      vq.Atom(1, [1.4, 0.0, 0.0])],
                             charge=0, multiplicity=1)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return vq, sysp, basis


@pytest.mark.parametrize(
    "guess, label",
    [(InitialGuess.SAP, "SAP"),
     (InitialGuess.PATOM, "PATOM"),
     (InitialGuess.HUECKEL, "HUECKEL"),
     (InitialGuess.MINAO, "MINAO")],
)
def test_periodic_guess_raises_notimplemented_at_python_edge(guess, label):
    """The Python periodic-driver guess seam (``initial_density_closed_shell``
    / ``initial_densities_open_shell``) intercepts a guess this seam cannot
    build from the supplied context and raises a clean ``NotImplementedError``
    naming the supported guesses, rather than the raw C++ ``RuntimeError`` from
    deep in the build. HUECKEL and MINAO are tested here without the periodic
    system/lattice context required by the closed-shell Python drivers."""
    from vibeqc.guess import (initial_density_closed_shell,
                              initial_densities_open_shell)
    _vq, sysp, basis = _periodic_h2_sto3g()
    mol = sysp.unit_cell_molecule()
    with pytest.raises(NotImplementedError,
                       match=f"{label} not yet implemented for periodic"):
        initial_density_closed_shell(mol, basis, 1, guess, is_periodic=True)
    with pytest.raises(NotImplementedError,
                       match=f"{label} not yet implemented for periodic"):
        initial_densities_open_shell(mol, basis, 1, 1, guess, is_periodic=True)


def test_periodic_minao_closed_python_seam_uses_driver_context():
    """Closed-shell Python periodic drivers can now pass the system/lattice
    context needed for the periodic MINAO projection."""
    from vibeqc.guess import initial_density_closed_shell
    vq, sysp, basis = _periodic_h2_sto3g()
    D = initial_density_closed_shell(
        sysp.unit_cell_molecule(),
        basis,
        1,
        InitialGuess.MINAO,
        is_periodic=True,
        periodic_system=sysp,
        lattice_opts=vq.LatticeSumOptions(),
    )
    assert D is not None and D.shape == (basis.nbasis, basis.nbasis)
    assert np.max(np.abs(D - D.T)) < 1e-10


def test_periodic_huckel_closed_python_seam_uses_driver_context():
    """Closed-shell Python periodic drivers can now pass the system/lattice
    context needed for the periodic HUECKEL Fock-mode density."""
    from vibeqc.guess import initial_density_closed_shell
    vq, sysp, basis = _periodic_h2_sto3g()
    D = initial_density_closed_shell(
        sysp.unit_cell_molecule(),
        basis,
        1,
        InitialGuess.HUECKEL,
        is_periodic=True,
        periodic_system=sysp,
        lattice_opts=vq.LatticeSumOptions(),
    )
    assert D is not None and D.shape == (basis.nbasis, basis.nbasis)
    assert np.max(np.abs(D - D.T)) < 1e-10


def test_periodic_sap_closed_python_seam_uses_driver_context():
    """Closed-shell Python periodic drivers can now pass the system/lattice
    context needed for the periodic SAP Fock-mode density."""
    from vibeqc.guess import initial_density_closed_shell
    vq, sysp, basis = _periodic_h2_sto3g()
    D = initial_density_closed_shell(
        sysp.unit_cell_molecule(),
        basis,
        1,
        InitialGuess.SAP,
        is_periodic=True,
        periodic_system=sysp,
        lattice_opts=vq.LatticeSumOptions(),
    )
    assert D is not None and D.shape == (basis.nbasis, basis.nbasis)
    assert np.max(np.abs(D - D.T)) < 1e-10


def test_periodic_guess_seam_allows_supported():
    """The periodic guess seam must NOT reject the supported guesses — SAD /
    HCORE go through to the engine (regression guard so the gate above can't
    over-fire)."""
    from vibeqc.guess import initial_density_closed_shell
    _vq, sysp, basis = _periodic_h2_sto3g()
    mol = sysp.unit_cell_molecule()
    # SAD returns a density; HCORE returns None (driver diagonalises Hcore).
    D = initial_density_closed_shell(mol, basis, 1, InitialGuess.SAD,
                                     is_periodic=True)
    assert D is not None and D.shape[0] == D.shape[1]
    none_ok = initial_density_closed_shell(mol, basis, 1, InitialGuess.HCORE,
                                           is_periodic=True)
    assert none_ok is None


# ---- ATOMSPIN: per-atom broken-symmetry spin seed -----------------------
#
# atomic_spins tags each atom +1 (majority alpha) / -1 (majority beta) / 0
# (unpolarised) to seed a broken-symmetry SAD start, the vibe-qc analogue of
# CRYSTAL's ATOMSPIN. The canonical demonstration is stretched H2 (the
# Coulson-Fischer RHF->UHF instability): the spin-symmetric default guess
# stays on the RHF solution, while seeding alpha on H1 / beta on H2 reaches
# the lower broken-symmetry minimum.


def _stretched_h2():
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])],
                   multiplicity=1)
    return mol, BasisSet(mol, "sto-3g")


def _stretched_h2_at(r):
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, r])],
                   multiplicity=1)
    return mol, BasisSet(mol, "sto-3g")


def _cr2_at(r):
    mol = Molecule([Atom(24, [0.0, 0.0, -r / 2]), Atom(24, [0.0, 0.0, r / 2])],
                   multiplicity=1)
    return mol, BasisSet(mol, "sto-3g")


def test_atomic_spins_seeds_broken_symmetry_stretched_h2():
    """H2/STO-3G at R=3.0 bohr: with the stability escape disabled, the
    symmetric SAD guess stays on the RHF-like saddle (E=-0.88527500,
    <S^2>=0 -- the textbook UHF internal instability); atomic_spins=[+1,-1]
    localises alpha on H1 / beta on H2 and converges to the broken-symmetry
    minimum E=-0.95101800 Ha, <S^2>=0.774 (PySCF UHF/STO-3G, atom-localised
    BS init; 0.0657 Ha below RHF).

    Since the UHF-ABOVE-ROHF-VARIATIONAL-INVERSION fix, the DEFAULT driver
    escapes that saddle without any seed (asserted at the end): the
    stability analysis detects the negative Hessian eigenvalue and the
    anchored restart lands the same BS minimum. The seeded arm remains the
    deterministic way to CHOOSE the broken-symmetry pattern (and is never
    rotated away, check-only)."""
    mol, basis = _stretched_h2()

    o_sym = _uhf_opts(InitialGuess.SAD)
    o_sym.stability_check = False  # pin the legacy saddle as the control
    r_sym = run_uhf(mol, basis, o_sym)
    assert r_sym.converged
    assert r_sym.energy == pytest.approx(-0.88527500, abs=1e-6)
    assert abs(r_sym.s_squared) < 1e-3   # symmetric guess: no broken symmetry

    opts = _uhf_opts(InitialGuess.SAD)
    opts.atomic_spins = [1, -1]
    r_bs = run_uhf(mol, basis, opts)
    assert r_bs.converged
    assert r_bs.energy == pytest.approx(-0.95101800, abs=1e-6)
    assert r_bs.s_squared == pytest.approx(0.7742, abs=1e-2)
    assert r_bs.energy < r_sym.energy - 1e-3   # BS is genuinely lower

    # Default (stability on, no seed): the saddle is escaped to the same
    # broken-symmetry minimum.
    r_def = run_uhf(mol, basis, _uhf_opts(InitialGuess.SAD))
    assert r_def.converged
    assert r_def.energy == pytest.approx(-0.95101800, abs=1e-6)
    assert r_def.n_stability_restarts >= 1


def test_atomic_spins_works_with_auto_guess():
    """AUTO resolves to SAD for open shells, so atomic_spins must seed the
    broken-symmetry start under AUTO too."""
    mol, basis = _stretched_h2()
    opts = _uhf_opts(InitialGuess.AUTO)
    opts.atomic_spins = [1, -1]
    r = run_uhf(mol, basis, opts)
    assert r.converged
    assert r.energy == pytest.approx(-0.95101800, abs=1e-6)
    assert r.s_squared == pytest.approx(0.7742, abs=1e-2)


def test_atomic_spins_global_sign_flip_is_equivalent():
    """The total Sz is fixed by the multiplicity, not atomic_spins, so a
    global sign flip [+1,-1] -> [-1,+1] is the same broken-symmetry state
    (alpha<->beta relabelled): identical energy and <S^2>."""
    mol, basis = _stretched_h2()
    o1 = _uhf_opts(InitialGuess.SAD); o1.atomic_spins = [1, -1]
    o2 = _uhf_opts(InitialGuess.SAD); o2.atomic_spins = [-1, 1]
    r1 = run_uhf(mol, basis, o1)
    r2 = run_uhf(mol, basis, o2)
    assert r1.energy == pytest.approx(r2.energy, abs=1e-8)
    assert r1.s_squared == pytest.approx(r2.s_squared, abs=1e-6)


def test_atomic_spins_requires_sad_guess():
    """atomic_spins is a SAD-density seed; pairing it with a non-SAD guess
    (HCORE) raises rather than silently ignoring the tags."""
    mol, basis = _stretched_h2()
    opts = _uhf_opts(InitialGuess.HCORE)
    opts.atomic_spins = [1, -1]
    with pytest.raises(RuntimeError, match="requires the SAD guess"):
        run_uhf(mol, basis, opts)


def test_atomic_spins_length_must_match_atoms():
    """A wrong-length atomic_spins list (not aligned to atom order) raises."""
    mol, basis = _stretched_h2()
    opts = _uhf_opts(InitialGuess.SAD)
    opts.atomic_spins = [1, -1, 1]   # 3 tags, 2 atoms
    with pytest.raises(ValueError, match="must match.*atom count"):
        run_uhf(mol, basis, opts)


def test_atomic_spins_forwarded_through_uks():
    """The UKS driver forwards atomic_spins to the same engine path, taking
    precedence over the UKS Fock-diagonalise split. On stretched H2
    (B3LYP/STO-3G, R=3.0 bohr), atomic_spins=[+1,-1] selects the lower
    broken-symmetry pattern directly. The default symmetric SAD start reaches
    that same internally stable determinant through the stability follow; it
    must not remain on the closed-shell saddle (issue #447)."""
    mol, basis = _stretched_h2()

    def _uks(spins=None):
        o = UKSOptions()
        o.functional = "b3lyp"
        o.conv_tol_energy = 1e-10
        o.conv_tol_grad = 1e-6
        o.max_iter = 200
        o.grid.n_radial = 40
        o.grid.n_theta = 14
        o.grid.n_phi = 28
        o.initial_guess = InitialGuess.SAD
        if spins is not None:
            o.atomic_spins = spins
        return o

    r_default = run_uks(mol, basis, _uks())
    r_bs = run_uks(mol, basis, _uks([1, -1]))
    assert r_default.converged and r_bs.converged
    assert r_default.n_stability_restarts >= 1
    assert r_bs.n_stability_restarts == 0
    assert not r_default.internal_instability
    assert r_default.stability_eigenvalue >= -_uks().stability_tol
    assert r_default.energy == pytest.approx(r_bs.energy, abs=1e-10)
    assert r_default.s_squared == pytest.approx(r_bs.s_squared, abs=1e-8)
    assert r_default.s_squared > 0.1


# ---- SPINLOCK: pattern-hold of the broken-symmetry seed via MOM ----------
#
# spinlock_iterations holds the seeded (atomic_spins) broken-symmetry
# occupation by maximum-overlap (MOM) selection for the first N SCF cycles,
# then releases to aufbau. On easy cases the seed already holds, so SPINLOCK
# must be a no-op on the converged result; its value is on hard systems where
# plain aufbau would let the seed collapse to the symmetric solution.


def test_spinlock_does_not_disturb_easy_case_h2():
    """SPINLOCK must not change the converged result on an easy case: H2/STO-3G
    R=3.0 bohr with atomic_spins=[+1,-1] reaches the same broken-symmetry
    minimum with or without spinlock_iterations (MOM selects the same occupied
    orbitals as aufbau when the seed already holds)."""
    mol, basis = _stretched_h2()
    o_nolock = _uhf_opts(InitialGuess.SAD)
    o_nolock.atomic_spins = [1, -1]
    o_lock = _uhf_opts(InitialGuess.SAD)
    o_lock.atomic_spins = [1, -1]
    o_lock.spinlock_mode = SpinlockMode.PATTERN_HOLD
    o_lock.spinlock_iterations = 8
    r_nolock = run_uhf(mol, basis, o_nolock)
    r_lock = run_uhf(mol, basis, o_lock)
    assert r_lock.converged
    assert r_lock.energy == pytest.approx(r_nolock.energy, abs=1e-8)
    assert r_lock.s_squared == pytest.approx(r_nolock.s_squared, abs=1e-6)


def test_spinlock_pattern_hold_molecular_uks():
    """PATTERN_HOLD on the molecular UKS driver protects an atomic_spins
    broken-symmetry seed on a weak-exchange functional. Deeply stretched
    H2 (R=5.0 bohr, PBE/STO-3G) with atomic_spins=[+1,-1] and a 10-cycle
    hold converges to the strongly-broken localized state the seed encodes
    (<S^2> near 1). The accelerator stays suspended through the hold window
    (asserted via scf_trace): MOM only selects WHICH orbitals are occupied;
    it cannot stop Fock extrapolation from rotating the occupied orbitals
    toward the spin-symmetric attractor, so extrapolating across held
    iterates would collapse the seed with max-overlap = aufbau throughout
    (no level crossing ever occurs). Mirrors the multi-k periodic pin
    test_periodic_spinlock_pattern_hold_multik_uks."""
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [5.0, 0.0, 0.0])],
                   multiplicity=1)
    basis = BasisSet(mol, "sto-3g")
    o = UKSOptions()
    o.functional = "pbe"
    o.conv_tol_energy = 1e-9
    o.conv_tol_grad = 1e-6
    o.max_iter = 200
    o.grid.n_radial = 40
    o.grid.n_theta = 14
    o.grid.n_phi = 28
    o.initial_guess = InitialGuess.SAD
    o.atomic_spins = [1, -1]
    o.spinlock_mode = SpinlockMode.PATTERN_HOLD
    o.spinlock_iterations = 10
    r = run_uks(mol, basis, o)
    assert r.converged
    assert r.s_squared > 0.5     # the broken-symmetry pattern was held
    # The accelerator stays suspended through the hold window (the fix that
    # makes the hold actually hold): no accelerator subspace before release.
    assert all(it.diis_subspace == 0 for it in r.scf_trace[:10])


def test_spinlock_spin_schedule_releases_to_target():
    """SPINLOCK SPIN_SCHEDULE (mode A) converges at a locked spin for N cycles,
    then releases to the multiplicity target. On H2/STO-3G at equilibrium
    (R=1.4 bohr, singlet), locking triplet (spinlock_value=2) for 5 cycles then
    releasing must still reach the singlet ground state, identical to a plain
    singlet UHF (proving the release, i.e. it is not stuck in the locked
    state)."""
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
                   multiplicity=1)
    basis = BasisSet(mol, "sto-3g")
    r_plain = run_uhf(mol, basis, _uhf_opts(InitialGuess.SAD))
    o = _uhf_opts(InitialGuess.SAD)
    o.spinlock_mode = SpinlockMode.SPIN_SCHEDULE
    o.spinlock_value = 2          # lock triplet (n_alpha - n_beta = 2) early
    o.spinlock_iterations = 5
    r_sched = run_uhf(mol, basis, o)
    assert r_sched.converged
    assert r_sched.energy == pytest.approx(r_plain.energy, abs=1e-7)


def test_spinlock_value_parity_guard():
    """A spinlock_value with the wrong parity for the electron count raises
    (2 electrons cannot have n_alpha - n_beta = 1)."""
    mol, basis = _stretched_h2()
    o = _uhf_opts(InitialGuess.SAD)
    o.spinlock_mode = SpinlockMode.SPIN_SCHEDULE
    o.spinlock_value = 1
    o.spinlock_iterations = 5
    with pytest.raises(RuntimeError, match="incompatible"):
        run_uhf(mol, basis, o)


# ---- READ: restart an SCF from a prior result / file --------------------
#
# READ rebuilds the starting density from a prior calculation instead of an
# atomic guess. Three sources — an in-memory result, a .qvf, and a .molden —
# all converge to the same energy as from-scratch (it's a guess, not a new
# method) in far fewer iterations. When the prior geometry/basis differ from
# the current ones the density is projected onto the current basis (the
# NEB / geometry-scan restart path).


def _h2o():
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    return mol, BasisSet(mol, "6-31g*")


def test_read_inmemory_restart_rhf():
    mol, basis = _h2o()
    base = run_rhf(mol, basis, _rhf_opts(InitialGuess.AUTO))
    r = run_rhf(mol, basis, _rhf_opts(InitialGuess.READ), read_from=base)
    assert r.converged
    assert abs(r.energy - base.energy) < 1e-9
    assert r.n_iter < base.n_iter          # a real restart, not from scratch


def test_read_inmemory_restart_uhf():
    A2B = 1.0 / 0.529177210903
    atoms = [(8, [0.0, 0.0, 0.0]), (1, [0.95 * A2B, 0.0, 0.0])]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms], multiplicity=2)
    basis = BasisSet(mol, "6-31g*")
    base = run_uhf(mol, basis, _uhf_opts(InitialGuess.SAD))
    r = run_uhf(mol, basis, _uhf_opts(InitialGuess.READ), read_from=base)
    assert r.converged
    assert abs(r.energy - base.energy) < 1e-9
    assert r.n_iter < base.n_iter
    assert r.s_squared == pytest.approx(base.s_squared, abs=1e-3)


def test_read_inmemory_restart_rks():
    from vibeqc import RKSOptions, run_rks
    mol, basis = _h2o()
    o = RKSOptions(); o.functional = "pbe"; o.conv_tol_energy = 1e-11
    base = run_rks(mol, basis, o)
    o2 = RKSOptions(); o2.functional = "pbe"; o2.conv_tol_energy = 1e-11
    o2.initial_guess = InitialGuess.READ
    r = run_rks(mol, basis, o2, read_from=base)
    assert r.converged
    assert abs(r.energy - base.energy) < 1e-9
    assert r.n_iter < base.n_iter


def test_read_molden_roundtrip_rhf(tmp_path):
    from vibeqc.output.formats.molden import write_molden
    mol, basis = _h2o()
    base = run_rhf(mol, basis, _rhf_opts(InitialGuess.AUTO))
    path = str(tmp_path / "h2o.molden")
    write_molden(path, mol, basis, base)
    o = _rhf_opts(InitialGuess.READ); o.read_path = path
    r = run_rhf(mol, basis, o)
    assert r.converged
    assert abs(r.energy - base.energy) < 1e-9
    assert r.n_iter < base.n_iter


@pytest.mark.parametrize("suffix", [".molden", ".molden.input"])
def test_read_molden_infers_pre_normalized_contractions(
    tmp_path, monkeypatch, suffix
):
    """READ accepts Molden exporters that write normalized contractions.

    ORCA's ``orca_2mkl`` writes the primitive-normalized contraction values
    carried by its AO basis, whereas vibe-qc's writer follows the common raw
    Gaussian-coefficient convention. The MO orthonormality condition must
    select the correct interpretation instead of silently changing the
    electron count during projection (issue 12).
    """
    from vibeqc.guess_read import resolve_read_density_closed
    from vibeqc.output.formats import molden as molden_format

    mol, basis = _h2o()
    base = run_rhf(mol, basis, _rhf_opts(InitialGuess.AUTO))
    path = str(tmp_path / f"pre_normalized{suffix}")

    # Synthetic ORCA-style fixture made entirely from vibe-qc data: suppress
    # the writer's normal de-normalization so [GTO] contains the same
    # primitive-normalized coefficients stored by BasisSet.shells().
    monkeypatch.setattr(
        molden_format,
        "_primitive_normalisation",
        lambda exponent, angular_momentum: 1.0,
    )
    molden_format.write_molden(path, mol, basis, base)

    options = _rhf_opts(InitialGuess.READ)
    options.read_path = path
    density = resolve_read_density_closed(options, mol, basis, None)
    overlap = np.asarray(_core.compute_overlap(basis))
    electron_count = np.einsum("ij,ji->", density, overlap)
    assert electron_count == pytest.approx(mol.n_electrons(), abs=1e-8)
    np.testing.assert_allclose(
        density,
        np.asarray(base.density),
        rtol=0.0,
        atol=5e-8,
    )

    result = run_rhf(mol, basis, options)
    assert result.converged
    assert result.energy == pytest.approx(base.energy, abs=1e-9)
    assert result.n_iter < base.n_iter


def test_read_molden_infers_pre_normalized_uhf_spin_blocks(
    tmp_path, monkeypatch
):
    """Normalization inference preserves both spin densities independently."""
    from vibeqc.guess_read import resolve_read_densities_open
    from vibeqc.output.formats import molden as molden_format

    atoms = [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0]),
    ]
    mol = Molecule(atoms, multiplicity=2)
    basis = BasisSet(mol, "6-31g*")
    base = run_uhf(mol, basis, _uhf_opts(InitialGuess.SAD))
    path = str(tmp_path / "pre_normalized_uhf.molden.input")

    monkeypatch.setattr(
        molden_format,
        "_primitive_normalisation",
        lambda exponent, angular_momentum: 1.0,
    )
    molden_format.write_molden(path, mol, basis, base)

    options = _uhf_opts(InitialGuess.READ)
    options.read_path = path
    density_alpha, density_beta = resolve_read_densities_open(
        options, mol, basis, None
    )
    overlap = np.asarray(_core.compute_overlap(basis))
    assert np.einsum("ij,ji->", density_alpha, overlap) == pytest.approx(
        5.0, abs=1e-8
    )
    assert np.einsum("ij,ji->", density_beta, overlap) == pytest.approx(
        4.0, abs=1e-8
    )
    np.testing.assert_allclose(
        density_alpha, np.asarray(base.density_alpha), rtol=0.0, atol=5e-8
    )
    np.testing.assert_allclose(
        density_beta, np.asarray(base.density_beta), rtol=0.0, atol=5e-8
    )


def test_read_molden_honors_explicit_ao_indices(tmp_path):
    """Indexed MO rows may be reordered without changing the density."""
    from vibeqc.guess_read import resolve_read_density_closed
    from vibeqc.output.formats.molden import write_molden

    mol, basis = _h2o()
    base = run_rhf(mol, basis, _rhf_opts(InitialGuess.AUTO))
    path = tmp_path / "reordered.molden"
    write_molden(path, mol, basis, base)

    lines = path.read_text(encoding="utf-8").splitlines()
    coefficient_rows = []
    in_first_mo = False
    for index, line in enumerate(lines):
        low = line.strip().lower()
        if low == "[mo]":
            continue
        if low.startswith("sym="):
            if in_first_mo:
                break
            in_first_mo = True
            continue
        parts = line.split()
        if in_first_mo and len(parts) == 2 and parts[0].isdigit():
            coefficient_rows.append(index)
    reversed_rows = [lines[index] for index in reversed(coefficient_rows)]
    for index, row in zip(coefficient_rows, reversed_rows):
        lines[index] = row
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    options = _rhf_opts(InitialGuess.READ)
    options.read_path = str(path)
    density = resolve_read_density_closed(options, mol, basis, None)
    np.testing.assert_allclose(
        density,
        np.asarray(base.density),
        rtol=0.0,
        atol=5e-8,
    )


def test_read_molden_rejects_nonorthonormal_orbitals(tmp_path):
    """Neither contraction convention may silently accept damaged MOs."""
    from vibeqc.guess_read import resolve_read_density_closed
    from vibeqc.output.formats.molden import write_molden

    mol, basis = _h2o()
    base = run_rhf(mol, basis, _rhf_opts(InitialGuess.AUTO))
    path = tmp_path / "damaged.molden"
    write_molden(path, mol, basis, base)

    lines = path.read_text(encoding="utf-8").splitlines()
    in_mo_block = False
    for index, line in enumerate(lines):
        if line.strip().lower() == "[mo]":
            in_mo_block = True
            continue
        parts = line.split()
        if in_mo_block and len(parts) == 2 and parts[0].isdigit():
            lines[index] = f" {parts[0]} {2.0 * float(parts[1]):.12e}"
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    options = _rhf_opts(InitialGuess.READ)
    options.read_path = str(path)
    with pytest.raises(ValueError, match="not orthonormal"):
        resolve_read_density_closed(options, mol, basis, None)


def test_read_molden_rejects_incomplete_occupied_orbital(tmp_path):
    """An omitted occupied-MO row must not silently remove two electrons."""
    from vibeqc.guess_read import resolve_read_density_closed
    from vibeqc.output.formats.molden import write_molden

    mol, basis = _h2o()
    base = run_rhf(mol, basis, _rhf_opts(InitialGuess.AUTO))
    path = tmp_path / "truncated.molden"
    write_molden(path, mol, basis, base)

    lines = path.read_text(encoding="utf-8").splitlines()
    in_first_mo = False
    for index, line in enumerate(lines):
        if line.strip().lower() == "[mo]":
            in_first_mo = True
            continue
        parts = line.split()
        if in_first_mo and len(parts) == 2 and parts[0].isdigit():
            del lines[index]
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    options = _rhf_opts(InitialGuess.READ)
    options.read_path = str(path)
    with pytest.raises(ValueError, match="occupied.*AO coefficients"):
        resolve_read_density_closed(options, mol, basis, None)


def test_read_molden_rejects_missing_occupation(tmp_path):
    """A missing Occup field must not silently turn an occupied MO virtual."""
    from vibeqc.guess_read import resolve_read_density_closed
    from vibeqc.output.formats.molden import write_molden

    mol, basis = _h2o()
    base = run_rhf(mol, basis, _rhf_opts(InitialGuess.AUTO))
    path = tmp_path / "missing_occupation.molden"
    write_molden(path, mol, basis, base)

    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip().lower().startswith("occup="):
            del lines[index]
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    options = _rhf_opts(InitialGuess.READ)
    options.read_path = str(path)
    with pytest.raises(ValueError, match="missing its Occup field"):
        resolve_read_density_closed(options, mol, basis, None)


def test_read_molden_rejects_nonfinite_virtual_coefficient(tmp_path):
    """A NaN virtual coefficient must not poison a zero-occupied density."""
    from vibeqc.guess_read import resolve_read_density_closed
    from vibeqc.output.formats.molden import write_molden

    mol, basis = _h2o()
    base = run_rhf(mol, basis, _rhf_opts(InitialGuess.AUTO))
    path = tmp_path / "nonfinite_virtual.molden"
    write_molden(path, mol, basis, base)

    lines = path.read_text(encoding="utf-8").splitlines()
    virtual = False
    for index, line in enumerate(lines):
        low = line.strip().lower()
        if low.startswith("occup="):
            virtual = float(line.split("=", 1)[1]) == 0.0
            continue
        parts = line.split()
        if virtual and len(parts) == 2 and parts[0].isdigit():
            lines[index] = f" {parts[0]} nan"
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    options = _rhf_opts(InitialGuess.READ)
    options.read_path = str(path)
    with pytest.raises(ValueError, match="non-finite MO coefficient"):
        resolve_read_density_closed(options, mol, basis, None)


def _write_qvf(stem, mol, basis, result, method):
    from vibeqc.output.plan import OutputPlan
    from vibeqc.output.formats.qvf import write_qvf, qvf_wf_data
    plan = OutputPlan.from_run_job_kwargs(
        output=stem, method=method, basis="6-31g*", functional=None)
    write_qvf(stem, plan, molecule=mol, result=result, method=method,
              basis="6-31g*", wf_data=qvf_wf_data(result, basis, mol))
    return stem + ".qvf"


def test_read_qvf_roundtrip_rhf(tmp_path):
    mol, basis = _h2o()
    base = run_rhf(mol, basis, _rhf_opts(InitialGuess.AUTO))
    qvf = _write_qvf(str(tmp_path / "h2o"), mol, basis, base, "rhf")
    o = _rhf_opts(InitialGuess.READ); o.read_path = qvf
    r = run_rhf(mol, basis, o)
    assert r.converged
    assert abs(r.energy - base.energy) < 1e-9
    assert r.n_iter < base.n_iter


def test_read_qvf_roundtrip_uhf(tmp_path):
    A2B = 1.0 / 0.529177210903
    atoms = [(8, [0.0, 0.0, 0.0]), (1, [0.95 * A2B, 0.0, 0.0])]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms], multiplicity=2)
    basis = BasisSet(mol, "6-31g*")
    base = run_uhf(mol, basis, _uhf_opts(InitialGuess.SAD))
    qvf = _write_qvf(str(tmp_path / "oh"), mol, basis, base, "uhf")
    o = _uhf_opts(InitialGuess.READ); o.read_path = qvf
    r = run_uhf(mol, basis, o)
    assert r.converged
    assert abs(r.energy - base.energy) < 1e-9
    assert r.n_iter < base.n_iter


def test_read_projects_across_a_shifted_geometry(tmp_path):
    """The NEB / geometry-scan path: a prior .qvf at one geometry restarts an
    SCF at a *different* geometry by projecting the prior density onto the new
    basis (D = P·D_prior·Pᵀ). The defining correctness property is that the
    projected restart still reaches the from-scratch minimum of the *new*
    geometry — proving the cross-basis projection is right. (The iteration
    speed-up of a restart is asserted by the same-geometry round-trip tests;
    for a projected restart it depends on step size and tolerance.)"""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "6-31g*")
    base = run_rhf(mol, basis, _rhf_opts(InitialGuess.AUTO))
    qvf = _write_qvf(str(tmp_path / "h2o"), mol, basis, base, "rhf")

    shifted = Molecule([Atom(atoms[0][0], [0.0, 0.0, 0.10])]
                       + [Atom(Z, list(xyz)) for Z, xyz in atoms[1:]])
    sbasis = BasisSet(shifted, "6-31g*")
    scratch = run_rhf(shifted, sbasis, _rhf_opts(InitialGuess.AUTO))
    o = _rhf_opts(InitialGuess.READ); o.read_path = qvf
    r = run_rhf(shifted, sbasis, o)
    assert r.converged
    assert abs(r.energy - scratch.energy) < 1e-8   # same minimum, projected


def test_read_requires_a_source():
    """READ with neither read_from nor read_path is a clear error."""
    mol, basis = _h2o()
    with pytest.raises((ValueError, RuntimeError), match="(?i)READ"):
        run_rhf(mol, basis, _rhf_opts(InitialGuess.READ))


# ==========================================================================
# PERIODIC case for the initial-guess / convergence features.
#
# READ / ATOMSPIN / SPINLOCK landed molecular-first; these mirror the
# molecular tests above for the periodic (Γ-point and multi-k) drivers.
# The periodic SCF runs in the Python drivers, so the broken-symmetry seed
# rides on the g=0 cell density (which Bloch-sums to a broken-symmetry D(k)).
# A large cubic box at Γ reduces to the molecular limit, so a periodic-Γ
# broken-symmetry result must reproduce the molecular UHF reference above —
# the cleanest correctness anchor that needs no external reference.
# ==========================================================================


def _periodic_stretched_h2(box_bohr: float = 30.0, r_bohr: float = 3.0):
    """Stretched H2 (R=3.0 bohr) in a large cubic box: the periodic analog of
    ``_stretched_h2`` (molecular-limit regime at Γ)."""
    import vibeqc as vq

    lat = box_bohr * np.eye(3)
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [r_bohr, 0.0, 0.0])]
    sysp = vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=1)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def test_periodic_atomic_spins_breaks_symmetry_gamma_uhf_ewald():
    """ATOMSPIN on the Γ-only UHF Ewald driver. The symmetric SAD start
    reproduces the molecular-limit RHF energy (-0.885275 Ha, <S^2>=0); the
    per-atom seed [+1,-1] localises α on H1 / β on H2 and converges to the
    same broken-symmetry minimum the molecular test reaches
    (E=-0.951018 Ha, <S^2>=0.774). The Γ result matching the molecular UHF
    reference to ~1e-7 is the correctness anchor (no external reference)."""
    import vibeqc as vq

    sysp, basis = _periodic_stretched_h2()

    def _run(spins):
        o = vq.PeriodicRHFOptions()
        o.initial_guess = vq.InitialGuess.SAD
        o.conv_tol_energy = 1e-10
        o.max_iter = 300
        if spins is not None:
            o.atomic_spins = spins
        return vq.run_uhf_periodic_gamma_ewald3d(sysp, basis, o, verbose=0)

    r_sym = _run(None)
    assert r_sym.converged
    assert r_sym.energy == pytest.approx(-0.88527500, abs=1e-4)
    assert abs(r_sym.s_squared) < 1e-3

    r_bs = _run([1, -1])
    assert r_bs.converged
    assert r_bs.energy == pytest.approx(-0.95101800, abs=1e-4)
    assert r_bs.s_squared == pytest.approx(0.7742, abs=2e-2)
    assert r_bs.energy < r_sym.energy - 1e-2   # broken-symmetry is lower


def test_periodic_atomic_spins_global_sign_flip_equivalent_gamma_uhf():
    """As molecular: a global sign flip [+1,-1] -> [-1,+1] is the same
    broken-symmetry state (α<->β relabelled) — identical energy and <S^2>."""
    import vibeqc as vq

    sysp, basis = _periodic_stretched_h2()

    def _run(spins):
        o = vq.PeriodicRHFOptions()
        o.initial_guess = vq.InitialGuess.SAD
        o.conv_tol_energy = 1e-10
        o.max_iter = 300
        o.atomic_spins = spins
        return vq.run_uhf_periodic_gamma_ewald3d(sysp, basis, o, verbose=0)

    r1 = _run([1, -1])
    r2 = _run([-1, 1])
    assert r1.energy == pytest.approx(r2.energy, abs=1e-7)
    assert r1.s_squared == pytest.approx(r2.s_squared, abs=1e-5)


def test_periodic_atomic_spins_breaks_symmetry_gdf_uhf():
    """ATOMSPIN on the GDF UHF driver (the PySCF-parity-natural backend). GDF
    open-shell otherwise starts spin-degenerate (C_β = C_α from Hcore), so the
    guarded engine hook is what enables a broken-symmetry start. The exxdiv
    Madelung term shifts the absolute energy vs the Ewald driver, so the test
    asserts the qualitative break, not the molecular-limit number."""
    import vibeqc as vq

    sysp, basis = _periodic_stretched_h2()

    def _run(spins):
        o = vq.PeriodicRHFOptions()
        o.initial_guess = vq.InitialGuess.SAD
        o.conv_tol_energy = 1e-10
        o.max_iter = 300
        if spins is not None:
            o.atomic_spins = spins
        return vq.run_pbc_gdf_uhf(sysp, basis, o, verbose=0)

    r_sym = _run(None)
    r_bs = _run([1, -1])
    assert r_sym.converged and r_bs.converged
    assert abs(r_sym.s_squared) < 1e-3            # spin-degenerate start
    assert r_bs.s_squared > 0.5                   # seed broke symmetry
    assert r_bs.energy < r_sym.energy - 1e-2      # broken-symmetry is lower


def test_periodic_atomic_spins_multik_uhf_ewald_breaks_symmetry():
    """ATOMSPIN on the multi-k UHF Ewald driver reaches a broken-symmetry
    basin instead of collapsing to the symmetric SAD solution."""
    import vibeqc as vq
    from vibeqc import run_uhf_periodic_multi_k_ewald3d

    sysp, basis = _periodic_stretched_h2()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])

    def _run(spins=None):
        o = vq.PeriodicRHFOptions()
        o.initial_guess = vq.InitialGuess.SAD
        o.max_iter = 40
        o.conv_tol_energy = 1e-7
        o.conv_tol_grad = 1e-5
        if spins is not None:
            o.atomic_spins = spins
        return run_uhf_periodic_multi_k_ewald3d(sysp, basis, kmesh, o, verbose=0)

    r_sym = _run(None)
    r_bs = _run([1, -1])
    assert r_sym.converged and r_bs.converged
    assert abs(r_sym.s_squared) < 1e-3
    assert r_bs.s_squared > 0.5
    assert r_bs.energy < r_sym.energy - 1e-2


def test_periodic_atomic_spins_rejected_for_closed_shell(tmp_path):
    """atomic_spins is an open-shell broken-symmetry seed; run_periodic_job
    rejects it for a closed-shell method rather than silently ignoring it."""
    import vibeqc as vq

    sysp, basis = _periodic_stretched_h2()
    with pytest.raises(ValueError, match="(?i)atomic_spins.*UHF.*UKS"):
        vq.run_periodic_job(
            sysp, basis, method="RHF", atomic_spins=[1, -1],
            dry_run=True, output=str(tmp_path / "atomspin_rhf"),
        )


# ---- READ: periodic restart ---------------------------------------------
#
# A Gamma periodic READ restart injects the prior calculation's g=0 cell
# density (projected onto the current cell basis) as the SCF start; it
# Bloch-sums to a k-independent D(k). Closed-shell multi-k GDF/GPW/GAPW
# restarts accept in-memory native results and rebuild each per-k D(k) from
# complex Bloch coefficients and occupations. File-backed multi-k READ remains
# fail-closed because QVF wavefunction.gto carries only one selected k-point.


def _periodic_h2(box_bohr: float = 20.0, r_bohr: float = 1.4):
    """Closed-shell H2 (equilibrium) in a box — a quick periodic Γ system."""
    import vibeqc as vq

    lat = box_bohr * np.eye(3)
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [r_bohr, 0.0, 0.0])]
    sysp = vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=1)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def test_periodic_read_inmemory_restart_rhf_gdf():
    """In-memory Γ RHF restart: a prior result's g=0 density restarts the SCF
    to the same energy in fewer iterations (the defining restart property)."""
    import vibeqc as vq
    from vibeqc.guess_read import resolve_periodic_read_density_closed

    sysp, basis = _periodic_h2()

    o1 = vq.PeriodicRHFOptions()
    o1.initial_guess = vq.InitialGuess.SAD
    o1.conv_tol_energy = 1e-9
    o1.max_iter = 100
    base = vq.run_pbc_gdf_rhf(sysp, basis, o1, verbose=0)
    assert base.converged

    o2 = vq.PeriodicRHFOptions()
    o2.initial_guess = vq.InitialGuess.READ
    o2.conv_tol_energy = 1e-9
    o2.max_iter = 100
    o2.read_density = resolve_periodic_read_density_closed(basis, read_from=base)
    r = vq.run_pbc_gdf_rhf(sysp, basis, o2, verbose=0)
    assert r.converged
    assert abs(r.energy - base.energy) < 1e-9
    assert r.n_iter < base.n_iter


def test_periodic_read_inmemory_restart_uhf_preserves_magnetism():
    """An open-shell Γ UHF restart preserves the prior's broken-symmetry state:
    restart a converged ATOMSPIN solution and reach the same energy + ⟨S²⟩ in
    fewer iterations (the per-spin g=0 densities carry the magnetisation)."""
    import vibeqc as vq
    from vibeqc.guess_read import resolve_periodic_read_densities_open

    sysp, basis = _periodic_stretched_h2()

    o1 = vq.PeriodicRHFOptions()
    o1.initial_guess = vq.InitialGuess.SAD
    o1.conv_tol_energy = 1e-9
    o1.max_iter = 300
    o1.atomic_spins = [1, -1]
    base = vq.run_pbc_gdf_uhf(sysp, basis, o1, verbose=0)
    assert base.converged and base.s_squared > 0.5

    o2 = vq.PeriodicRHFOptions()
    o2.initial_guess = vq.InitialGuess.READ
    o2.conv_tol_energy = 1e-9
    o2.max_iter = 300
    da, db = resolve_periodic_read_densities_open(basis, read_from=base)
    o2.read_density_alpha = da
    o2.read_density_beta = db
    r = vq.run_pbc_gdf_uhf(sysp, basis, o2, verbose=0)
    assert r.converged
    assert abs(r.energy - base.energy) < 1e-9
    assert r.s_squared == pytest.approx(base.s_squared, abs=1e-4)
    assert r.n_iter < base.n_iter


def test_periodic_read_qvf_roundtrip_rhf(tmp_path):
    """The NEB / geometry-scan restart path end-to-end: run_periodic_job writes
    a Γ .qvf (wavefunction.gto = real Γ MO coefficients), and a second job with
    initial_guess='read' restarts from that file to the same energy in fewer
    iterations."""
    import glob

    import vibeqc as vq

    sysp, basis = _periodic_h2()
    stem = str(tmp_path / "scratch")
    base = vq.run_periodic_job(
        sysp, basis, method="RHF", jk_method="gdf", max_iter=100,
        conv_tol_energy=1e-9, output=stem, output_qvf=True,
        citations=False, verbose=0,
    )
    qvf = glob.glob(stem + "*.qvf")
    assert qvf, "no .qvf written"

    r = vq.run_periodic_job(
        sysp, basis, method="RHF", jk_method="gdf", initial_guess="read",
        read_from=qvf[0], max_iter=100, conv_tol_energy=1e-9,
        output=str(tmp_path / "restart"), citations=False, verbose=0,
    )
    assert r.converged
    assert abs(r.energy - base.energy) < 1e-9
    assert r.n_iter < base.n_iter


def test_periodic_open_shell_qvf_writes_spin_resolved_dos(tmp_path):
    """Open-shell periodic QVF must write spin-resolved DOS/PDOS arrays.

    The QVF writer requires ``dos.total`` as ``[2, n_points]`` and
    ``dos.projected`` as ``[2, n_channels, n_points]`` when ``n_spin=2``.
    """
    import json
    import zipfile

    import vibeqc as vq

    lat = 20.0 * np.eye(3)
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [3.0, 0.0, 0.0])]
    sysp = vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=3)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "triplet_h2"

    result = vq.run_periodic_job(
        sysp,
        basis,
        method="UHF",
        jk_method="gdf",
        output=stem,
        output_qvf=True,
        dos_kmesh=(1, 1, 1),
        max_iter=80,
        conv_tol_energy=1e-9,
        citations=False,
        verbose=0,
    )
    assert result.converged
    qvf = stem.with_suffix(".qvf")
    assert qvf.is_file()

    with zipfile.ZipFile(qvf, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        sections = {section["kind"]: section for section in manifest["sections"]}
        assert sections["dos.total"]["n_spin"] == 2
        assert sections["dos.projected"]["n_spin"] == 2
        energies = np.frombuffer(zf.read("dos/energies.bin"), dtype=np.float64)
        dos = np.frombuffer(zf.read("dos/total.bin"), dtype=np.float64)
        pdos = np.frombuffer(zf.read("dos/projections.bin"), dtype=np.float64)

    assert dos.size == 2 * energies.size
    assert pdos.size == (
        2 * len(sections["dos.projected"]["channels"]) * energies.size
    )


def test_periodic_read_multik_legacy_qvf_source_raises(tmp_path):
    """An archive without source basis/cell data cannot define a restart."""
    import json
    import zipfile

    import vibeqc as vq

    sysp, basis = _periodic_h2()
    qvf = tmp_path / "prev.qvf"
    with zipfile.ZipFile(qvf, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({"qvf_version": 1, "sections": []}),
        )
    with pytest.raises(
        ValueError,
        match="source structure and basis",
    ):
        vq.run_periodic_job(
            sysp, basis, method="RHF", jk_method="gdf", initial_guess="read",
            read_from=qvf, kpoints=[2, 1, 1], output=str(tmp_path / "mk"),
            citations=False, verbose=0,
        )


def test_periodic_read_multik_non_qvf_file_source_raises(tmp_path):
    """Molden cannot carry all k blocks for a multi-k periodic restart."""
    import vibeqc as vq

    sysp, basis = _periodic_h2()
    molden = tmp_path / "prev.molden"
    molden.write_text("[Molden Format]\n", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="(?i)non-QVF|per-k"):
        vq.run_periodic_job(
            sysp, basis, method="RHF", jk_method="gdf", initial_guess="read",
            read_from=molden, kpoints=[2, 1, 1], output=str(tmp_path / "mk_molden"),
            citations=False, verbose=0,
        )


def test_periodic_read_requires_a_source():
    """READ with no source (no read_density, no read_path) is a clear error."""
    import vibeqc as vq

    sysp, basis = _periodic_h2()
    o = vq.PeriodicRHFOptions()
    o.initial_guess = vq.InitialGuess.READ
    with pytest.raises((ValueError, RuntimeError), match="(?i)READ"):
        vq.run_pbc_gdf_rhf(sysp, basis, o, verbose=0)


# ---- SPINLOCK: periodic open-shell magnetic convergence ------------------
#
# PATTERN_HOLD holds the seeded broken-symmetry occupied subspace by maximum
# overlap (MOM) for the first ``spinlock_iterations`` cycles, then releases —
# protecting an ATOMSPIN seed from collapsing to the symmetric solution. On an
# easy case (where the seed already holds) it must be a no-op, mirroring the
# molecular SPINLOCK tests. SPIN_SCHEDULE locks n_alpha-n_beta = spinlock_value
# for the first spinlock_iterations cycles, then releases to the multiplicity
# target (a two-phase SCF). Support matrix: PATTERN_HOLD on the Γ UHF Ewald +
# multi-k UKS Ewald drivers; SPIN_SCHEDULE on the Γ UHF/UKS Ewald drivers;
# everything else fails closed.


def test_periodic_spinlock_pattern_hold_no_op_gamma_uhf():
    """PATTERN_HOLD must not change the converged result on an easy case:
    stretched H2 at Γ with atomic_spins=[+1,-1] reaches the same broken-
    symmetry minimum (S²=0.77) with or without the MOM hold — MOM selects the
    same occupied orbitals as aufbau when the seed already holds."""
    import vibeqc as vq
    from vibeqc._vibeqc_core import SpinlockMode

    sysp, basis = _periodic_stretched_h2()

    def _run(hold):
        o = vq.PeriodicRHFOptions()
        o.initial_guess = vq.InitialGuess.SAD
        o.conv_tol_energy = 1e-10
        o.max_iter = 300
        o.atomic_spins = [1, -1]
        if hold:
            o.spinlock_mode = SpinlockMode.PATTERN_HOLD
            o.spinlock_iterations = 8
        return vq.run_uhf_periodic_gamma_ewald3d(sysp, basis, o, verbose=0)

    r0 = _run(False)
    r1 = _run(True)
    assert r1.converged
    assert r1.energy == pytest.approx(r0.energy, abs=1e-8)
    assert r1.s_squared == pytest.approx(r0.s_squared, abs=1e-6)
    assert r1.s_squared > 0.5  # still the broken-symmetry solution


def test_periodic_spinlock_spin_schedule_releases_to_target_gamma_uhf():
    """SPIN_SCHEDULE (mode A) converges a locked spin for N cycles, then
    releases to the multiplicity target. On H2/STO-3G at equilibrium (Γ,
    singlet), locking triplet (spinlock_value=2) for 5 cycles then releasing
    must still reach the plain singlet ground state — proving the release."""
    import vibeqc as vq
    from vibeqc._vibeqc_core import SpinlockMode

    sysp, basis = _periodic_h2()

    o0 = vq.PeriodicRHFOptions()
    o0.initial_guess = vq.InitialGuess.SAD
    o0.conv_tol_energy = 1e-10
    o0.max_iter = 200
    r_plain = vq.run_uhf_periodic_gamma_ewald3d(sysp, basis, o0, verbose=0)

    o = vq.PeriodicRHFOptions()
    o.initial_guess = vq.InitialGuess.SAD
    o.conv_tol_energy = 1e-10
    o.max_iter = 200
    o.spinlock_mode = SpinlockMode.SPIN_SCHEDULE
    o.spinlock_value = 2          # lock triplet early
    o.spinlock_iterations = 5
    r_sched = vq.run_uhf_periodic_gamma_ewald3d(sysp, basis, o, verbose=0)
    assert r_sched.converged
    assert r_sched.energy == pytest.approx(r_plain.energy, abs=1e-7)
    # The caller's options object is restored (two-phase mutates in place).
    assert o.spinlock_mode == SpinlockMode.SPIN_SCHEDULE
    assert o.initial_guess == vq.InitialGuess.SAD


def test_periodic_spinlock_spin_schedule_parity_guard():
    """A spinlock_value with the wrong parity for the electron count raises
    (2 electrons cannot have n_alpha - n_beta = 1)."""
    import vibeqc as vq
    from vibeqc._vibeqc_core import SpinlockMode

    sysp, basis = _periodic_h2()
    o = vq.PeriodicRHFOptions()
    o.spinlock_mode = SpinlockMode.SPIN_SCHEDULE
    o.spinlock_value = 1
    o.spinlock_iterations = 5
    with pytest.raises(ValueError, match="(?i)incompatible|parity"):
        vq.run_uhf_periodic_gamma_ewald3d(sysp, basis, o, verbose=0)


def test_periodic_spinlock_unsupported_driver_fails_closed():
    """SPINLOCK on a driver that does not implement it fails closed with a
    clear error rather than silently ignoring the request. The multi-k UHF
    Ewald driver supports an ATOMSPIN start but not the MOM / two-phase
    SPINLOCK machinery, so every SPINLOCK mode raises there. The Γ-Ewald,
    GDF and BIPOLE drivers now support both modes, so they are no longer the
    fail-closed example."""
    import vibeqc as vq
    from vibeqc._vibeqc_core import SpinlockMode

    sysp, basis = _periodic_h2()
    o = vq.PeriodicRHFOptions()
    o.spinlock_mode = SpinlockMode.SPIN_SCHEDULE
    o.spinlock_value = 0
    o.spinlock_iterations = 5
    with pytest.raises(NotImplementedError, match="(?i)SPINLOCK.*not implemented"):
        vq.run_uhf_periodic_multi_k_ewald3d(sysp, basis, (1, 1, 1), o, verbose=0)


@pytest.mark.slow
def test_periodic_spinlock_pattern_hold_multik_uks():
    """PATTERN_HOLD on the multi-k UKS path, the end-to-end pin for
    protecting an AFM/broken-symmetry seed on multi-k. A uniformly stretched
    H chain (both neighbour distances 5 bohr: atoms at 0 and 5 in a 10-bohr
    box) at kmesh (2,1,1) with atomic_spins=[+1,-1]: the per-k per-spin MOM
    hold runs end-to-end, DIIS stays suspended through the hold window, and
    the SCF converges to the strongly-broken localized state the seed encodes
    (<S^2> = 0.946). Two hard-won constraints on this pin: (1) DIIS must not
    extrapolate across held iterates; that used to collapse the seed from
    spin populations +-0.99 to +-0.28 within 3 cycles by continuous orbital
    rotation, which the occupation-selecting MOM hold cannot see (no level
    crossing ever occurs). The scf_trace assertion below pins the suspension.
    (2) The box must keep BOTH neighbour distances stretched: the original
    8-bohr box put the periodic image at 8-5 = 3 bohr, inside PBE's
    Coulson-Fischer point, where no strongly-polarized stationary state
    exists at all; hold or no hold, the SCF can only reach the weak-BS
    <S^2> ~ 0.08 or the symmetric solution there. Marked slow: a
    dissociated-H-chain multi-k UKS SCF."""
    import vibeqc as vq
    from vibeqc import run_uks_periodic_multi_k_ewald3d
    from vibeqc._vibeqc_core import SpinlockMode

    lat = np.diag([10.0, 10.0, 10.0])
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [5.0, 0.0, 0.0])]
    sysp = vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=1)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])

    o = vq.PeriodicKSOptions()
    o.functional = "pbe"
    o.initial_guess = vq.InitialGuess.SAD
    o.conv_tol_energy = 1e-7
    o.conv_tol_grad = 1e-5
    o.max_iter = 300
    o.atomic_spins = [1, -1]
    o.spinlock_mode = SpinlockMode.PATTERN_HOLD
    o.spinlock_iterations = 10
    r = run_uks_periodic_multi_k_ewald3d(sysp, basis, kmesh, o, verbose=0)
    assert r.converged
    assert r.s_squared > 0.5     # the broken-symmetry pattern was held
    # DIIS stays suspended through the hold window (the fix that makes the
    # hold actually hold): no accelerator subspace before release.
    assert all(it.diis_subspace == 0 for it in r.scf_trace[:10])


def _periodic_stretched_h_chain():
    """Uniformly stretched H chain: atoms at 0 and 5 bohr in a 10-bohr box,
    so BOTH neighbour distances are 5 bohr, outside PBE's Coulson-Fischer
    point. (An 8-bohr box puts the periodic image at 8-5 = 3 bohr, inside
    the CF point, where NO strongly-polarized stationary state exists and
    an <S^2> > 0.5 assertion can never be satisfied; see
    test_periodic_spinlock_pattern_hold_multik_uks.)"""
    import vibeqc as vq

    lat = np.diag([10.0, 10.0, 10.0])
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [5.0, 0.0, 0.0])]
    sysp = vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=1)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _assert_uks_hold_pin(r, hold_iters):
    """Common assertions for the per-driver UKS PATTERN_HOLD pins: the SCF
    converged onto the strongly-broken state the seed encodes, and the
    accelerator stayed suspended through the hold window (the fix that makes
    the hold actually hold on weak-exchange functionals)."""
    assert r.converged
    assert r.s_squared > 0.5     # the broken-symmetry pattern was held
    assert all(it.diis_subspace == 0 for it in r.scf_trace[:hold_iters])


@pytest.mark.slow
def test_periodic_spinlock_pattern_hold_gamma_uks():
    """PATTERN_HOLD holds an atomic_spins AFM seed on the Γ UKS Ewald driver
    (weak exchange: PBE prefers the symmetric attractor, so this exercises
    the accelerator suspension, not just the MOM hook). Uniformly stretched
    H chain, PBE/STO-3G. Mirrors the multi-k pin."""
    import vibeqc as vq
    from vibeqc._vibeqc_core import SpinlockMode

    sysp, basis = _periodic_stretched_h_chain()
    o = vq.PeriodicKSOptions()
    o.functional = "pbe"
    o.initial_guess = vq.InitialGuess.SAD
    o.conv_tol_energy = 1e-7
    o.conv_tol_grad = 1e-5
    o.max_iter = 300
    o.atomic_spins = [1, -1]
    o.spinlock_mode = SpinlockMode.PATTERN_HOLD
    o.spinlock_iterations = 10
    r = vq.run_uks_periodic_gamma_ewald3d(sysp, basis, o, verbose=0)
    _assert_uks_hold_pin(r, 10)


@pytest.mark.slow
def test_periodic_spinlock_pattern_hold_gdf_uks():
    """PATTERN_HOLD holds an atomic_spins AFM seed on the GDF UKS driver.
    The GDF path routes through PeriodicSCFAccelerator (DIIS / EDIIS /
    ADIIS / KDIIS); the hold suspends whichever accelerator resolved.
    Uniformly stretched H chain, PBE/STO-3G. Mirrors the multi-k pin."""
    import vibeqc as vq
    from vibeqc._vibeqc_core import SpinlockMode

    sysp, basis = _periodic_stretched_h_chain()
    o = vq.PeriodicRHFOptions()
    o.initial_guess = vq.InitialGuess.SAD
    o.conv_tol_energy = 1e-7
    o.conv_tol_grad = 1e-5
    o.max_iter = 300
    o.atomic_spins = [1, -1]
    o.spinlock_mode = SpinlockMode.PATTERN_HOLD
    o.spinlock_iterations = 10
    r = vq.run_pbc_gdf_uks(sysp, basis, o, functional="pbe", verbose=0)
    _assert_uks_hold_pin(r, 10)


@pytest.mark.slow
def test_periodic_spinlock_pattern_hold_bipole_uks():
    """PATTERN_HOLD holds an atomic_spins AFM seed on the BIPOLE UKS driver
    (MultiKPeriodicUHFAccelerator suspension + the per-k MOM machinery).
    Uniformly stretched H chain, PBE/STO-3G at Γ. Mirrors the multi-k pin."""
    import vibeqc as vq
    from vibeqc._vibeqc_core import SpinlockMode

    sysp, basis = _periodic_stretched_h_chain()
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    o = vq.PeriodicKSOptions()
    o.functional = "pbe"
    o.initial_guess = vq.InitialGuess.SAD
    o.conv_tol_energy = 1e-7
    o.conv_tol_grad = 1e-5
    o.max_iter = 300
    o.atomic_spins = [1, -1]
    o.spinlock_mode = SpinlockMode.PATTERN_HOLD
    o.spinlock_iterations = 10
    r = vq.run_pbc_bipole_uks(sysp, basis, kmesh, o, verbose=0)
    _assert_uks_hold_pin(r, 10)


def _gdf_uhf_spinlock(mode=None, iters=0):
    """GDF UHF on stretched H2 with an ATOMSPIN seed, optional SPINLOCK."""
    import vibeqc as vq
    sysp, basis = _periodic_stretched_h2()
    o = vq.PeriodicRHFOptions()
    o.initial_guess = vq.InitialGuess.SAD
    o.conv_tol_energy = 1e-9
    o.max_iter = 200
    o.atomic_spins = [1, -1]
    if mode is not None:
        o.spinlock_mode = mode
        o.spinlock_iterations = iters
    return vq.run_pbc_gdf_uhf(sysp, basis, o, verbose=0)


def test_periodic_spinlock_pattern_hold_no_op_gdf_uhf():
    """PATTERN_HOLD on the GDF UHF driver (the PySCF-parity backend, which is
    otherwise spin-degenerate at the guess). On the easy stretched-H2 case the
    seed already holds, so the MOM hold is a no-op — same energy and ⟨S²⟩ as
    without the lock (its value is on hard systems where aufbau would collapse
    the seed). Confirms the GDF SCF-loop MOM hook runs end-to-end."""
    from vibeqc._vibeqc_core import SpinlockMode
    r0 = _gdf_uhf_spinlock()
    rlock = _gdf_uhf_spinlock(SpinlockMode.PATTERN_HOLD, 8)
    assert rlock.converged
    assert rlock.energy == pytest.approx(r0.energy, abs=1e-7)
    assert rlock.s_squared == pytest.approx(r0.s_squared, abs=1e-4)


def test_periodic_spinlock_spin_schedule_releases_gdf_uhf():
    """SPIN_SCHEDULE on the GDF UHF driver: lock the singlet (value=0) for 5
    cycles, then release to the target — must still converge (proving the
    two-phase delegation + GDF READ-restart of phase 2 work end-to-end)."""
    from vibeqc._vibeqc_core import SpinlockMode
    r = _gdf_uhf_spinlock(SpinlockMode.SPIN_SCHEDULE, 5)
    assert r.converged


def _bipole_uhf_spinlock(mode=None, iters=0):
    import vibeqc as vq
    sysp, basis = _periodic_stretched_h2()
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    o = vq.PeriodicRHFOptions()
    o.initial_guess = vq.InitialGuess.SAD
    o.conv_tol_energy = 1e-9
    o.max_iter = 200
    o.atomic_spins = [1, -1]
    if mode is not None:
        o.spinlock_mode = mode
        o.spinlock_iterations = iters
    return vq.run_pbc_bipole_uhf(sysp, basis, kmesh, o, verbose=0)


def test_periodic_spinlock_pattern_hold_no_op_bipole_uhf():
    """PATTERN_HOLD on the BIPOLE UHF driver, which reuses its per-k MOM
    machinery (the same kernel as ``use_mom``) gated to the spinlock window.
    No-op on the easy stretched-H2 case, as for the other drivers."""
    from vibeqc._vibeqc_core import SpinlockMode
    r0 = _bipole_uhf_spinlock()
    rlock = _bipole_uhf_spinlock(SpinlockMode.PATTERN_HOLD, 8)
    assert rlock.converged
    assert rlock.energy == pytest.approx(r0.energy, abs=1e-6)
    assert rlock.s_squared == pytest.approx(r0.s_squared, abs=1e-4)


def test_periodic_spinlock_spin_schedule_releases_bipole_uhf():
    """SPIN_SCHEDULE on the Gamma BIPOLE UHF scaffold: the two-phase
    run locks then releases via the g=0 READ restart and converges."""
    from vibeqc._vibeqc_core import SpinlockMode
    r = _bipole_uhf_spinlock(SpinlockMode.SPIN_SCHEDULE, 5)
    assert r.converged


def test_periodic_spinlock_via_run_periodic_job(tmp_path):
    """run_periodic_job exposes SPINLOCK (string mode) for UHF/UKS, rejects it
    for closed-shell methods, and rejects an unknown mode — mirroring the
    atomic_spins exposure."""
    import vibeqc as vq
    sysp, basis = _periodic_stretched_h2(box_bohr=12.0)

    # Rejected for closed-shell RHF (validated before the dry-run manifest).
    with pytest.raises(ValueError, match="(?i)spinlock.*UHF.*UKS"):
        vq.run_periodic_job(
            sysp, basis, method="RHF", spinlock="pattern_hold",
            spinlock_iterations=5, dry_run=True, output=str(tmp_path / "rhf"))
    # Unknown mode rejected.
    with pytest.raises(ValueError, match="(?i)unknown spinlock"):
        vq.run_periodic_job(
            sysp, basis, method="UHF", spinlock="bogus",
            spinlock_iterations=5, dry_run=True, output=str(tmp_path / "bad"))
    # Missing spinlock_iterations rejected.
    with pytest.raises(ValueError, match="(?i)spinlock_iterations"):
        vq.run_periodic_job(
            sysp, basis, method="UHF", spinlock="pattern_hold",
            dry_run=True, output=str(tmp_path / "noiter"))
    # Accepted for UHF (dry-run: validates + builds the manifest, no SCF).
    vq.run_periodic_job(
        sysp, basis, method="UHF", spinlock="pattern_hold",
        spinlock_iterations=8, atomic_spins=[1, -1],
        dry_run=True, output=str(tmp_path / "uhf"))


def test_read_rejects_unknown_extension(tmp_path):
    mol, basis = _h2o()
    bogus = tmp_path / "prev.txt"
    bogus.write_text("not a wavefunction")
    o = _rhf_opts(InitialGuess.READ); o.read_path = str(bogus)
    with pytest.raises((ValueError, RuntimeError), match="(?i)(qvf|molden|extension)"):
        run_rhf(mol, basis, o)


# ---- broken-symmetry singlet class: default-guess state selection ----
#
# The spin-averaged n_alpha == n_beta start (a2e190e46) must NOT block
# the broken-symmetry singlet class. The mechanism that descends from
# the symmetric saddle is the default-on stability escape (Seeger-Pople
# 1977 conditions; Lehtola 2020 §10), not the guess: measured identical
# solutions before and after a2e190e46 on the full set (H2 past the
# Coulson-Fischer point, O3, twisted ethylene, p-benzyne, Cr2 dimer;
# deepseek-zed guess-policy investigation 2026-08-15, handovers/
# HANDOVER_GUESS_SPIN_SYMMETRY.md). A deliberate AFM pattern remains
# available via atomic_spins (tested above). Values below are the
# internally measured post-fix pins, cross-checked out-of-process
# against PySCF for H2 (E=-0.95101800, <S^2>=0.7742 at R=3.0, the
# existing atom-spin test) and reproduced bit-identical on the pre-fix
# build for H2/O3/C2H4/pC6H4. The Cr2 escape gap is tracked as GitLab
# issue 138 (default run under-descends vs the [+1,-1]-seeded basin).

def _o3_631gstar(stretch=1.0):
    """O3 C2v, both O-O bonds = 2.4032 bohr * stretch, apex angle 116.78."""
    d = 2.4032 * stretch
    th = np.deg2rad(116.78) / 2
    x, z = d * np.sin(th), d * np.cos(th)
    return Molecule(
        [Atom(8, [0.0, 0.0, 0.0]),
         Atom(8, [x, 0.0, z]),
         Atom(8, [-x, 0.0, z])], multiplicity=1)


def _twisted_ethylene_90():
    """C2H4, one CH2 plane rotated 90 deg about the C-C axis (D2d)."""
    cc = 1.339 * ANGSTROM_TO_BOHR
    ch = 1.087 * ANGSTROM_TO_BOHR
    half = np.deg2rad(117.6) / 2
    xy, z = ch * np.sin(half), ch * np.cos(half)
    return Molecule(
        [Atom(6, [0.0, 0.0, -cc / 2]), Atom(6, [0.0, 0.0, cc / 2]),
         Atom(1, [xy, 0.0, -cc / 2 - z]), Atom(1, [-xy, 0.0, -cc / 2 - z]),
         Atom(1, [0.0, xy, cc / 2 + z]), Atom(1, [0.0, -xy, cc / 2 + z])],
        multiplicity=1)


def test_h2_below_coulson_fischer_stays_symmetric():
    """At R=2.0 bohr the symmetric solution is the UHF minimum; the default
    start (spin-averaged SAD) must converge to it bit-symmetrically."""
    mol, basis = _stretched_h2_at(2.0)
    r = run_uhf(mol, basis, _uhf_opts(InitialGuess.AUTO))
    assert r.converged
    assert abs(r.s_squared) < 1e-6
    assert r.n_stability_restarts == 0
    np.testing.assert_allclose(
        np.asarray(r.density_alpha), np.asarray(r.density_beta),
        rtol=0.0, atol=1e-12)


def test_h2_default_escapes_past_coulson_fischer():
    """At R=2.4 bohr (past the CF point) the default driver must escape the
    symmetric saddle to the broken-symmetry minimum, matching the
    atomic_spins reference exactly (a2e190e46 did not regress this)."""
    mol, basis = _stretched_h2_at(2.4)
    r_def = run_uhf(mol, basis, _uhf_opts(InitialGuess.AUTO))
    assert r_def.converged
    assert r_def.n_stability_restarts >= 1
    assert r_def.energy == pytest.approx(-0.9898973986, abs=1e-6)
    assert r_def.s_squared == pytest.approx(0.328585, abs=1e-2)

    o_seed = _uhf_opts(InitialGuess.SAD)
    o_seed.atomic_spins = [1, -1]
    r_seed = run_uhf(mol, basis, o_seed)
    assert r_seed.energy == pytest.approx(r_def.energy, abs=1e-8)
    assert r_seed.s_squared == pytest.approx(r_def.s_squared, abs=1e-4)


def test_o3_default_escapes_to_broken_symmetry():
    """O3/6-31G* at equilibrium: the symmetric SAD start converges to the
    symmetric saddle; the stability escape descends to the broken-symmetry
    singlet (85.8 mHa lower). Identical pre/post a2e190e46."""
    mol = _o3_631gstar(1.0)

    o_nostab = _uhf_opts(InitialGuess.AUTO)
    o_nostab.stability_check = False
    r_sym = run_uhf(mol, "6-31g*", o_nostab)
    assert r_sym.converged
    assert abs(r_sym.s_squared) < 1e-6
    assert r_sym.energy == pytest.approx(-224.2451365582, abs=1e-6)

    r_def = run_uhf(mol, "6-31g*", _uhf_opts(InitialGuess.AUTO))
    assert r_def.converged
    assert r_def.n_stability_restarts >= 1
    assert r_def.energy == pytest.approx(-224.3309684972, abs=1e-6)
    assert r_def.s_squared == pytest.approx(0.929960, abs=1e-2)
    assert r_def.energy < r_sym.energy - 1e-3


def test_twisted_ethylene_default_escapes_to_biradical():
    """Twisted C2H4 (90 deg)/STO-3G: default driver reaches the broken-
    symmetry biradical (148.7 mHa below the symmetric saddle)."""
    mol = _twisted_ethylene_90()

    o_nostab = _uhf_opts(InitialGuess.AUTO)
    o_nostab.stability_check = False
    r_sym = run_uhf(mol, "sto-3g", o_nostab)
    assert r_sym.converged
    assert abs(r_sym.s_squared) < 1e-6
    assert r_sym.energy == pytest.approx(-76.8553699161, abs=1e-6)

    r_def = run_uhf(mol, "sto-3g", _uhf_opts(InitialGuess.AUTO))
    assert r_def.converged
    assert r_def.n_stability_restarts >= 1
    assert r_def.energy == pytest.approx(-77.0040512410, abs=1e-6)
    assert r_def.s_squared == pytest.approx(1.043392, abs=1e-2)


def _rigidly_rotated(mol, axis, degrees):
    """The same molecule in a different orientation (Rodrigues)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    t = np.deg2rad(degrees)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    R = np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)
    return Molecule(
        [Atom(a.Z, list(R @ np.asarray(list(a.xyz), dtype=float))) for a in mol.atoms],
        mol.charge, mol.multiplicity)


def test_twisted_ethylene_symmetric_control_is_rotation_invariant():
    """A converged energy cannot depend on how the molecule is oriented (#210).

    90-deg twisted ethylene has an exactly degenerate frontier at the guess
    Fock -- the D2d e-pair, two orthogonal carbon p orbitals, gap ~2e-15 Ha --
    and one electron per spin to place in it. A bare aufbau truncation
    resolves that tie with whatever basis the eigensolver returns for the
    degenerate subspace, which is not a physical quantity: it varies with
    orientation and with the LAPACK build. The two stationary points it
    selects between are 58.090 mHa apart, and the lower one is correct: it is
    the delocalized, spatially symmetric solution, while the higher carries a
    spurious C+/C- charge separation.

    Swept rather than parametrised on a fixed orientation list on purpose:
    WHICH orientations land on the wrong basin is itself not reproducible (it
    moves between builds, and between the active and passive rotation
    conventions), so a short fixed list can pass vacuously while the defect is
    present. What reproduces every time is that some orientation differs.
    """
    reference = _twisted_ethylene_90()
    opts = _uhf_opts(InitialGuess.AUTO)
    opts.stability_check = False

    r_reference = run_uhf(reference, "sto-3g", opts)
    assert r_reference.converged
    assert r_reference.energy == pytest.approx(-76.8553699161, abs=1e-6)

    orientations = [((1, 0, 0), d) for d in (5, 10, 20, 25, 37, 45, 73, 90)]
    orientations += [((0, 0, 1), d) for d in (20, 45, 73)]
    orientations += [((1, 1, 1), d) for d in (20, 60, 123)]

    energies = {}
    for axis, degrees in orientations:
        rotated = run_uhf(_rigidly_rotated(reference, axis, degrees), "sto-3g", opts)
        assert rotated.converged, (axis, degrees)
        energies[(axis, degrees)] = rotated.energy

    spread = max(energies.values()) - min(energies.values())
    assert spread < 1e-9, (
        "the converged energy is orientation-dependent over "
        f"{len(orientations)} rigid rotations: spread {1e3 * spread:.3f} mHa; "
        f"energies {sorted(set(round(e, 9) for e in energies.values()))}"
    )
    for (axis, degrees), energy in energies.items():
        assert energy == pytest.approx(r_reference.energy, abs=1e-9), (axis, degrees)


def test_p_benzyne_default_escapes_to_biradical():
    """p-benzyne/6-31G*: default driver reaches the broken-symmetry
    biradical (179.2 mHa below the symmetric saddle)."""
    s = 1.40 * ANGSTROM_TO_BOHR
    ch = 1.08 * ANGSTROM_TO_BOHR
    atoms = []
    for i in range(6):
        ang = np.deg2rad(60 * i)
        atoms.append(Atom(6, [s * np.cos(ang), s * np.sin(ang), 0.0]))
    for i in (1, 2, 4, 5):
        ang = np.deg2rad(60 * i)
        atoms.append(Atom(1, [(s + ch) * np.cos(ang), (s + ch) * np.sin(ang), 0.0]))
    mol = Molecule(atoms, multiplicity=1)

    r_def = run_uhf(mol, "6-31g*", _uhf_opts(InitialGuess.AUTO))
    assert r_def.converged
    assert r_def.n_stability_restarts >= 1
    assert r_def.energy == pytest.approx(-229.4204881566, abs=1e-6)
    assert r_def.s_squared == pytest.approx(1.835773, abs=1e-2)


@pytest.mark.parametrize(
    ("distance", "energy_ceiling"),
    [
        (2.5, -2064.11761),
        (3.0, -2064.42469),
    ],
)
def test_cr2_default_escape_reaches_stable_basin(distance, energy_ceiling):
    """Issue 138: true defaults must escape both Cr2 singlet saddles.

    R=2.5 exposed a symmetry-confined Davidson start that certified a higher
    Hessian root; R=3.0 exposed a restart gate that rejected ordinary nHa
    uphill relaxation within a substantially lower basin. Both witnesses must
    now finish with a converged global stability verdict, not merely a lower
    energy or an exhausted retry count.
    """
    mol, basis = _cr2_at(distance)
    opts = UHFOptions()
    result = run_uhf(mol, basis, opts)

    assert result.converged
    assert result.stability_checked
    assert result.stability_analysis_converged
    assert result.n_stability_restarts >= 1
    assert not result.internal_instability
    assert result.stability_eigenvalue >= -opts.stability_tol
    assert result.energy <= energy_ceiling


@pytest.mark.parametrize(
    "atoms,charge,multiplicity,n_alpha,n_beta",
    [
        ([(8, [0, 0, 0]), (1, [0, 0, 1.8])], 0, 2, 5, 4),
        ([(2, [0, 0, 0])], 1, 2, 1, 0),
        ([(3, [0, 0, 0])], 1, 1, 1, 1),
        ([(8, [0, 0, 0]), (8, [0, 0, 2.3])], 0, 3, 9, 7),
        ([(26, [0, 0, 0])], 0, 5, 15, 11),
    ],
)
def test_raw_native_sad_has_exact_charged_spin_populations(
    atoms, charge, multiplicity, n_alpha, n_beta
):
    mol = Molecule([Atom(z, xyz) for z, xyz in atoms], charge, multiplicity)
    basis = BasisSet(mol, "sto-3g")
    s = _core.compute_overlap(basis)
    da, db = _core._guess_open_shell_density(
        mol, basis, n_alpha, n_beta, InitialGuess.SAD, is_periodic=False
    )
    assert np.trace(da @ s) == pytest.approx(n_alpha, abs=1e-11)
    assert np.trace(db @ s) == pytest.approx(n_beta, abs=1e-11)
    assert np.linalg.eigvalsh(da).min() >= -1e-11
    assert np.linalg.eigvalsh(db).min() >= -1e-11
    assert np.allclose(da, da.T, atol=1e-14)
    assert np.allclose(db, db.T, atol=1e-14)


def test_complex_weighted_density_normalization_preserves_phase_and_seed():
    from vibeqc.guess import normalize_density_guess, normalize_spin_density_guess

    alpha = np.array([[2, 0.2j], [-0.2j, 0.5]], dtype=complex)
    beta = np.array([[0.3, -0.1j], [0.1j, 1.2]], dtype=complex)
    overlaps = [
        np.array([[1.2, 0.1j], [-0.1j, 0.9]]),
        np.array([[0.8, -0.2j], [0.2j, 1.4]]),
    ]
    weights = [0.37, 0.63]
    density = normalize_density_guess(alpha + beta, overlaps, 4, weights=weights)
    assert np.iscomplexobj(density)
    assert np.max(np.abs(density.imag)) > 0
    assert sum(w * np.trace(density @ sk) for w, sk in zip(weights, overlaps)) == pytest.approx(4)
    da, db = normalize_spin_density_guess(
        alpha, beta, overlaps, 2, 2, weights=weights
    )
    for matrix, target in ((da, 2), (db, 2)):
        assert sum(w * np.trace(matrix @ sk) for w, sk in zip(weights, overlaps)) == pytest.approx(target)
        assert np.allclose(matrix, matrix.conj().T)
        assert np.linalg.eigvalsh(matrix).min() >= 0


@pytest.mark.parametrize("weights", [[1, -0.1], [0.2, 0.3], [float("nan"), 1], [1]])
def test_density_guess_rejects_invalid_k_weights(weights):
    from vibeqc.guess import normalize_density_guess

    with pytest.raises(ValueError, match="k weights"):
        normalize_density_guess(np.eye(2), [np.eye(2), np.eye(2)], 2, weights=weights)


@pytest.mark.parametrize("populations", [(3, 3), (3, 2)])
def test_atomic_seed_normalization_preserves_signs_and_unpolarized_atoms(populations):
    mol = Molecule(
        [Atom(1, [0, 0, 3 * i]) for i in range(3)],
        charge=3 - sum(populations), multiplicity=populations[0] - populations[1] + 1,
    )
    basis = BasisSet(mol, "sto-3g")
    alpha = np.diag([5., 1., 2.])
    beta = np.diag([1., 3., 2.])
    metric = np.array([[1.2, .1j, 0], [-.1j, .8, .03], [0, .03, 1.]])
    da, db = _core._normalize_atomic_guess(
        mol, basis, alpha, beta, metric, *populations, [1, -1, 0]
    )
    for density, target in zip((da, db), populations):
        assert np.trace(density @ metric) == pytest.approx(target, abs=1e-12)
        np.testing.assert_allclose(density, density.conj().T, atol=1e-14)
        assert np.linalg.eigvalsh(density).min() >= -1e-12
    spin = np.diag((da - db) @ metric).real
    assert spin[0] > 0
    assert spin[1] < 0
    assert spin[2] == pytest.approx(0, abs=1e-14)


def test_atomic_seed_rejects_off_atom_blocks_before_normalization():
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 3])])
    basis = BasisSet(mol, "sto-3g")
    alpha = np.array([[1, .4], [.4, .2]])
    beta = np.array([[.1, -.2], [-.2, 1]])
    metric = np.array([[1, .2], [.2, 1]])
    with pytest.raises(ValueError, match="block-diagonal"):
        _core._normalize_atomic_guess(mol, basis, alpha, beta, metric, 1, 1, [1, -1])


def test_atomic_seed_uses_route_metric_before_spin_attenuation():
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 3])])
    basis = BasisSet(mol, "sto-3g")
    metric = np.diag([.2, 1.])
    da, db = _core._normalize_atomic_guess(
        mol, basis, np.diag([5., 1.]), np.diag([1., 2.]), metric, 5, 4, [1, -1]
    )
    assert np.trace(da @ metric) == pytest.approx(5, abs=1e-12)
    assert np.trace(db @ metric) == pytest.approx(4, abs=1e-12)
    spin = np.diag((da - db) @ metric).real
    assert spin[0] > 0 > spin[1]
    with pytest.raises(ValueError, match="sign-preserving"):
        _core._normalize_atomic_guess(
            mol, basis, np.diag([5., 1.]), np.diag([1., 2.]), metric, 9, 0, [1, -1]
        )


@pytest.mark.parametrize("tags,valid", [([0, 0], True), ([1, -1], False)])
def test_zero_electron_atomic_seed_contract(tags, valid):
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 3])], charge=2)
    basis = BasisSet(mol, "sto-3g")
    if valid:
        da, db = _core._guess_open_shell_density(
            mol, basis, 0, 0, InitialGuess.SAD, atomic_spins=tags
        )
        np.testing.assert_array_equal(da, np.zeros((2, 2)))
        np.testing.assert_array_equal(db, da)
    else:
        with pytest.raises(ValueError, match="incompatible"):
            _core._guess_open_shell_density(
                mol, basis, 0, 0, InitialGuess.SAD, atomic_spins=tags
            )


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("kind", ["AUTO", "HCORE", "SAD", "SAP", "PATOM", "HUECKEL", "MINAO"])
def test_native_external_jk_executes_the_same_selected_guess(method, kind):
    """The direct numerical entry point consumes the public construction."""
    import vibeqc as vq

    mol = Molecule([Atom(8, [0, 0, 0]), Atom(1, [0, 0, 1.8]),
                    Atom(1, [1.7, 0, -0.5])])
    basis = BasisSet(mol, "sto-3g")
    opts = getattr(vq, method.upper() + "Options")()
    opts.initial_guess = InitialGuess.__members__[kind]
    opts.max_iter = 0
    if hasattr(opts, "stability_check"):
        opts.stability_check = False
    if method.endswith("ks"):
        opts.grid.n_radial = 20
        opts.grid.lebedev_order = 11
    expected = getattr(vq, "run_" + method)(mol, basis, opts)
    overlap = np.asarray(vq.compute_overlap(basis))
    hcore = np.asarray(vq.compute_kinetic(basis)) + np.asarray(vq.compute_nuclear(basis, mol))
    jk = vq.make_direct_jk_builder(basis)
    spin = method.startswith("u")
    args = [basis, 5, 5] if spin else [basis, 10]
    args.extend([overlap, hcore, mol.nuclear_repulsion(), jk])
    if method.endswith("ks"):
        args.append(vq.build_grid(mol, opts.grid))
    actual = getattr(vq, "run_" + method + "_scf_with_jk")(
        *args, opts, molecule=mol,
    )
    names = ("density_alpha", "density_beta") if spin else ("density",)
    for name in names:
        density = np.asarray(getattr(actual, name))
        np.testing.assert_allclose(density, getattr(expected, name), atol=1e-11)
        assert np.trace(density @ overlap) == pytest.approx(5 if spin else 10, abs=1e-10)
    assert actual.guess_selection.requested == opts.initial_guess
    assert actual.guess_selection.effective == expected.guess_selection.effective
    assert actual.guess_selection.transport == actual.guess_selection.effective
    assert actual.restart_basis.nbasis == basis.nbasis


def test_native_prepared_metadata_cannot_claim_an_unexecuted_builder():
    import vibeqc as vq
    from vibeqc.guess import GuessSelection

    mol = Molecule([Atom(2, [0, 0, 0])])
    basis = BasisSet(mol, "sto-3g")
    overlap = vq.compute_overlap(basis)
    hcore = vq.compute_kinetic(basis) + vq.compute_nuclear(basis, mol)
    opts = RHFOptions()
    opts.initial_guess = InitialGuess.SAP
    opts.max_iter = 0
    for selection in (
        GuessSelection(InitialGuess.SAP, InitialGuess.HCORE, InitialGuess.HCORE),
        GuessSelection(InitialGuess.SAP, InitialGuess.SAP, InitialGuess.READ),
    ):
        with pytest.raises(ValueError, match="provenance"):
            vq.run_rhf_scf_with_jk(
                basis, 2, overlap, hcore, 0, vq.make_direct_jk_builder(basis),
                opts, molecule=mol, guess_selection=selection,
            )
    opts.initial_guess = InitialGuess(500)
    with pytest.raises(ValueError, match="initial_guess"):
        vq.run_rhf_scf_with_jk(
            basis, 2, overlap, hcore, 0, vq.make_direct_jk_builder(basis),
            opts, initial_density=np.eye(1), molecule=mol,
        )


@pytest.mark.parametrize("method", ["rhf", "rks", "uhf", "uks", "rohf", "roks"])
def test_read_result_projects_geometry_and_exact_target_populations(method):
    import vibeqc as vq
    from vibeqc.guess_read import resolve_read_density_closed, resolve_read_densities_open

    spin = method in ("uhf", "uks", "rohf", "roks")
    atoms = [Atom(8, [0, 0, 0]), Atom(1, [0, 0, 1.8])] if spin else [
        Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4]),
    ]
    source = Molecule(atoms, multiplicity=2 if spin else 1)
    target = Molecule([Atom(a.Z, [a.xyz[0], a.xyz[1], a.xyz[2] * 1.3])
                       for a in atoms], multiplicity=2 if spin else 1)
    source_basis = BasisSet(source, "sto-3g")
    target_basis = BasisSet(target, "sto-3g")
    opts = getattr(vq, method.upper() + "Options")()
    opts.initial_guess = InitialGuess.SAD
    opts.max_iter = 0
    if hasattr(opts, "stability_check"):
        opts.stability_check = False
    prior = getattr(vq, "run_" + method)(source, source_basis, opts)
    assert prior.restart_basis.nbasis == source_basis.nbasis
    opts.initial_guess = InitialGuess.READ
    if spin:
        densities = resolve_read_densities_open(opts, target, target_basis, prior)
        populations = (5, 4)
        original = (prior.density_alpha, prior.density_beta)
    else:
        densities = (resolve_read_density_closed(opts, target, target_basis, prior),)
        populations = (2,)
        original = (prior.density,)
    overlap = np.asarray(vq.compute_overlap(target_basis))
    for density, population in zip(densities, populations):
        assert np.trace(density @ overlap) == pytest.approx(population, abs=1e-11)
        np.testing.assert_allclose(density, density.T, atol=1e-12)
    assert any(np.linalg.norm(d - old) > 1e-3 for d, old in zip(densities, original))


@pytest.mark.parametrize("source_kind", ["result", "preloaded"])
@pytest.mark.parametrize("populations", [(2, 0), (0, 2), (1.5, 0.5), (1, 1)])
def test_molecular_read_changes_spin_with_shared_density_normalization(source_kind, populations):
    """READ must populate empty channels and preserve local magnetic order."""
    from types import SimpleNamespace
    import vibeqc as vq
    from vibeqc.guess_read import resolve_read_densities_open

    mol = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 1.4])])
    basis = BasisSet(mol, "sto-3g")
    overlap = np.asarray(vq.compute_overlap(basis))
    # Distinct spatial distributions retain a local broken-symmetry pattern
    # even when the source's global spin populations require normalization.
    alpha = np.diag([populations[0], 0.]) / overlap[0, 0]
    beta = np.diag([0., populations[1]]) / overlap[1, 1]
    options = UHFOptions()
    options.initial_guess = InitialGuess.READ
    prior = SimpleNamespace(density_alpha=alpha, density_beta=beta, restart_basis=basis)
    if source_kind == "preloaded":
        options.read_density_alpha = alpha
        options.read_density_beta = beta
        prior = None
    da, db = resolve_read_densities_open(options, mol, basis, prior)
    if all(populations):
        np.testing.assert_allclose(da, alpha / populations[0], atol=1e-12)
        np.testing.assert_allclose(db, beta / populations[1], atol=1e-12)
    else:
        np.testing.assert_allclose(da + db, alpha + beta, atol=1e-12)
        np.testing.assert_allclose(da, db, atol=1e-12)
    for density in (da, db):
        assert np.trace(density @ overlap) == pytest.approx(1., abs=1e-12)
        assert np.linalg.eigvalsh(density).min() >= -1e-12


@pytest.mark.parametrize("method", ["uhf", "uks", "rohf", "roks"])
def test_molecular_read_from_fully_polarized_source_converges(method):
    import vibeqc as vq

    atoms = [Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 1.4])]
    triplet = Molecule(atoms, multiplicity=3)
    singlet = Molecule(atoms)
    source_basis = BasisSet(triplet, "sto-3g")
    basis = BasisSet(singlet, "sto-3g")
    source_options = UHFOptions()
    source_options.stability_check = False
    source = run_uhf(triplet, source_basis, source_options)
    assert source.converged
    options = getattr(vq, method.upper() + "Options")()
    if hasattr(options, "stability_check"):
        options.stability_check = False
    options.initial_guess = InitialGuess.READ
    driver = getattr(vq, "run_" + method)
    result = driver(singlet, basis, options, read_from=source)
    options.initial_guess = InitialGuess.HCORE
    reference = driver(singlet, basis, options)
    assert result.converged and reference.converged
    assert result.energy == pytest.approx(reference.energy, abs=1e-9)
    assert result.guess_selection.effective == InitialGuess.READ
    overlap = np.asarray(vq.compute_overlap(basis))
    for density in (result.density_alpha, result.density_beta):
        assert np.trace(density @ overlap) == pytest.approx(1., abs=1e-10)


def test_molecular_read_geometry_projection_retains_local_broken_symmetry():
    from types import SimpleNamespace
    import vibeqc as vq
    from vibeqc.guess_read import resolve_read_densities_open, _to_current_basis

    source = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 4.])])
    target = Molecule([Atom(1, [.3, 0., 0.]), Atom(1, [0., 0., 4.])])
    source_basis, basis = BasisSet(source, "sto-3g"), BasisSet(target, "sto-3g")
    prior = SimpleNamespace(density_alpha=np.diag([1., 0.]),
                            density_beta=np.diag([0., 1.]), restart_basis=source_basis)
    overlap = np.asarray(vq.compute_overlap(basis))
    projected = [_to_current_basis(d, source_basis, basis)
                 for d in (prior.density_alpha, prior.density_beta)]
    # Only one atom moves: projection changes the global spin counts even
    # though the source is a local AFM singlet. Atomic spin averaging here
    # would erase its magnetic order.
    assert abs(np.trace((projected[0] - projected[1]) @ overlap)) > 1e-4
    da, db = resolve_read_densities_open(UHFOptions(), target, basis, prior)
    assert np.linalg.norm(da - db) > .5
    for d in (da, db):
        assert np.trace(d @ overlap) == pytest.approx(1., abs=1e-12)


@pytest.mark.parametrize("complex_mos", [False, True])
def test_read_mo_payload_rejects_nonfinite_and_complex_occupations(complex_mos):
    from vibeqc.guess_read import _density_from_mos, _density_from_complex_mos

    build = _density_from_complex_mos if complex_mos else _density_from_mos
    with pytest.raises(ValueError):
        build(np.array([[complex(1, np.nan)]]), np.ones(1))
    with pytest.raises(ValueError, match="occupations"):
        build(np.ones((1, 1)), np.array([1 + 1j]))


def test_normalization_preserves_exact_singlet_and_valid_seed_bits():
    from vibeqc.guess import normalize_density_guess, normalize_spin_density_guess

    overlap = np.array([[1.0, 0.2], [0.2, 1.0]])
    density = np.eye(2) * (1.0 + 2e-15)
    np.testing.assert_array_equal(normalize_density_guess(density, overlap, 2), density)
    alpha, beta = normalize_spin_density_guess(density, density, overlap, 2, 2)
    np.testing.assert_array_equal(alpha, density)
    np.testing.assert_array_equal(beta, density)
    alpha, beta = normalize_spin_density_guess(3 * density, 3 * density, overlap, 2, 2)
    np.testing.assert_array_equal(alpha, beta)
    assert np.trace(alpha @ overlap) == pytest.approx(2, abs=1e-12)


def test_bloch_restart_exact_permutation_gamma_projection_and_refusals():
    from types import SimpleNamespace
    from vibeqc.guess_read import _map_periodic_restart
    from vibeqc.guess import normalize_density_k_guess
    from vibeqc import PeriodicSystem

    system = PeriodicSystem(3, np.eye(3) * 8, [Atom(2, [0, 0, 0])])
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    points = np.array([[0.13, 0.07, 0.02], [-0.13, -0.07, -0.02]])
    weights = np.array([0.37, 0.63])
    context = (basis, np.asarray(system.lattice), points, weights)
    blocks = [np.array([[1.5 + 0j]]), np.array([[2.2 + 0j]])]
    mesh = SimpleNamespace(kpoints=points[::-1], weights=weights[::-1])
    mapped = _map_periodic_restart(blocks, context, basis, system, mesh)
    for actual, expected in zip(mapped, blocks[::-1]):
        np.testing.assert_array_equal(actual, expected)
    normalized = normalize_density_k_guess(mapped, [np.eye(1)] * 2, mesh.weights, 2)
    assert sum(w * np.trace(d).real for w, d in zip(mesh.weights, normalized)) == pytest.approx(2, abs=1e-12)
    assert normalized[0][0, 0] / normalized[1][0, 0] == pytest.approx(2.2 / 1.5)
    gamma = (basis, np.asarray(system.lattice), np.zeros((1, 3)), np.ones(1))
    projected = _map_periodic_restart([np.array([[2.0]])], gamma, basis, system, mesh)
    assert len(projected) == 2
    np.testing.assert_allclose(projected, [np.array([[2.0]])] * 2, atol=1e-12)
    for incompatible in (
        SimpleNamespace(kpoints=points + 0.04, weights=weights),
        SimpleNamespace(kpoints=points, weights=weights[::-1]),
    ):
        with pytest.raises(NotImplementedError, match="incompatible k meshes"):
            _map_periodic_restart(blocks, context, basis, system, incompatible)
    changed = PeriodicSystem(3, np.eye(3) * 9, [Atom(2, [0, 0, 0])])
    with pytest.raises(NotImplementedError, match="same lattice"):
        _map_periodic_restart(blocks, context, basis, changed, mesh)


def test_bloch_restart_rejects_nonhermitian_density_before_repair():
    from vibeqc.guess import normalize_density_k_guess

    density = np.array([[1, 1j], [1j, 1]])
    with pytest.raises(ValueError, match="Hermitian"):
        normalize_density_k_guess([density], [np.eye(2)], [1], 2)


@pytest.mark.parametrize("method", ["RHF", "RKS"])
@pytest.mark.parametrize("tags", [[1, -1], [99]])
def test_direct_native_periodic_restricted_driver_refuses_atomic_seed(method, tags):
    import vibeqc as vq

    system = vq.PeriodicSystem(3, np.eye(3) * 16, [
        vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions() if method == "RHF" else vq.PeriodicKSOptions()
    opts.initial_guess = vq.InitialGuess.SAD
    opts.atomic_spins = tags
    opts.max_iter = 0
    with pytest.raises(NotImplementedError, match="atomic spin seed"):
        if method == "RHF":
            _core.run_rhf_periodic_gamma(system, basis, opts)
        else:
            _core.run_rks_periodic(system, basis, vq.monkhorst_pack(system, [1, 1, 1]), opts)


@pytest.mark.parametrize("tags", [[0.5], [1.9], [float("nan")]])
def test_public_periodic_atomic_tags_are_not_truncated(tmp_path, tags):
    import vibeqc as vq

    system = vq.PeriodicSystem(3, np.eye(3) * 16, [vq.Atom(1, [0, 0, 0])], 0, 2)
    with pytest.raises(ValueError, match="atomic_spins tags"):
        vq.run_periodic_job(
            system, "sto-3g", method="UHF", initial_guess="SAD",
            atomic_spins=tags, output=str(tmp_path / "seed"), dry_run=True,
        )


@pytest.mark.parametrize("method", ["rohf", "roks"])
def test_public_restricted_open_read_source_projects_changed_geometry(method):
    import vibeqc as vq

    molecule = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 1, 2)
    basis = BasisSet(molecule, "sto-3g")
    options = getattr(vq, method.upper() + "Options")()
    if method == "roks":
        options.functional = "lda"
    source = getattr(vq, "run_" + method)(molecule, basis, options)
    moved = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.8])], 1, 2)
    target = BasisSet(moved, "sto-3g")
    options.initial_guess = InitialGuess.READ
    result = getattr(vq, "run_" + method)(moved, target, options, read_from=source)
    assert result.converged
    overlap = np.asarray(vq.compute_overlap(target))
    assert np.trace(result.density_alpha @ overlap) == pytest.approx(1, abs=1e-11)
    assert np.trace(result.density_beta @ overlap) == pytest.approx(0, abs=1e-11)
    assert result.guess_selection.effective == InitialGuess.READ


def test_legacy_result_without_basis_cannot_claim_geometry_compatibility():
    from types import SimpleNamespace
    from vibeqc.guess_read import resolve_read_density_closed
    molecule = Molecule([Atom(2, [0., 0., 0.])])
    basis = BasisSet(molecule, "sto-3g")
    options = RHFOptions()
    options.initial_guess = InitialGuess.READ
    legacy = SimpleNamespace(density=np.ones((basis.nbasis, basis.nbasis)))
    with pytest.raises(ValueError, match="AO-basis snapshot"):
        resolve_read_density_closed(options, molecule, basis, legacy)


@pytest.mark.parametrize("method", ["uhf", "uks"])
def test_native_external_jk_refuses_unimplemented_spin_schedule(method):
    import vibeqc as vq

    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    basis = BasisSet(mol, "sto-3g")
    opts = getattr(vq, method.upper() + "Options")()
    opts.initial_guess = InitialGuess.HCORE
    opts.max_iter = 1
    opts.stability_check = False
    opts.spinlock_mode = vq.SpinlockMode.SPIN_SCHEDULE
    opts.spinlock_iterations = 1
    opts.spinlock_value = 2
    args = [basis, 1, 1, vq.compute_overlap(basis),
            vq.compute_kinetic(basis) + vq.compute_nuclear(basis, mol),
            mol.nuclear_repulsion(), vq.make_direct_jk_builder(basis)]
    if method == "uks":
        opts.grid.n_radial = 20
        opts.grid.lebedev_order = 11
        args.append(vq.build_grid(mol, opts.grid))
    with pytest.raises(ValueError, match="does not implement SPIN_SCHEDULE"):
        getattr(vq, "run_" + method + "_scf_with_jk")(*args, opts, molecule=mol)


@pytest.mark.parametrize("method", ["rhf", "rks", "uhf", "uks"])
def test_native_prepared_restart_removes_antisymmetric_roundoff(method):
    import vibeqc as vq

    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    basis = BasisSet(mol, "sto-3g")
    overlap = np.asarray(vq.compute_overlap(basis))
    density = np.linalg.inv(overlap)
    density[0, 1] += 1e-9
    density[1, 0] -= 1e-9
    assert np.trace(density @ overlap) == pytest.approx(2, abs=1e-14)
    opts = getattr(vq, method.upper() + "Options")()
    opts.initial_guess = InitialGuess.READ
    opts.max_iter = 0
    if hasattr(opts, "stability_check"):
        opts.stability_check = False
    spin = method.startswith("u")
    args = [basis, 1, 1] if spin else [basis, 2]
    args.extend([overlap, vq.compute_kinetic(basis) + vq.compute_nuclear(basis, mol),
                 mol.nuclear_repulsion(), vq.make_direct_jk_builder(basis)])
    if method.endswith("ks"):
        opts.grid.n_radial = 20
        opts.grid.lebedev_order = 11
        args.append(vq.build_grid(mol, opts.grid))
    payload = dict(init_alpha=density / 2, init_beta=density / 2) if spin else dict(
        initial_density=density)
    result = getattr(vq, "run_" + method + "_scf_with_jk")(
        *args, opts, molecule=mol, **payload,
    )
    for name in (("density_alpha", "density_beta") if spin else ("density",)):
        actual = np.asarray(getattr(result, name))
        np.testing.assert_array_equal(actual, actual.T)
        assert np.trace(actual @ overlap) == pytest.approx(1 if spin else 2, abs=1e-13)


@pytest.mark.parametrize("complex_dtype", [False, True])
def test_native_density_normalization_refuses_overflow(complex_dtype):
    dtype = complex if complex_dtype else float
    with pytest.raises(ValueError, match="finite.*metric trace"):
        _core._normalize_guess_density(
            np.eye(3, dtype=dtype) * 8e307, np.eye(3, dtype=dtype), 3,
        )


def _ecp_guess_atom(z=47, basis_name="def2-svp", *, xml=False):
    from vibeqc.ecp_metadata import attach_inline_ecp_options_from_basis_sidecar
    from vibeqc.guess import guess_ecp_context

    mol = Molecule([Atom(z, [0., 0., 0.])], 0, z % 2 + 1)
    basis = BasisSet(mol, basis_name)
    opts = UHFOptions()
    if xml:
        opts.ecp_centers = [_core.ECPCenter(z, [0., 0., 0.])]
        opts.ecp_library = "ecp28mdf"
    else:
        attach_inline_ecp_options_from_basis_sidecar(opts, mol, basis)
    context = guess_ecp_context(opts)
    assert context.active
    return mol, basis, opts, context


@pytest.mark.parametrize("xml", [False, True])
def test_ecp_atomic_sad_has_valence_configuration_before_normalization(xml):
    """Ag's 28-core removes [Ar]3d10, not the neutral Ni configuration."""
    mol, basis, opts, context = _ecp_guess_atom(xml=xml)
    density = np.asarray(_core.sad_density(mol, basis, context))
    overlap = np.asarray(_core.compute_overlap(basis))
    ao_l = np.array([int(sh.l) for sh in basis.shells()
                     for _ in range(2 * int(sh.l) + 1)])
    populations = np.diag(density @ overlap)
    np.testing.assert_allclose(
        [populations[ao_l == l].sum() for l in range(4)], [3, 6, 10, 0], atol=2e-10,
    )
    np.testing.assert_allclose(density, density.T, atol=1e-13)
    da, db = _core._guess_open_shell_density(
        mol, basis, 10, 9, InitialGuess.SAD, overlap=overlap,
        ecp_context=context,
    )
    assert np.trace(da @ overlap) == pytest.approx(10, abs=2e-11)
    assert np.trace(db @ overlap) == pytest.approx(9, abs=2e-11)
    # Equal physical spin counts must not introduce a hidden Hund seed.
    da, db = _core._guess_open_shell_density(
        mol, basis, 9, 9, InitialGuess.SAD, overlap=overlap, ecp_context=context,
    )
    np.testing.assert_array_equal(da, db)


@pytest.mark.parametrize("fault", ["center", "nan", "extra_block", "core_sum", "fractional_charge", "orphan_charge", "orphan_count"])
def test_ecp_guess_rejects_malformed_operator_context(fault):
    from vibeqc.guess import validate_guess_ecp

    mol, basis, opts, context = _ecp_guess_atom()
    if fault == "center":
        context.primitive_centers = [[1., 0., 0.]]
    elif fault == "nan":
        context.primitive_centers = [[float("nan"), 0., 0.]]
    elif fault == "extra_block":
        context.primitive_blocks = list(context.primitive_blocks) * 2
    elif fault == "core_sum":
        context.total_ncore = 18
    elif fault == "fractional_charge":
        context.effective_charges = [19.5]
    elif fault == "orphan_charge":
        context.primitive_blocks = []
        context.primitive_centers = []
    else:
        context.primitive_blocks = []
        context.primitive_centers = []
        context.effective_charges = []
    with pytest.raises(ValueError, match="initial guess"):
        validate_guess_ecp(InitialGuess.SAD, context, mol)


def test_ecp_sad_uses_actual_shells_and_does_not_alias_equal_elements():
    from vibeqc.guess import guess_ecp_context

    atom, atom_basis, opts, _ = _ecp_guess_atom()
    mol = Molecule([Atom(47, [0., 0., 0.]), Atom(47, [0., 0., 12.])])
    shells = []
    for a in range(2):
        for shell in atom_basis.shells():
            shell.origin = [0., 0., 12. * a]
            shell.atom_index = a
            if a:
                shell.exponents = [1.2 * x for x in shell.exponents]
            shells.append(shell)
    # A deliberately non-library name proves there is no atomic name reload.
    basis = BasisSet(mol, shells, "custom ECP atom pair", True)
    opts.ecp_primitive_blocks = list(opts.ecp_primitive_blocks) * 2
    opts.ecp_primitive_centers = [[0., 0., 0.], [0., 0., 12.]]
    opts.ecp_effective_charges = [19., 19.]
    opts.ecp_total_ncore = 56
    density = np.asarray(_core.sad_density(mol, basis, guess_ecp_context(opts)))
    n = atom_basis.nbasis
    for a in range(2):
        own = []
        for shell in shells[a * atom_basis.nshells:(a + 1) * atom_basis.nshells]:
            shell.origin = [0., 0., 0.]
            shell.atom_index = 0
            own.append(shell)
        own_basis = BasisSet(atom, own, "custom ECP atom", True)
        own_opts = UHFOptions()
        own_opts.ecp_primitive_blocks = [opts.ecp_primitive_blocks[a]]
        own_opts.ecp_primitive_centers = [[0., 0., 0.]]
        own_opts.ecp_effective_charges = [19.]
        own_opts.ecp_total_ncore = 28
        expected = _core.sad_density(atom, own_basis, guess_ecp_context(own_opts))
        np.testing.assert_allclose(density[a*n:(a+1)*n, a*n:(a+1)*n], expected, atol=1e-12)
    assert np.max(abs(density[:n, :n] - density[n:, n:])) > 1e-4


def test_ecp_atomic_guess_refuses_missing_angular_channel():
    mol, basis, opts, context = _ecp_guess_atom()
    pruned = BasisSet(mol, [s for s in basis.shells() if s.l != 2], "pruned Ag", True)
    with pytest.raises(NotImplementedError, match="cannot represent ECP valence"):
        _core.sad_density(mol, pruned, context)


@pytest.mark.parametrize("method", ["rhf", "uhf", "rohf", "rks", "uks", "roks"])
@pytest.mark.parametrize("guess", [InitialGuess.SAP, InitialGuess.MINAO])
def test_public_ecp_sap_minao_construct_valence_guesses(method, guess):
    import vibeqc as vq
    mol, basis, _, _ = _ecp_guess_atom(z=30, basis_name="lanl2dz")
    options = getattr(vq, method.upper() + "Options")()
    options.initial_guess = guess
    options.max_iter = 1
    if method.endswith("ks"):
        options.functional = "lda"
    result = getattr(vq, "run_" + method)(mol, basis, options)
    assert np.isfinite(result.energy)
    assert result.guess_selection.effective == guess
    densities = [result.density] if method in ('rhf','rks') else [result.density_alpha,result.density_beta]
    overlap = vq.compute_overlap(basis)
    assert sum(np.trace(d@overlap) for d in densities) == pytest.approx(12,abs=1e-8)


def test_periodic_signature_documents_literal_auto_default():
    """The published default of ``initial_guess`` is the literal ``"AUTO"``.

    #43 shipped an API reference reading ``initial_guess=<object object>``
    after a sentinel replaced the string. Python's own signature rendering
    shows the same text Sphinx would derive it from, so this half of the
    check runs in every environment, including the documented ``.[test]``
    install, which does not carry Sphinx (#251).
    """
    import inspect
    from vibeqc.periodic_runner import run_periodic_job

    parameter = inspect.signature(run_periodic_job).parameters["initial_guess"]
    assert parameter.default == "AUTO"
    # Render the default the way the reference does: without the annotation.
    plain = parameter.replace(annotation=inspect.Parameter.empty)
    rendered = str(inspect.Signature([plain]))
    assert "initial_guess='AUTO'" in rendered
    assert "<object object" not in rendered


def test_periodic_signature_renders_literal_auto_default_in_sphinx():
    """The rendering the published API reference actually uses (#43).

    Sphinx is a ``docs`` extra, not a ``test`` extra, so this half runs only
    where it is installed; the property it renders is pinned without Sphinx
    in the test above (#251).
    """
    pytest.importorskip("sphinx")
    import inspect
    from sphinx.util.inspect import stringify_signature
    from vibeqc.periodic_runner import run_periodic_job

    signature = inspect.signature(run_periodic_job)
    assert "initial_guess='AUTO'" in stringify_signature(signature, show_annotation=False)
    assert "<object object" not in stringify_signature(
        inspect.Signature([signature.parameters["initial_guess"]])
    )


@pytest.mark.parametrize("method", ["rhf", "uhf", "rohf", "rks", "uks", "roks"])
def test_public_molecular_ecp_sad_converges_with_exact_spin_populations(method):
    import vibeqc as vq

    spin = method in ("uhf", "rohf", "uks", "roks")
    molecule = Molecule([Atom(11, [0., 0., 0.]), Atom(1, [0., 0., 3.5])],
                        1 if spin else 0, 2 if spin else 1)
    basis = BasisSet(molecule, "lanl2dz")
    options = getattr(vq, method.upper() + "Options")()
    options.initial_guess = InitialGuess.SAD
    options.max_iter = 100
    if hasattr(options, "stability_check"):
        options.stability_check = False
    if method.endswith("ks"):
        options.functional = "lda"
    result = getattr(vq, "run_" + method)(molecule, basis, options)
    assert result.converged
    overlap = np.asarray(_core.compute_overlap(basis))
    densities = (result.density_alpha, result.density_beta) if spin else (result.density,)
    expected = (1, 0) if spin else (2,)
    for d, n in zip(densities, expected):
        assert np.trace(np.asarray(d) @ overlap) == pytest.approx(n, abs=1e-10)
    assert result.guess_selection.effective == InitialGuess.SAD


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("guess", [InitialGuess.SAP, InitialGuess.MINAO])
def test_native_ecp_restart_density_preserves_selector_provenance(method, guess):
    import vibeqc as vq
    from vibeqc.ecp_metadata import attach_inline_ecp_options_from_basis_sidecar

    molecule = Molecule([Atom(11, [0., 0., 0.]), Atom(1, [0., 0., 3.5])])
    basis = BasisSet(molecule, "lanl2dz")
    options = getattr(vq, method.upper() + "Options")()
    attach_inline_ecp_options_from_basis_sidecar(options, molecule, basis)
    options.initial_guess = guess
    options.max_iter = 0
    overlap = vq.compute_overlap(basis)
    # Only the capability boundary is under test: an injected density must
    # never excuse an explicit unsupported physical construction.
    density = np.linalg.inv(overlap) / basis.nbasis
    spin = method.startswith("u")
    args = [basis, 1, 1] if spin else [basis, 2]
    args += [overlap, np.eye(basis.nbasis), 0., vq.make_direct_jk_builder(basis)]
    if method.endswith("ks"):
        options.grid.n_radial = 10
        options.grid.lebedev_order = 11
        args.append(vq.build_grid(molecule, options.grid))
    kwargs = dict(init_alpha=density, init_beta=density) if spin else dict(initial_density=2*density)
    result = getattr(vq, "run_" + method + "_scf_with_jk")(
        *args, options, molecule=molecule, **kwargs)
    assert result.guess_selection.requested == guess
    assert result.guess_selection.effective == InitialGuess.READ
    assert result.guess_selection.transport == InitialGuess.READ


def test_fragmo_preserves_explicit_parent_ecp_override(monkeypatch):
    import vibeqc as vq
    from vibeqc.guess_fragmo import Fragment, resolve_fragmo_density_closed

    # This explicit XML operator replaces 10 core electrons. The basis has no
    # ECP sidecar, so dropping the request would create a bare-Z fragment.
    molecule = Molecule([Atom(30, [0., 0., 0.]), Atom(30, [0., 0., 12.])])
    basis = BasisSet(molecule, "6-31g")
    options = RHFOptions()
    options.ecp_centers = [_core.ECPCenter(30, list(a.xyz)) for a in molecule.atoms]
    options.ecp_library = "ecp10mdf"
    options.initial_guess = InitialGuess.FRAGMO
    original = vq.run_rhf
    seen = []
    def probe(mol, basis, opts):
        assert len(opts.ecp_centers) == 1 and opts.ecp_library == "ecp10mdf"
        result = original(mol, basis, opts)
        assert result.ecp_total_ncore == 10
        seen.append(result)
        return result
    monkeypatch.setattr(vq, "run_rhf", probe)
    density = resolve_fragmo_density_closed(
        options, molecule, basis, [Fragment([0]), Fragment([1])],
    )
    assert len(seen) == 2
    assert np.trace(density @ _core.compute_overlap(basis)) == pytest.approx(40, abs=1e-9)


@pytest.mark.parametrize('method', ['rhf_gamma', 'rhf_k', 'rks_k'])
def test_native_periodic_read_source_roundtrip(method):
    import vibeqc as vq
    system = vq.PeriodicSystem(3, np.eye(3)*16, [Atom(1,[0,0,0]), Atom(1,[0,0,1.4])])
    basis = BasisSet(system.unit_cell_molecule(), 'sto-3g')
    mesh = vq.monkhorst_pack(system, [1,1,3])
    options = (vq.PeriodicRHFOptions() if method == 'rhf_gamma' else
               vq.PeriodicSCFOptions() if method == 'rhf_k' else vq.PeriodicKSOptions())
    options.lattice_opts.cutoff_bohr = 3
    options.lattice_opts.nuclear_cutoff_bohr = 3
    options.max_iter = 30
    if method == 'rks_k':
        options.functional = 'lda'
        options.grid.n_radial = 15
        options.grid.lebedev_order = 11
    run = (lambda **kw: vq.run_rhf_periodic_gamma(system,basis,options,**kw)) if method == 'rhf_gamma' else (
        lambda **kw: (vq.run_rhf_periodic if method == 'rhf_k' else vq.run_rks_periodic)(system,basis,mesh,options,**kw))
    source = run()
    assert source.converged
    options.initial_guess = InitialGuess.READ
    restarted = run(read_from=source)
    assert restarted.converged
    assert restarted.energy == pytest.approx(source.energy, abs=1e-8)
    assert restarted.guess_selection.effective == InitialGuess.READ
    assert options.read_density.size == 0 if method == 'rhf_gamma' else options.read_density_k == []


@pytest.mark.parametrize('bad', ['missing','negative','nonhermitian','nonfinite','time_reversal'])
def test_native_periodic_read_rejects_invalid_blocks(bad):
    import vibeqc as vq
    system = vq.PeriodicSystem(3,np.eye(3)*16,[Atom(1,[0,0,0]),Atom(1,[0,0,1.4])])
    basis = BasisSet(system.unit_cell_molecule(),'sto-3g')
    mesh = vq.monkhorst_pack(system,[1,1,3])
    options = vq.PeriodicSCFOptions()
    options.initial_guess = InitialGuess.READ
    options.max_iter = 0
    options.lattice_opts.cutoff_bohr = 3
    options.lattice_opts.nuclear_cutoff_bohr = 3
    blocks = [np.eye(2,dtype=complex) for _ in range(3)]
    if bad == 'missing': blocks.pop()
    elif bad == 'negative': blocks[0][0,0] = -1
    elif bad == 'nonhermitian': blocks[0][0,1] = .3j
    elif bad == 'nonfinite': blocks[0][0,0] = np.nan
    else: blocks[1][0,0] = 2
    options.read_density_k = blocks
    with pytest.raises(ValueError, match='READ'):
        vq.run_rhf_periodic(system,basis,mesh,options)


@pytest.mark.parametrize('periodic',[False,True])
def test_ecp_minao_valence_reference(periodic):
    from vibeqc.guess import initial_density_closed_shell, initial_densities_open_shell
    mol,basis,options,context = _ecp_guess_atom(z=30,basis_name='lanl2dz')
    extra = dict(is_periodic=False)
    if periodic:
        import vibeqc as vq
        system = vq.PeriodicSystem(3,np.eye(3)*20,list(mol.atoms))
        lo = vq.LatticeSumOptions(); lo.cutoff_bohr=3
        extra = dict(is_periodic=True,periodic_system=system,lattice_opts=lo)
    s = _core.compute_overlap(basis)
    count = (mol.n_electrons() - context.total_ncore)//2
    d = initial_density_closed_shell(mol,basis,count,InitialGuess.MINAO,overlap=s,ecp_context=context,**extra)
    assert np.trace(d@s) == pytest.approx(2*count,abs=1e-10)
    assert np.linalg.eigvalsh(d).min() > -1e-10
    a,b = initial_densities_open_shell(mol,basis,count+1,count-1,InitialGuess.MINAO,overlap=s,ecp_context=context,**extra)
    assert np.trace(a@s) == pytest.approx(count+1,abs=1e-10)
    assert np.trace(b@s) == pytest.approx(count-1,abs=1e-10)


def test_periodic_fragmo_image_phases_and_translation_invariance():
    import vibeqc as vq
    from vibeqc.guess_fragmo import resolve_periodic_fragmo_source
    lattice=np.array([[16.,2.,4.],[0.,16.,3.],[0.,0.,16.]])
    system=vq.PeriodicSystem(3,lattice,[Atom(1,[0,0,.7]),Atom(1,lattice[:,2]-[0,0,.7])])
    basis=BasisSet(system.unit_cell_molecule(),'sto-3g')
    mesh=vq.monkhorst_pack(system,[1,1,3])
    source=resolve_periodic_fragmo_source(vq.PeriodicRHFOptions(),system,basis,mesh,[vq.PeriodicFragment([0,1],images=[[0,0,0],[0,0,-1]])])
    moved=resolve_periodic_fragmo_source(vq.PeriodicRHFOptions(),system,basis,mesh,[vq.PeriodicFragment([0,1],images=[[2,1,3],[2,1,2]])])
    np.testing.assert_allclose(source.density,moved.density,atol=1e-10)
    for k,d in zip(mesh.kpoints,source.density):
        assert d[0,1] == pytest.approx(source.density[0][0,1]*np.exp(-1j*np.dot(k,lattice[:,2])),abs=1e-10)
        assert np.linalg.eigvalsh(d).min() > -1e-12
    assert max(abs(d[0,1].imag) for d in source.density) > .1


@pytest.mark.parametrize('method,route', [('RHF','gdf'),('UHF','gdf'),('ROHF','gdf'),
    ('RKS','gdf'),('UKS','gdf'),('ROKS','bipole'),('UKS','gpw'),('RKS','gapw')])
def test_public_periodic_fragmo_transport(tmp_path,method,route):
    import vibeqc as vq
    system=vq.PeriodicSystem(3,np.eye(3)*12,[Atom(1,[6,6,5.3]),Atom(1,[6,6,6.7])])
    basis=BasisSet(system.unit_cell_molecule(),'sto-3g')
    # Exercise guess transport with a bounded reciprocal grid, as in READ tests.
    route_options = dict(rsgdf_ke_cutoff=12., rsgdf_tail_ke_cutoff=0.) if route == 'gdf' else {}
    result=vq.run_periodic_job(system,basis,method=method,jk_method=route,functional='lda' if method.endswith('KS') else None,cutoff_ha=8,
        kpoints=[1,1,3],initial_guess='FRAGMO',fragments=[[0,1]],
        output=tmp_path,output_qvf=False,max_iter=30,**route_options)
    assert result.converged
    assert result.guess_selection.requested == InitialGuess.FRAGMO
    assert result.guess_selection.effective == InitialGuess.FRAGMO
    assert result.guess_selection.transport == InitialGuess.READ


def test_ecp_sap_potential_matches_analytic_atomic_hartree_reference():
    import vibeqc as vq
    mol,basis,options,context = _ecp_guess_atom(z=30,basis_name='lanl2dz')
    density = _core.sad_density(mol,basis,context)
    grid_options = vq.GridOptions(); grid_options.n_radial=125; grid_options.lebedev_order=35
    grid = vq.build_grid(mol,grid_options)
    ao = vq.evaluate_ao(basis,grid.points)
    rho = np.einsum('pi,ij,pj->p',ao,density,ao)
    vx = -np.cbrt(3*np.maximum(rho,0)/np.pi)
    local_x = ao.T @ ((np.asarray(grid.weights)*vx)[:,None]*ao)
    nuclear = vq.compute_nuclear_with_charges(basis,[a.xyz for a in mol.atoms],context.effective_charges)
    operator = _core.compute_ecp_matrix_from_primitives(basis,np.asarray(context.primitive_centers).ravel(),context.primitive_blocks)
    coulomb = vq.make_direct_jk_builder(basis).build_J(density)
    expected = nuclear + operator + coulomb + local_x
    actual = _core.compute_vsap_ecp(mol,basis,context)
    np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=0)
    # The numerical lattice construction reduces to the same potential when
    # only the home-cell images lie in the integration domain.
    system = vq.PeriodicSystem(3,np.eye(3)*40,list(mol.atoms))
    lo = vq.LatticeSumOptions(); lo.cutoff_bohr=3; lo.nuclear_cutoff_bohr=3
    periodic = _core.compute_vsap_ecp_lattice(basis,system,lo,context)
    np.testing.assert_allclose(periodic.blocks[0],actual,atol=1e-11)


def test_native_read_skew_lattice_self_time_reversal_point():
    import vibeqc as vq
    lattice = np.array([[16.,4.,2.],[0.,17.,3.],[0.,0.,18.]])
    system=vq.PeriodicSystem(3,lattice,[Atom(1,[0,0,0]),Atom(1,[0,0,1.4])])
    basis=BasisSet(system.unit_cell_molecule(),'sto-3g')
    mesh=vq.monkhorst_pack(system,[2,1,1])
    opts=vq.PeriodicSCFOptions(); opts.max_iter=1
    opts.lattice_opts.cutoff_bohr=3; opts.lattice_opts.nuclear_cutoff_bohr=3
    opts.initial_guess=InitialGuess.READ
    opts.read_density_k=[np.eye(2),np.eye(2)]
    assert np.isfinite(vq.run_rhf_periodic(system,basis,mesh,opts).energy)


def test_native_periodic_fragmo_uses_shared_source_and_restores_options():
    import vibeqc as vq
    system=vq.PeriodicSystem(3,np.eye(3)*16,[Atom(1,[0,0,0]),Atom(1,[0,0,1.4])])
    basis=BasisSet(system.unit_cell_molecule(),'sto-3g')
    opts=vq.PeriodicRHFOptions(); opts.initial_guess=InitialGuess.FRAGMO
    opts.lattice_opts.cutoff_bohr=3; opts.lattice_opts.nuclear_cutoff_bohr=3
    result=vq.run_rhf_periodic_gamma(system,basis,opts,fragments=[[0,1]])
    assert result.converged
    assert result.guess_selection.effective==InitialGuess.FRAGMO
    assert opts.initial_guess==InitialGuess.FRAGMO
    assert opts.read_density.size==0


@pytest.mark.parametrize('method',['rhf','rks'])
def test_native_dispatch_consumes_prepared_k_density(method):
    import vibeqc as vq
    system=vq.PeriodicSystem(3,np.eye(3)*16,[Atom(1,[0,0,0]),Atom(1,[0,0,1.4])])
    basis=BasisSet(system.unit_cell_molecule(),'sto-3g')
    mesh=vq.monkhorst_pack(system,[1,1,3])
    opts=vq.PeriodicSCFOptions() if method=='rhf' else vq.PeriodicKSOptions()
    opts.max_iter=1; opts.lattice_opts.cutoff_bohr=3; opts.lattice_opts.nuclear_cutoff_bohr=3
    opts.initial_guess=InitialGuess.HCORE
    source=[np.diag([2.,0.]) for _ in mesh.kpoints]
    run=getattr(vq,'run_'+method+'_periodic_scf')
    prepared=run(system,basis,mesh,opts,initial_density_k=source)
    opts.initial_guess=InitialGuess.READ; opts.read_density_k=source
    direct=getattr(vq,'run_'+method+'_periodic')(system,basis,mesh,opts)
    assert prepared.energy==pytest.approx(direct.energy,abs=1e-12)
    assert prepared.guess_selection.requested==InitialGuess.HCORE
    assert prepared.guess_selection.effective==InitialGuess.READ


def test_native_read_normalizes_weighted_population_without_refilling_each_k():
    import vibeqc as vq
    system=vq.PeriodicSystem(3,np.eye(3)*16,[Atom(1,[0,0,0]),Atom(1,[0,0,1.4])])
    basis=BasisSet(system.unit_cell_molecule(),'sto-3g')
    mesh=vq.monkhorst_pack(system,[1,1,3])
    opts=vq.PeriodicSCFOptions(); opts.initial_guess=InitialGuess.READ; opts.max_iter=1
    opts.lattice_opts.cutoff_bohr=3; opts.lattice_opts.nuclear_cutoff_bohr=3
    blocks=[np.diag([2.,0.]),np.diag([0.,1.]),np.diag([0.,1.])]
    def energy(d):
        opts.read_density_k=d
        return vq.run_rhf_periodic(system,basis,mesh,opts).energy
    actual=energy(blocks)
    assert actual==pytest.approx(energy([np.eye(2)]*3),abs=1e-12)
    assert abs(actual-energy([d*2/np.trace(d) for d in blocks]))>1e-5


def test_periodic_fragmo_refuses_images_on_nonperiodic_axes():
    import vibeqc as vq
    from vibeqc.guess_fragmo import resolve_periodic_fragmo_source
    system=vq.PeriodicSystem(2,np.eye(3)*16,[Atom(2,[0,0,0])])
    basis=BasisSet(system.unit_cell_molecule(),'sto-3g')
    with pytest.raises(ValueError,match='periodic lattice axes'):
        resolve_periodic_fragmo_source(vq.PeriodicRHFOptions(),system,basis,
            vq.monkhorst_pack(system,[1,1,1]),[vq.PeriodicFragment([0],images=[[0,0,1]])])


def test_python_retained_k_restart_skew_lattice_time_reversal():
    import vibeqc as vq
    from vibeqc.guess import periodic_restart_lattice_density
    lattice=np.array([[16.,4.,2.],[0.,17.,3.],[0.,0.,18.]])
    system=vq.PeriodicSystem(3,lattice,[Atom(2,[0,0,0])])
    basis=BasisSet(system.unit_cell_molecule(),'sto-3g')
    mesh=vq.monkhorst_pack(system,[2,1,1])
    lo=vq.LatticeSumOptions(); lo.cutoff_bohr=3
    cells=vq.compute_overlap_lattice(basis,system,lo).cells
    result=periodic_restart_lattice_density([np.eye(1)]*2,[np.eye(1)]*2,
        mesh.weights,1,mesh,cells,retained_k_system=system)
    np.testing.assert_allclose(result.blocks[0],np.eye(1),atol=1e-12)


@pytest.mark.parametrize('guess',[InitialGuess.SAP,InitialGuess.MINAO])
def test_xml_ecp_atomic_reference_preserves_valence_spin_counts(guess):
    from vibeqc.guess import initial_densities_open_shell
    mol,basis,options,context=_ecp_guess_atom(xml=True)
    overlap=_core.compute_overlap(basis)
    a,b=initial_densities_open_shell(mol,basis,10,9,guess,is_periodic=False,
        overlap=overlap,ecp_context=context)
    for density,count in ((a,10),(b,9)):
        assert np.trace(density@overlap)==pytest.approx(count,abs=1e-9)
        assert np.linalg.eigvalsh(density).min()>-1e-10


def test_periodic_minao_zero_electron_reference_is_empty():
    from vibeqc._vibeqc_core import compute_minao_density_periodic
    import vibeqc as vq
    system=vq.PeriodicSystem(3,np.eye(3)*16,[Atom(2,[0,0,0])],charge=2)
    basis=BasisSet(system.unit_cell_molecule(),'sto-3g')
    lo=vq.LatticeSumOptions();lo.cutoff_bohr=3
    density=compute_minao_density_periodic(system.unit_cell_molecule(),basis,system,
        vq.compute_overlap(basis),0,lo)
    np.testing.assert_array_equal(density,np.zeros((basis.nbasis,basis.nbasis)))
