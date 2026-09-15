"""Restricted open-shell Hartree--Fock (ROHF) tests.

Two tiers:

* **Pure-math** --- exercise the Roothaan coupling, the closed-shell
  reduction, the fixed-point property and the SCF stationarity condition
  with *synthetic* integrals (no real basis needed).  These pin the
  algorithm itself.
* **End-to-end** --- ``run_rohf`` on real molecules: ROHF on a singlet
  must equal RHF (real-integral anchor), open-shell determinants are
  spin-pure (``<S^2> = S(S+1)`` exactly) and lie at or above UHF, and
  energies match PySCF's ROHF where available.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, Molecule, UHFOptions, run_rhf, run_uhf
from vibeqc.rohf import (
    ROHFOptions,
    _aufbau_densities,
    _diis_extrapolate,
    _make_hf_fock_builder,
    _orthonormaliser,
    _roothaan_occupations,
    roothaan_effective_fock,
    run_roothaan_scf,
    run_rohf,
)

from .conftest import GEOMETRIES

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


# ---------------------------------------------------------------------------
# Synthetic-integral helpers (shared by the pure-math tier)
# ---------------------------------------------------------------------------


def _synth(nbf, naux, *, gapped=False, seed=20260616):
    """Random SPD overlap, symmetric core, 8-fold-symmetric chemist ERIs.

    ``gapped=True`` -> diagonally dominant core + gently coupled ERIs, so
    the aufbau ground state is unambiguous (well-conditioned SCF).
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((nbf, nbf))
    s = a @ a.T + nbf * np.eye(nbf)
    if gapped:
        h = np.diag(-2.0 * np.arange(nbf, 0, -1, dtype=float))
        p = 0.1 * rng.standard_normal((nbf, nbf))
        h = h + 0.5 * (p + p.T)
        ll = 0.15 * rng.standard_normal((naux, nbf, nbf))
    else:
        h = rng.standard_normal((nbf, nbf))
        h = 0.5 * (h + h.T) - 2.0 * np.eye(nbf)
        ll = rng.standard_normal((naux, nbf, nbf))
    ll = 0.5 * (ll + ll.transpose(0, 2, 1))
    eri = np.einsum("apq,ars->pqrs", ll, ll)  # chemist (pq|rs), full symmetry
    return s, h, eri


def _jk(eri):
    return (
        lambda d: np.einsum("ijkl,kl->ij", eri, d),  # J
        lambda d: np.einsum("ikjl,kl->ij", eri, d),  # K
    )


def _brute_rhf(s, h, eri, nocc, max_iter=2000):
    """Pure-aufbau damped RHF -> an aufbau-self-consistent (E, D, F)."""
    bj, bk = _jk(eri)
    w, v = np.linalg.eigh(s)
    x = v / np.sqrt(w)
    eps, c = np.linalg.eigh(x.T @ h @ x)
    c = x @ c
    d = 2 * c[:, :nocc] @ c[:, :nocc].T
    for _ in range(max_iter):
        f = h + bj(d) - 0.5 * bk(d)
        eps, c = np.linalg.eigh(x.T @ f @ x)
        c = x @ c
        d_new = 2 * c[:, :nocc] @ c[:, :nocc].T
        if np.abs(d_new - d).max() < 1e-12:
            d = d_new
            break
        d = 0.5 * d_new + 0.5 * d
    f = h + bj(d) - 0.5 * bk(d)
    e = np.einsum("ij,ij->", d, h) + 0.5 * np.einsum("ij,ij->", d, bj(d) - 0.5 * bk(d))
    return e, d, f


# ---------------------------------------------------------------------------
# Pure-math tier
# ---------------------------------------------------------------------------


def test_rohf_diis_uses_hermitian_inner_product_for_complex_errors():
    """Complex Bloch residuals use the real Hermitian Pulay metric."""
    focks = [
        np.array([[1.0, 0.2j], [-0.2j, 2.0]]),
        np.array([[1.5, -0.1j], [0.1j, 2.5]]),
    ]
    errors = [
        np.array([[0.0, 1.0 + 2.0j], [-1.0 + 2.0j, 0.0]]),
        np.array([[0.0, -0.5 + 0.25j], [0.5 + 0.25j, 0.0]]),
    ]

    pulay = np.empty((3, 3))
    pulay[-1, :] = pulay[:, -1] = -1.0
    pulay[-1, -1] = 0.0
    for i in range(2):
        for j in range(2):
            pulay[i, j] = np.vdot(errors[i].ravel(), errors[j].ravel()).real
    coefficients = np.linalg.solve(pulay, np.array([0.0, 0.0, -1.0]))
    expected = coefficients[0] * focks[0] + coefficients[1] * focks[1]

    actual = _diis_extrapolate(focks, errors)

    assert np.allclose(actual, expected, atol=1.0e-14)
    assert np.allclose(actual, actual.conj().T, atol=1.0e-14)


def test_roothaan_block_structure():
    """The effective Fock has Coulson's coupling blocks exactly:
    diagonal = Fc, closed-open = Fb, open-virtual = Fa, closed-virtual = Fc."""
    rng = np.random.default_rng(1)
    n, nc, no = 7, 2, 2
    na, nb = nc + no, nc
    s = np.eye(n)
    c = np.linalg.qr(rng.standard_normal((n, n)))[0]  # orthonormal MOs
    dma, dmb = _aufbau_densities(c, na, nb)
    fa = rng.standard_normal((n, n))
    fa = 0.5 * (fa + fa.T)
    fb = rng.standard_normal((n, n))
    fb = 0.5 * (fb + fb.T)
    fc = 0.5 * (fa + fb)

    feff = roothaan_effective_fock(fa, fb, dma, dmb, s)
    m = c.T @ feff @ c
    ma, mb, mc = c.T @ fa @ c, c.T @ fb @ c, c.T @ fc @ c
    cc, oo, vv = slice(0, nc), slice(nc, na), slice(na, n)

    assert np.allclose(m[cc, cc], mc[cc, cc], atol=1e-10)
    assert np.allclose(m[oo, oo], mc[oo, oo], atol=1e-10)
    assert np.allclose(m[vv, vv], mc[vv, vv], atol=1e-10)
    assert np.allclose(m[cc, oo], mb[cc, oo], atol=1e-10)  # closed-open = Fb
    assert np.allclose(m[oo, vv], ma[oo, vv], atol=1e-10)  # open-virtual = Fa
    assert np.allclose(m[cc, vv], mc[cc, vv], atol=1e-10)  # closed-virtual = Fc
    assert np.allclose(m, m.T, atol=1e-10)


def test_rohf_closed_shell_formula_reduction():
    """At a closed-shell density (Da = Db), the ROHF per-spin Fock and the
    ROHF electronic energy collapse exactly onto the RHF Fock + energy."""
    s, h, eri = _synth(6, 5, gapped=True)
    bj, bk = _jk(eri)
    nocc = 2
    fbuild = _make_hf_fock_builder(h, bj, bk)
    e_rhf, d_rhf, f_rhf = _brute_rhf(s, h, eri, nocc)

    da = db = 0.5 * d_rhf
    focka, _fockb, e_elec = fbuild(da, db)
    assert abs(e_elec - e_rhf) < 1e-10
    assert np.abs(focka - f_rhf).max() < 1e-10


def test_rohf_rhf_density_is_fixed_point():
    """One Roothaan step from an aufbau-self-consistent RHF density
    reproduces the RHF effective Fock and density (closed-shell fixed
    point)."""
    s, h, eri = _synth(6, 5, gapped=True)
    bj, bk = _jk(eri)
    nocc = 2
    fbuild = _make_hf_fock_builder(h, bj, bk)
    _e_rhf, d_rhf, f_rhf = _brute_rhf(s, h, eri, nocc)

    da = db = 0.5 * d_rhf
    focka, fockb, _ = fbuild(da, db)
    feff = roothaan_effective_fock(focka, fockb, da, db, s)
    assert np.abs(feff - f_rhf).max() < 1e-9

    x = _orthonormaliser(s, 1e-12)
    _eps, c = np.linalg.eigh(x.T @ feff @ x)
    c = x @ c
    da2, db2 = _aufbau_densities(c, nocc, nocc)
    assert np.abs((da2 + db2) - d_rhf).max() < 1e-8


def test_rohf_open_shell_occupation_uses_alpha_energy():
    """Closed shells follow the effective Fock; open shells are selected by
    alpha orbital energy among the non-core candidates."""
    mo_energy = np.array([-5.0, -1.0, -0.8, -0.7])
    mo_energy_alpha = np.array([-5.0, -1.0, 0.2, -2.0])

    occ = _roothaan_occupations(
        mo_energy, mo_energy_alpha, n_alpha=3, n_beta=2
    )

    assert np.allclose(occ, [2.0, 2.0, 0.0, 1.0])


def test_rohf_scf_stationarity_synthetic():
    """The converged ROHF effective Fock has vanishing inter-shell blocks
    (closed-open, closed-virtual, open-virtual) in the MO basis --- the
    ROHF stationarity condition --- and S^2 is exact."""
    s, h, eri = _synth(8, 6, gapped=True)
    bj, bk = _jk(eri)
    na, nb = 4, 2  # two doubly-occupied + two singly-occupied
    fbuild = _make_hf_fock_builder(h, bj, bk)
    x = _orthonormaliser(s, 1e-10)
    _eps, c = np.linalg.eigh(x.T @ h @ x)
    c = x @ c
    da, db = _aufbau_densities(c, na, nb)

    opts = ROHFOptions(max_iter=400, conv_tol_energy=1e-12, conv_tol_grad=1e-9)
    res = run_roothaan_scf(s, h, 0.0, na, nb, fbuild, opts, da, db)
    assert res.converged

    m = res.mo_coeffs.T @ res.fock @ res.mo_coeffs
    cc, oo, vv = slice(0, nb), slice(nb, na), slice(na, 8)
    off = max(
        np.abs(m[cc, oo]).max(),
        np.abs(m[cc, vv]).max(),
        np.abs(m[oo, vv]).max(),
    )
    assert off < 1e-6

    spin = 0.5 * (na - nb)
    assert res.s_squared == pytest.approx(spin * (spin + 1.0), abs=1e-12)
    assert np.allclose(res.mo_occupations[:nb], 2.0)
    assert np.allclose(res.mo_occupations[nb:na], 1.0)
    assert np.allclose(res.mo_occupations[na:], 0.0)


def test_rohf_energy_weighted_density_matches_fd():
    """The ROHF energy-weighted density W = Da Fa Da + Db Fb Db reproduces
    the nuclear gradient: dE/dλ = Tr(P dH) + ½Tr(P dJ) − ½Σ Tr(Dσ dKσ)
    − Tr(W dS) matches a finite-difference of the converged ROHF energy
    on a synthetic system with analytic integral derivatives. (Pins the
    subtle Lagrangian term; the naive eigenvalue form would fail here.)"""
    from vibeqc.rohf import (
        _aufbau_densities,
        _make_hf_fock_builder,
        _orthonormaliser,
        rohf_energy_weighted_density,
        run_roothaan_scf,
    )

    rng = np.random.default_rng(2026)
    n, naux, na, nb = 7, 6, 4, 2
    def sym(a):
        return 0.5 * (a + a.T)

    A = rng.standard_normal((n, n))
    s0 = A @ A.T + n * np.eye(n)
    s1 = 0.05 * sym(rng.standard_normal((n, n)))
    h0 = np.diag(-2.0 * np.arange(n, 0, -1.0)) + 0.5 * sym(0.1 * rng.standard_normal((n, n)))
    h1 = 0.05 * sym(rng.standard_normal((n, n)))
    l0 = 0.15 * rng.standard_normal((naux, n, n))
    l0 = 0.5 * (l0 + l0.transpose(0, 2, 1))
    l1 = 0.05 * rng.standard_normal((naux, n, n))
    l1 = 0.5 * (l1 + l1.transpose(0, 2, 1))

    def eri_at(lam):
        return np.einsum("apq,ars->pqrs", l0 + lam * l1, l0 + lam * l1)

    def run(lam):
        s, h = s0 + lam * s1, h0 + lam * h1
        e = eri_at(lam)
        def bj(d):
            return np.einsum("ijkl,kl->ij", e, d)

        def bk(d):
            return np.einsum("ikjl,kl->ij", e, d)

        fb = _make_hf_fock_builder(h, bj, bk)
        x = _orthonormaliser(s, 1e-12)
        _w, c = np.linalg.eigh(x.T @ h @ x)
        c = x @ c
        da, db = _aufbau_densities(c, na, nb)
        return run_roothaan_scf(
            s, h, 0.0, na, nb, fb,
            ROHFOptions(max_iter=400, conv_tol_energy=1e-13, conv_tol_grad=1e-10),
            da, db,
        )

    r = run(0.0)
    da, db = r.density_alpha, r.density_beta
    p = da + db
    deri = np.einsum("apq,ars->pqrs", l1, l0) + np.einsum("apq,ars->pqrs", l0, l1)
    dj = np.einsum("ijkl,kl->ij", deri, p)
    dka = np.einsum("ikjl,kl->ij", deri, da)
    dkb = np.einsum("ikjl,kl->ij", deri, db)
    g_hf = (np.einsum("ij,ij->", p, h1)
            + 0.5 * np.einsum("ij,ij->", p, dj)
            - 0.5 * (np.einsum("ij,ij->", da, dka) + np.einsum("ij,ij->", db, dkb)))
    w = rohf_energy_weighted_density(da, db, r.fock_alpha, r.fock_beta)
    g_analytic = g_hf - np.einsum("ij,ij->", w, s1)

    h = 1e-5
    g_fd = (run(+h).energy - run(-h).energy) / (2 * h)
    assert abs(g_analytic - g_fd) < 1e-8


# ---------------------------------------------------------------------------
# End-to-end tier (real integrals; runs on a build box)
# ---------------------------------------------------------------------------


def _tight_rohf_opts() -> ROHFOptions:
    return ROHFOptions(
        max_iter=500,
        conv_tol_energy=1e-12,
        conv_tol_grad=1e-8,
        diis_subspace_size=10,
    )


@pytest.mark.parametrize(
    "mol_key,basis_name",
    [("H2", "sto-3g"), ("H2", "6-31g*"), ("H2O", "sto-3g"), ("H2O", "6-31g*")],
)
def test_rohf_on_closed_shell_matches_rhf(mol_key, basis_name):
    """For closed-shell singlets ROHF must reproduce RHF exactly (no open
    shell -> the Roothaan effective Fock is the closed-shell Fock)."""
    atoms = GEOMETRIES[mol_key]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, basis_name)

    from vibeqc import RHFOptions

    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    rhf_opts.conv_tol_grad = 1e-10
    r_rhf = run_rhf(mol, basis, rhf_opts)
    r_rohf = run_rohf(mol, basis, _tight_rohf_opts())

    assert r_rohf.converged and r_rhf.converged
    assert r_rohf.energy == pytest.approx(r_rhf.energy, abs=1e-8)
    assert abs(r_rohf.s_squared) < 1e-12  # singlet


def test_rohf_ase_calculator_energy_and_forces():
    """The VibeQC ASE calculator runs ROHF (restricted_open=True) with
    analytic forces, and matches the standalone run_rohf energy."""
    pytest.importorskip("ase")
    from ase import Atoms

    import vibeqc as vq
    from vibeqc.ase import VibeQC

    oh = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.97, 0.0, 0.0]])  # Angstrom
    oh.calc = VibeQC(basis="sto-3g", multiplicity=2, restricted_open=True)
    e_ev = oh.get_potential_energy()
    forces = oh.get_forces()
    assert np.isfinite(e_ev)
    assert forces.shape == (2, 3) and np.all(np.isfinite(forces))

    # Energy matches the standalone driver (eV -> Ha).
    from ase.units import Bohr  # noqa: F401  (ensure ase.units importable)

    mol = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.97 / 0.529177210903, 0.0, 0.0])],
        multiplicity=2,
    )
    e_ha = vq.run_rohf(mol, vq.BasisSet(mol, "sto-3g"), _tight_rohf_opts()).energy
    assert e_ev / 27.211386245988 == pytest.approx(e_ha, abs=1e-6)


def test_rohf_h_atom_spin_pure():
    """Single H atom (one electron): <S^2> = 0.75 exactly."""
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    r = run_rohf(mol, basis, _tight_rohf_opts())
    assert r.converged
    assert r.s_squared == pytest.approx(0.75, abs=1e-12)


NH2_DOUBLET_ATOMS = [
    (7, [0.0, 0.0, 0.139456 * ANGSTROM_TO_BOHR]),
    (1, [0.0, 1.443510 * ANGSTROM_TO_BOHR, -0.487243 * ANGSTROM_TO_BOHR]),
    (1, [0.0, -1.443510 * ANGSTROM_TO_BOHR, -0.487243 * ANGSTROM_TO_BOHR]),
]


def test_rohf_nh2_sto3g_default_options_land_in_ground_basin():
    """IID #119 regression: with the DEFAULT knobs
    (scf_accelerator="ediis_diis", damping=0.5, diis_start_iter=1) NH2
    radical ROHF/STO-3G must converge to the ground-state fixed point.

    The pre-fix EDIIS branch coupled the extrapolated Fock pair
    (which is F(D_tilde) with D_tilde = sum_i c_i D_i on the history
    hull) to the Roothaan projector built from the *current* density.
    A damped first step (damping=0.5 mixes the SAD guess into the first
    DIIS update) made the two densities differ enough that the
    mismatched effective Fock jumped the SCF into an excited stationary
    point, -54.5031 Ha (converged=true) instead of the ground state.
    The pinned value is the PySCF 2.14 ROHF/STO-3G reference on the same
    geometry (-54.57191120814234 Ha); the wrong basin sits +68.8 mHa
    above it.  Tolerance is 1e-6 Ha (the same band as the PySCF parity
    test) because the native direct JK build shows a ~1e-8 Ha
    process-to-process noise on this machine.
    """
    mol = Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in NH2_DOUBLET_ATOMS],
        charge=0,
        multiplicity=2,
    )
    basis = BasisSet(mol, "sto-3g")
    # Plain defaults -- not _tight_rohf_opts(): the regression was in the
    # default-path damping/accelerator interplay.
    r = run_rohf(mol, basis, ROHFOptions())
    assert r.converged
    assert r.energy == pytest.approx(-54.5719112, abs=1e-6)
    assert r.s_squared == pytest.approx(0.75, abs=1e-10)
    # Ground-state occupation pattern: 4 closed + 1 open, n_alpha=5/n_beta=4.
    assert np.allclose(
        np.sort(np.asarray(r.mo_occupations))[::-1],
        [2.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0],
    )


OPEN_SHELL_CASES = [
    ("Li-doublet", [(3, [0.0, 0.0, 0.0])], "sto-3g", 0, 2),
    ("OH-doublet", [(8, [0.0, 0.0, 0.0]),
                    (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])], "sto-3g", 0, 2),
    ("O2-triplet", [(8, [0.0, 0.0, -0.6]), (8, [0.0, 0.0, 0.6])], "sto-3g", 0, 3),
    ("NH2-doublet", NH2_DOUBLET_ATOMS, "sto-3g", 0, 2),
]


@pytest.mark.parametrize(
    "label,atoms,basis_name,charge,mult",
    OPEN_SHELL_CASES,
    ids=[c[0] for c in OPEN_SHELL_CASES],
)
def test_rohf_open_shell_spin_pure_and_above_uhf(
    label, atoms, basis_name, charge, mult
):
    """Open-shell ROHF is spin-pure (<S^2> exact) and variationally at or
    above UHF on the same system (UHF has strictly more freedom)."""
    mol = Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in atoms],
        charge=charge,
        multiplicity=mult,
    )
    basis = BasisSet(mol, basis_name)
    r = run_rohf(mol, basis, _tight_rohf_opts())
    assert r.converged, f"{label}: ROHF did not converge in {r.n_iter} iters"

    spin = 0.5 * (mult - 1)
    assert r.s_squared == pytest.approx(spin * (spin + 1.0), abs=1e-10)

    u_opts = UHFOptions()
    u_opts.max_iter = 500
    u_opts.conv_tol_energy = 1e-12
    u_opts.conv_tol_grad = 1e-8
    r_uhf = run_uhf(mol, basis, u_opts)
    if r_uhf.converged:
        # UHF <= ROHF (variational); allow a tiny numerical slack.
        assert r_uhf.energy <= r.energy + 1e-7


def test_rohf_orbital_table_single_column():
    """ROHF/ROKS get a dedicated single-column orbital table with
    closed/open/virtual occupations + SOMO markers, not the UHF
    two-column layout with identical alpha/beta energies."""
    from vibeqc.output.formats.scf_log import _format_orbital_table_rohf

    class _Mock:
        method = "rohf"
        mo_energies = [-20.1, -1.3, -0.7, -0.6, -0.55, 0.2, 0.8]
        mo_occupations = [2, 2, 2, 2, 1, 0, 0]  # n_alpha=5, n_beta=4

    out = _format_orbital_table_rohf(_Mock(), n_alpha=5, n_beta=4, n_virtual=2)
    assert "restricted open-shell" in out
    assert "SOMO" in out and "LUMO" in out
    # the singly-occupied orbital shows occ 1.0; closed shells show 2.0
    assert "1.0" in out and out.count("2.0") >= 4


def test_rohf_orbital_table_emits_homo_lumo_values():
    """The ROHF orbital table must include explicit HOMO, LUMO, and gap
    in its footer line (not just markers on the rows)."""
    from vibeqc.output.formats.scf_log import _format_orbital_table_rohf

    class _Mock:
        method = "rohf"
        mo_energies = [-2.0, -1.5, -0.4, 0.1, 0.5]
        mo_occupations = [2, 2, 1, 0, 0]  # HOMO at idx 2 (-0.4), LUMO at idx 3 (0.1)

    out = _format_orbital_table_rohf(_Mock(), n_alpha=3, n_beta=2, n_virtual=2)
    assert "HOMO:" in out
    assert "LUMO:" in out
    assert "gap:" in out


def test_rohf_orbital_table_rhf_emits_homo_lumo_values():
    """The RHF orbital table must also include explicit HOMO, LUMO, gap."""
    from vibeqc.output.formats.scf_log import _format_orbital_table_rhf

    class _Mock:
        mo_energies = [-5.0, -0.5, 0.3, 0.8]

    out = _format_orbital_table_rhf(_Mock(), n_occ=2, n_virtual=2)
    assert "HOMO:" in out
    assert "LUMO:" in out
    assert "gap:" in out


def test_guest_saunders_canonicalize_density_invariant():
    """Guest-Saunders canonicalisation preserves the total density and
    per-spin densities (subspace rotations leave occupations unchanged)."""
    from vibeqc.rohf import guest_saunders_canonicalize

    rng = np.random.default_rng(42)
    nbf = 8
    n_alpha, n_beta = 5, 3
    # Random orthonormal MO coefficients.
    C = np.linalg.qr(rng.standard_normal((nbf, nbf)))[0]
    # Random symmetric Fock matrices.
    Fa = rng.standard_normal((nbf, nbf))
    Fa = 0.5 * (Fa + Fa.T)
    Fb = rng.standard_normal((nbf, nbf))
    Fb = 0.5 * (Fb + Fb.T)

    eps_gs, C_gs = guest_saunders_canonicalize(C, Fa, Fb, n_alpha, n_beta)

    # Density from original MOs (aufbau: first n_beta=3 doubly occ,
    # next n_alpha-n_beta=2 singly occ = alpha only).
    Da = C[:, :n_alpha] @ C[:, :n_alpha].T
    Db = C[:, :n_beta] @ C[:, :n_beta].T
    # Density from canonical MOs (same occupation pattern).
    Da_gs = C_gs[:, :n_alpha] @ C_gs[:, :n_alpha].T
    Db_gs = C_gs[:, :n_beta] @ C_gs[:, :n_beta].T

    assert np.allclose(Da, Da_gs, atol=1e-12)
    assert np.allclose(Db, Db_gs, atol=1e-12)
    assert np.allclose(Da + Db, Da_gs + Db_gs, atol=1e-12)


def test_guest_saunders_eigenvalues_distinct_from_roothaan():
    """Guest-Saunders eigenvalues must differ from the Roothaan effective
    Fock eigenvalues for a genuine open-shell case."""
    from vibeqc.rohf import guest_saunders_canonicalize, roothaan_effective_fock

    rng = np.random.default_rng(123)
    nbf = 6
    n_alpha, n_beta = 4, 2
    C = np.linalg.qr(rng.standard_normal((nbf, nbf)))[0]
    Fa = rng.standard_normal((nbf, nbf))
    Fa = 0.5 * (Fa + Fa.T)
    Fb = rng.standard_normal((nbf, nbf))
    Fb = 0.5 * (Fb + Fb.T)

    # Aufbau densities.
    Da = C[:, :n_alpha] @ C[:, :n_alpha].T
    Db = C[:, :n_beta] @ C[:, :n_beta].T
    S = np.eye(nbf)

    # Roothaan eigenvalues.
    Feff = roothaan_effective_fock(Fa, Fb, Da, Db, S)
    eps_roothaan = np.linalg.eigvalsh(C.T @ Feff @ C)

    # Guest-Saunders eigenvalues.
    eps_gs, _ = guest_saunders_canonicalize(C, Fa, Fb, n_alpha, n_beta)

    # They should differ for at least one eigenvalue (the open-shell ones).
    # In practice the open-shell and virtual blocks differ.
    assert not np.allclose(eps_roothaan, eps_gs, atol=1e-6), (
        "Guest-Saunders eigenvalues should differ from Roothaan "
        "for an open-shell system"
    )


def test_rohf_result_records_canonicalization():
    """An end-to-end ROHF run must record the canonicalisation convention
    on the result and preserve the Roothaan eigenvalues."""
    mol = Molecule([Atom(8, [0.0, 0.0, 0.0])], multiplicity=3)
    basis = BasisSet(mol, "sto-3g")
    opts = _tight_rohf_opts()
    r = run_rohf(mol, basis, opts)
    assert r.converged
    assert r.rohf_canonicalization == "guest-saunders"
    assert r.mo_energies_roothaan is not None
    assert r.mo_coeffs_roothaan is not None
    assert len(r.mo_energies) == len(r.mo_energies_roothaan)


def test_rohf_homo_lumo_accessible():
    """The HOMO and LUMO orbital energies must be extractable from the
    result as the last occupied and first virtual eigenvalues."""
    mol = Molecule(
        [Atom(8, [0.0, 0.0, -0.6]), Atom(8, [0.0, 0.0, 0.6])],
        multiplicity=3,
    )
    basis = BasisSet(mol, "sto-3g")
    opts = _tight_rohf_opts()
    r = run_rohf(mol, basis, opts)
    assert r.converged

    # HOMO = highest occupied (index n_alpha - 1).
    assert r.n_alpha < len(r.mo_energies), "need virtual orbitals for LUMO"
    homo = float(r.mo_energies[r.n_alpha - 1])
    # LUMO = first virtual (index n_alpha).
    lumo = float(r.mo_energies[r.n_alpha])
    assert homo < 0.0  # bound state
    assert lumo > homo  # sensible gap


def test_rohf_energy_unchanged_by_canonicalization():
    """Guest-Saunders canonicalisation must not change the total energy —
    only the eigenvalues and MO shapes differ."""
    from vibeqc.rohf import guest_saunders_canonicalize

    mol = Molecule([Atom(8, [0.0, 0.0, 0.0])], multiplicity=3)
    basis = BasisSet(mol, "sto-3g")
    opts = _tight_rohf_opts()
    r = run_rohf(mol, basis, opts)
    assert r.converged

    # The energy is invariant — the density is unchanged by subspace rotations.
    # We verify that the Roothaan MO coefficients produce the same density
    # (and hence the same energy) as the canonical ones.
    C_roothaan = np.asarray(r.mo_coeffs_roothaan)
    C_gs = np.asarray(r.mo_coeffs)
    n_a, n_b = r.n_alpha, r.n_beta

    Da_rooth = C_roothaan[:, :n_a] @ C_roothaan[:, :n_a].T
    Db_rooth = C_roothaan[:, :n_b] @ C_roothaan[:, :n_b].T
    Da_gs = C_gs[:, :n_a] @ C_gs[:, :n_a].T
    Db_gs = C_gs[:, :n_b] @ C_gs[:, :n_b].T

    assert np.allclose(Da_rooth, Da_gs, atol=1e-10)
    assert np.allclose(Db_rooth, Db_gs, atol=1e-10)


def test_rohf_run_job_records_canonicalization_in_system():
    """Full run_job ROHF must record rohf_canonicalization in the
    .system manifest so downstream tools know the convention."""
    import tempfile, os
    import vibeqc as vq

    mol = Molecule(
        [Atom(8, [0.0, 0.0, -0.6]), Atom(8, [0.0, 0.0, 0.6])],
        multiplicity=3,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "test")
        result = vq.run_job(
            method="rohf",
            basis="sto-3g",
            molecule=mol,
            output=out,
            write_molden_file=False,
            write_population_file=False,
            write_xyz_file=False,
        )
        assert result.converged
        # .system must contain the canonicalization field.
        system_path = out + ".system"
        assert os.path.exists(system_path)
        with open(system_path) as f:
            content = f.read()
        assert "rohf_canonicalization" in content
        assert '"guest-saunders"' in content


def test_rohf_atomization_spin_pure_references(tmp_path):
    """ROHF atomization uses spin-pure free-atom references (open-shell O,
    H computed with ROHF) — the energy-only path reduces to run_rohf."""
    import vibeqc as vq
    from vibeqc.atomization import atomic_ground_state_energy

    # Free O atom (Z=8, triplet) via ROHF: spin-pure reference.
    e_o = atomic_ground_state_energy(8, "rohf", "sto-3g")
    assert e_o < 0.0  # bound atom

    # End-to-end: run_job(method="rohf", atomization=True) writes the block.
    h2o = vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [1.81, 0.0, 0.0]),
            vq.Atom(1, [-0.45, 1.75, 0.0]),
        ],
    )
    res = vq.run_job(
        h2o,
        basis="sto-3g",
        method="rohf",
        atomization=True,
        output=str(tmp_path / "h2o_rohf_atm"),
        write_molden_file=False,
        citations=False,
    )
    assert res is not None
    out = (tmp_path / "h2o_rohf_atm.out").read_text()
    assert "Atomization energy" in out


def test_roks_atomization_reference():
    """ROKS free-atom reference (open-shell C, PBE) runs and is bound."""
    from vibeqc.atomization import atomic_ground_state_energy

    e_c = atomic_ground_state_energy(6, "roks", "sto-3g", functional="pbe")
    assert e_c < 0.0


def test_rohf_cas_reference_orbitals_orthonormal():
    """The ROHF CAS-reference orbital provider returns S-orthonormal MOs."""
    from vibeqc import compute_overlap
    from vibeqc.solvers._hamiltonian import get_hf_orbital_provider

    mol = Molecule(
        [Atom(8, [0.0, 0.0, 0.0]),
         Atom(1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
        multiplicity=2,
    )
    basis = BasisSet(mol, "sto-3g")
    C = get_hf_orbital_provider(mol, basis, method="rohf")
    S = np.asarray(compute_overlap(basis))
    n = S.shape[0]
    assert C.shape == (n, n)
    # C^T S C = I  (one orthonormal spin-restricted set)
    assert np.allclose(C.T @ S @ C, np.eye(n), atol=1e-8)


def test_rohf_cas_reference_runs_in_runjob(tmp_path):
    """cas_reference='rohf' gives a spin-pure CAS reference end-to-end."""
    import vibeqc as vq

    li = vq.Molecule([vq.Atom(3, [0.0, 0.0, 0.0])], multiplicity=2)
    res = vq.run_job(
        li,
        basis="sto-3g",
        method="casci",
        active_space=(2, 1),  # 2 orbitals, 1 active electron
        cas_reference="rohf",
        output=str(tmp_path / "li_casci_rohf"),
        write_molden_file=False,
        citations=False,
    )
    assert res is not None


def test_rohf_invalid_cas_reference_rejected(tmp_path):
    li = Molecule([Atom(3, [0.0, 0.0, 0.0])], multiplicity=2)
    import vibeqc as vq

    with pytest.raises(ValueError, match="cas_reference"):
        vq.run_job(li, basis="sto-3g", method="casci", active_space=(2, 1),
                   cas_reference="bogus",
                   output=str(tmp_path / "invalid_cas_reference"),
                   citations=False)


def test_rohf_geometry_optimization_runs(tmp_path):
    """ROHF geometry optimisation via finite-difference forces (the
    wavefunction-solver FD path) lowers the energy and completes."""
    import vibeqc as vq

    oh = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [1.9, 0.0, 0.0])],
        multiplicity=2,
    )
    out = tmp_path / "oh_rohf_opt"
    res = vq.run_job(
        oh,
        basis="sto-3g",
        method="rohf",
        optimize=True,
        max_opt_steps=5,
        output=str(out),
        write_molden_file=False,
        citations=False,
    )
    assert res is not None
    assert (tmp_path / "oh_rohf_opt.out").exists()


def test_rohf_analytic_gradient_matches_fd():
    """compute_rohf_gradient matches a finite-difference of the run_rohf
    energy on a real open-shell doublet (OH / STO-3G)."""
    import vibeqc as vq
    from vibeqc import compute_rohf_gradient

    def energy(dx):
        mol = vq.Molecule(
            [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [1.83 + dx, 0.0, 0.0])],
            multiplicity=2,
        )
        return vq.run_rohf(mol, vq.BasisSet(mol, "sto-3g"), _tight_rohf_opts()).energy

    mol = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [1.83, 0.0, 0.0])],
        multiplicity=2,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    g = compute_rohf_gradient(mol, basis, vq.run_rohf(mol, basis, _tight_rohf_opts()))
    h = 1e-4
    g_fd = (energy(h) - energy(-h)) / (2 * h)  # dE/d(H x-position) = g[1, 0]
    assert abs(float(g[1, 0]) - g_fd) < 1e-5


def test_rohf_density_fitted_gradient_matches_fd():
    """The spin-resolved DF J/K assembly differentiates the same ROHF PES."""
    import vibeqc as vq
    from vibeqc import compute_rohf_gradient

    aux_name = "def2-svp-jk"
    scf_opts = _tight_rohf_opts()
    scf_opts.density_fit = True
    scf_opts.aux_basis = aux_name
    grad_opts = vq.GradientOptions()
    grad_opts.density_fit = True
    grad_opts.aux_basis = aux_name

    def calculate(dx):
        mol = vq.Molecule(
            [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [1.83 + dx, 0.0, 0.0])],
            multiplicity=2,
        )
        basis = vq.BasisSet(mol, "def2-svp")
        result = vq.run_rohf(mol, basis, scf_opts)
        assert result.converged
        return mol, basis, result

    mol, basis, result = calculate(0.0)
    gradient = compute_rohf_gradient(
        mol, basis, result, gradient_options=grad_opts
    )
    h = 1e-4
    energy_plus = calculate(h)[2].energy
    energy_minus = calculate(-h)[2].energy
    gradient_fd = (energy_plus - energy_minus) / (2.0 * h)
    assert float(gradient[1, 0]) == pytest.approx(gradient_fd, abs=1e-5)


def test_rohf_rijcosx_gradient_fails_closed():
    import vibeqc as vq
    from vibeqc import compute_rohf_gradient

    mol = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [1.83, 0.0, 0.0])],
        multiplicity=2,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    result = vq.run_rohf(mol, basis, _tight_rohf_opts())
    grad_opts = vq.GradientOptions()
    grad_opts.density_fit = True
    grad_opts.aux_basis = "def2-svp-jk"
    grad_opts.cosx = True
    with pytest.raises(NotImplementedError, match="RIJCOSX"):
        compute_rohf_gradient(
            mol, basis, result, gradient_options=grad_opts
        )


def test_rohf_frequencies_run(tmp_path):
    """ROHF Hessian/frequencies work (FD of the analytic ROHF gradient)."""
    import vibeqc as vq

    oh = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [1.83, 0.0, 0.0])],
        multiplicity=2,
    )
    vq.run_job(
        oh,
        basis="sto-3g",
        method="rohf",
        hessian=True,
        output=str(tmp_path / "oh_rohf_hess"),
        write_molden_file=False,
        citations=False,
    )
    out = (tmp_path / "oh_rohf_hess.out").read_text()
    assert "Vibrational Frequencies" in out


def test_roks_hessian_is_gated(tmp_path):
    """ROKS Hessian is still gated (its XC gradient is not analytic yet)."""
    import vibeqc as vq

    oh = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [1.83, 0.0, 0.0])],
        multiplicity=2,
    )
    with pytest.raises(NotImplementedError, match="M5b"):
        vq.run_job(
            oh,
            basis="sto-3g",
            method="roks",
            functional="pbe",
            hessian=True,
            output=str(tmp_path / "oh_roks_hess"),
            citations=False,
        )


# ---------------------------------------------------------------------------
# initial_guess normalisation (regression: the field used to be fail-open)
# ---------------------------------------------------------------------------
#
# ``ROHFOptions.initial_guess`` is a plain Python field rather than the
# pybind11-typed slot RHF/UHF/RKS/UKS carry. Before the fix it was only ever
# consulted as ``isinstance(guess, str) and guess.lower() == "core"``, so
# assigning the ``InitialGuess`` enum -- or any string other than "core" --
# was silently accepted and ignored: the driver used SAD regardless. A user
# who pinned a guess got a different one with no diagnostic.


def _guess_probe_system():
    """OH doublet + the matrices ``_initial_densities`` needs."""
    from vibeqc import _vibeqc_core as core

    mol = Molecule(
        [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.8])],
        multiplicity=2,
    )
    basis = BasisSet(mol, "sto-3g")
    s = np.asarray(core.compute_overlap(basis), dtype=float)
    hcore = np.asarray(core.compute_kinetic(basis), dtype=float) + np.asarray(
        core.compute_nuclear(basis, mol), dtype=float
    )
    x = _orthonormaliser(s, 1e-8)
    return mol, basis, s, hcore, x


def _density_for_guess(guess, options_cls=ROHFOptions):
    """Alpha initial density the driver actually builds for ``guess``."""
    from vibeqc.rohf import _initial_densities

    mol, basis, s, hcore, x = _guess_probe_system()
    opts = options_cls()
    opts.initial_guess = guess  # assignment after construction, as a user would
    d_alpha, _d_beta = _initial_densities(mol, basis, 5, 4, opts, s, hcore, x)
    return np.asarray(d_alpha, dtype=float)


def test_rohf_initial_guess_enum_assignment_takes_effect():
    """Assigning the ``InitialGuess`` enum must change the guess actually used.

    This is the regression: pre-fix, ``initial_guess = InitialGuess.HCORE``
    produced the SAD density (the enum was silently ignored because the
    consumption site only recognised the string "core").
    """
    from vibeqc import _vibeqc_core as core

    d_sad = _density_for_guess("sad")
    d_core_str = _density_for_guess("core")
    d_core_enum = _density_for_guess(core.InitialGuess.HCORE)

    # Guard: the two guesses are genuinely distinguishable on this system,
    # so the assertion below cannot pass vacuously.
    assert not np.allclose(d_sad, d_core_str, atol=1e-8)

    # The enum must land on the same density as its canonical string, and
    # must NOT silently fall back to SAD.
    assert np.allclose(d_core_enum, d_core_str, atol=1e-12)
    assert not np.allclose(d_core_enum, d_sad, atol=1e-8)


def test_rohf_initial_guess_accepts_enum_and_canonical_strings():
    """Both spellings normalise to the same effective ``InitialGuess``."""
    from vibeqc import _vibeqc_core as core
    from vibeqc.rohf import _resolve_initial_guess

    cases = [
        ("sad", core.InitialGuess.SAD),
        ("SAD", core.InitialGuess.SAD),
        (core.InitialGuess.SAD, core.InitialGuess.SAD),
        ("core", core.InitialGuess.HCORE),  # documented alias
        ("hcore", core.InitialGuess.HCORE),
        (core.InitialGuess.HCORE, core.InitialGuess.HCORE),
        ("auto", core.InitialGuess.AUTO),
        (core.InitialGuess.AUTO, core.InitialGuess.AUTO),
    ]
    for value, expected in cases:
        opts = ROHFOptions()
        opts.initial_guess = value
        assert _resolve_initial_guess(opts) is expected, f"{value!r}"

    # The selector remains AUTO until the system-aware execution boundary.
    assert _resolve_initial_guess(ROHFOptions()) is core.InitialGuess.AUTO


@pytest.mark.parametrize("bad", ["bogus-guess", "", None, 42, 3.5])
def test_rohf_initial_guess_rejects_unknown_values(bad):
    """An unrecognised guess raises instead of silently reverting to SAD."""
    with pytest.raises(ValueError, match="unknown initial_guess"):
        _density_for_guess(bad)


@pytest.mark.parametrize("guess", ["sap", "patom", "hueckel", "minao"])
def test_rohf_overlap_dependent_guesses_execute(guess):
    """The native builder receives the molecular overlap and field (#685)."""
    density = _density_for_guess(guess)
    _, _, overlap, _, _ = _guess_probe_system()
    assert np.trace(density @ overlap) == pytest.approx(5.0, abs=1e-10)
    assert not np.allclose(density, _density_for_guess("hcore"))


def test_roks_initial_guess_normalisation_matches_rohf():
    """``ROKSOptions`` inherits the field, so it must inherit the fix."""
    from vibeqc import _vibeqc_core as core
    from vibeqc.rohf import _resolve_initial_guess
    from vibeqc.roks import ROKSOptions

    opts = ROKSOptions()
    opts.initial_guess = core.InitialGuess.HCORE
    assert _resolve_initial_guess(opts) is core.InitialGuess.HCORE

    d_enum = _density_for_guess(core.InitialGuess.HCORE, options_cls=ROKSOptions)
    d_str = _density_for_guess("core", options_cls=ROKSOptions)
    d_sad = _density_for_guess("sad", options_cls=ROKSOptions)
    assert np.allclose(d_enum, d_str, atol=1e-12)
    assert not np.allclose(d_enum, d_sad, atol=1e-8)

    bad = ROKSOptions()
    bad.initial_guess = "bogus-guess"
    with pytest.raises(ValueError, match="unknown initial_guess"):
        _resolve_initial_guess(bad)


def test_rohf_converges_from_either_guess_spelling():
    """End-to-end: the pinned guess is honoured and both reach the same SCF."""
    from vibeqc import _vibeqc_core as core

    mol = Molecule(
        [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.8])],
        multiplicity=2,
    )
    basis = BasisSet(mol, "sto-3g")

    opts_enum = _tight_rohf_opts()
    opts_enum.initial_guess = core.InitialGuess.HCORE
    r_enum = run_rohf(mol, basis, opts_enum)

    opts_str = _tight_rohf_opts()
    opts_str.initial_guess = "core"
    r_str = run_rohf(mol, basis, opts_str)

    assert r_enum.converged and r_str.converged
    assert r_enum.energy == pytest.approx(r_str.energy, abs=1e-9)


@pytest.mark.parametrize(
    "label,atoms,basis_name,charge,mult",
    OPEN_SHELL_CASES,
    ids=[c[0] for c in OPEN_SHELL_CASES],
)
def test_rohf_energy_matches_pyscf(label, atoms, basis_name, charge, mult):
    """Energy + <S^2> parity against PySCF's ROHF where PySCF is present."""
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    m = gto.Mole()
    m.unit = "Bohr"
    m.atom = [[Z, tuple(xyz)] for Z, xyz in atoms]
    m.basis = basis_name
    m.charge = charge
    m.spin = mult - 1  # 2S
    m.verbose = 0
    m.build()
    mf = scf.ROHF(m)
    mf.conv_tol = 1e-12
    mf.kernel()
    ref_e = mf.e_tot

    mol = Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in atoms],
        charge=charge,
        multiplicity=mult,
    )
    basis = BasisSet(mol, basis_name)
    r = run_rohf(mol, basis, _tight_rohf_opts())
    assert r.converged
    assert abs(r.energy - ref_e) < 1e-6, (
        f"{label}: E_vibeqc={r.energy:.10f} E_pyscf={ref_e:.10f}"
    )


# ---------------------------------------------------------------------------
# Issue #119 -- the DEFAULT accelerator returned a converged, unwarned
# wrong answer on the near-equilibrium NH2 radical
# ---------------------------------------------------------------------------

# NH2 radical at r(N-H) = 1.024 A, angle(H-N-H) = 103.4 deg -- the geometry
# the mf003 / ml003s reference rows use. ORCA ROHF/STO-3G on the filed
# geometry: -54.834411062624 Ha (issue #119); the value pinned below is
# vibe-qc's own on this Cartesian realisation of the same internals, which
# agrees with it to 8.5e-05 Ha (the geometry is not bit-identical to the
# archived deck).
_NH2_R_ANGSTROM = 1.024
_NH2_ANGLE_DEG = 103.4
_NH2_HALF = math.radians(0.5 * _NH2_ANGLE_DEG)
NH2_EQUILIBRIUM_ATOMS = [
    (7, [0.0, 0.0, 0.0]),
    (1, [0.0,
         _NH2_R_ANGSTROM * math.sin(_NH2_HALF) * ANGSTROM_TO_BOHR,
         _NH2_R_ANGSTROM * math.cos(_NH2_HALF) * ANGSTROM_TO_BOHR]),
    (1, [0.0,
         -_NH2_R_ANGSTROM * math.sin(_NH2_HALF) * ANGSTROM_TO_BOHR,
         _NH2_R_ANGSTROM * math.cos(_NH2_HALF) * ANGSTROM_TO_BOHR]),
]
NH2_EQUILIBRIUM_ROHF_STO3G = -54.8344963877

# Every accelerator this driver offers. The cross-route agreement below is
# the invariant that was missing when #119 was filed: the accelerator
# changes the PATH through the SCF manifold, never the fixed point, so a
# route that disagrees with the others is a defect by construction and is
# cheap to detect.
_ALL_ROHF_ACCELERATORS = (
    "ediis_diis",  # the default
    "diis",
    "ediis",
    "adiis",
    "kdiis",
    "r_cdiis",
    "ad_cdiis",
)


def _nh2_equilibrium():
    mol = Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in NH2_EQUILIBRIUM_ATOMS],
        charge=0,
        multiplicity=2,
    )
    return mol, BasisSet(mol, "sto-3g")


def test_rohf_nh2_equilibrium_default_route_reaches_the_ground_basin():
    """Issue #119: plain defaults must not converge to the excited state.

    The SAD start this driver takes is *density mode* -- no JKBuilder --
    and ``GuessEngine::build_open_shell`` returns the raw Hund-split atomic
    densities there, whose spin traces are the sum of the free-atom spin
    populations: 7 alpha / 2 beta for NH2, i.e. S_z = 5/2 instead of the
    doublet's 1/2. UHF re-diagonalises per spin and refills by aufbau, so
    it shrugs that off; the Roothaan loop builds its closed/open/virtual
    projectors from the density itself and hands the same iterate to the
    accelerator history, where EDIIS keeps it as a hull vertex carrying an
    energy 0.83 Ha BELOW the true ground state (a non-N-representable
    density is not bounded by the variational minimum).

    Pre-fix the default ``ediis_diis`` route converged -- ``converged =
    True``, zero warnings -- to the excited 2A1 stationary point at
    -54.7393101912 Ha, +95.1 mHa (about 60 kcal/mol) above the 2B1 ground
    state, while ``diis`` / no-DIIS / level-shift all found the ground
    state. See :func:`vibeqc.rohf._renormalise_guess_spin_density`.
    """
    mol, basis = _nh2_equilibrium()
    r = run_rohf(mol, basis, ROHFOptions())  # plain defaults, as filed
    assert r.converged
    assert r.energy == pytest.approx(NH2_EQUILIBRIUM_ROHF_STO3G, abs=1e-7)
    assert r.s_squared == pytest.approx(0.75, abs=1e-10)
    # 4 closed + 1 open; the excited basin has the same occupation PATTERN,
    # so the energy above is what separates them.
    assert np.allclose(
        np.sort(np.asarray(r.mo_occupations))[::-1],
        [2.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0],
    )


@pytest.mark.parametrize("accelerator", _ALL_ROHF_ACCELERATORS)
def test_rohf_nh2_equilibrium_every_accelerator_agrees(accelerator):
    """Issue #119: the SCF accelerator must not change the fixed point.

    Four routes (``diis``, no-DIIS, level-shift, and the C++ DIIS object
    behind ``r_cdiis``) already agreed on -54.8344963877 Ha while the
    default disagreed by +95 mHa. That disagreement was itself a detectable
    signal that nothing checked, which is why it is asserted here rather
    than only pinning the default.
    """
    mol, basis = _nh2_equilibrium()
    opts = ROHFOptions()
    opts.scf_accelerator = accelerator
    r = run_rohf(mol, basis, opts)
    assert r.converged
    assert r.energy == pytest.approx(NH2_EQUILIBRIUM_ROHF_STO3G, abs=1e-7)
    assert r.s_squared == pytest.approx(0.75, abs=1e-10)


@pytest.mark.parametrize(
    "label,accelerator_free_options",
    [
        ("no-DIIS", {"use_diis": False, "damping": 0.7, "max_iter": 300}),
        ("level-shift", {"level_shift": 0.5}),
    ],
)
def test_rohf_nh2_equilibrium_accelerator_free_routes_agree(
    label, accelerator_free_options
):
    """The two routes that bypass extrapolation entirely (#119)."""
    mol, basis = _nh2_equilibrium()
    opts = ROHFOptions()
    for key, value in accelerator_free_options.items():
        setattr(opts, key, value)
    r = run_rohf(mol, basis, opts)
    assert r.converged
    assert r.energy == pytest.approx(NH2_EQUILIBRIUM_ROHF_STO3G, abs=1e-7)


@pytest.mark.parametrize(
    "label,atoms,multiplicity,basis_name,n_alpha,n_beta",
    [
        ("Li-doublet", [(3, [0.0, 0.0, 0.0])], 2, "sto-3g", 2, 1),
        ("N-quartet", [(7, [0.0, 0.0, 0.0])], 4, "sto-3g", 5, 2),
        ("OH-doublet", [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 1.8331])],
         2, "sto-3g", 5, 4),
        ("OH-doublet-svp", [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 1.8331])],
         2, "def2-svp", 5, 4),
        ("NH2-doublet", NH2_EQUILIBRIUM_ATOMS, 2, "sto-3g", 5, 4),
        ("O2-triplet", [(8, [0.0, 0.0, -1.14]), (8, [0.0, 0.0, 1.14])],
         3, "sto-3g", 9, 7),
        ("CH3-doublet",
         [(6, [0.0, 0.0, 0.0]), (1, [2.04, 0.0, 0.0]),
          (1, [-1.02, 1.77, 0.0]), (1, [-1.02, -1.77, 0.0])],
         2, "sto-3g", 5, 4),
        ("NO-doublet", [(7, [0.0, 0.0, 0.0]), (8, [0.0, 0.0, 2.17])],
         2, "sto-3g", 8, 7),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_rohf_initial_guess_carries_the_target_spin_electron_counts(
    label, atoms, multiplicity, basis_name, n_alpha, n_beta
):
    """The guess handed to the Roothaan loop must be a (n_alpha, n_beta) state.

    Root cause of issue #119. Measured before the fix (want -> got):
    NH2 5/4 -> 7/2, CH3 5/4 -> 7/2, OH 5/4 -> 6/3, O2 9/7 -> 10/6 --
    the free-atom spin populations summed with no projection onto the
    molecule's spin partition. Atoms were already correct, which is why
    the single-atom cases here are guards against a vacuous parametrisation
    rather than regressions.
    """
    from vibeqc import _vibeqc_core as core
    from vibeqc.rohf import _initial_densities

    mol = Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in atoms],
        charge=0,
        multiplicity=multiplicity,
    )
    basis = BasisSet(mol, basis_name)
    s = np.asarray(core.compute_overlap(basis), dtype=float)
    hcore = np.asarray(core.compute_kinetic(basis), dtype=float) + np.asarray(
        core.compute_nuclear(basis, mol), dtype=float
    )
    x = _orthonormaliser(s, 1e-7)
    d_alpha, d_beta = _initial_densities(
        mol, basis, n_alpha, n_beta, ROHFOptions(), s, hcore, x
    )
    assert float(np.trace(s @ d_alpha)) == pytest.approx(n_alpha, abs=1e-8)
    assert float(np.trace(s @ d_beta)) == pytest.approx(n_beta, abs=1e-8)


# ---------------------------------------------------------------------------
# GitLab #487 -- the Roothaan coupling must use the accelerator's own density
# ---------------------------------------------------------------------------


def _install_roothaan_coupling_spy(monkeypatch):
    """Instrument ``run_roothaan_scf`` for the #487 checks.

    Records, for every cycle whose effective Fock was built from an
    EXTRAPOLATED (F_a, F_b) pair, how far that pair is from the HF Fock of
    the density it was coupled with.  At the HF level
    Σ_i c_i F(D_i) == F(Σ_i c_i D_i) exactly, so a consistent coupling
    leaves round-off; coupling the extrapolated pair with any other density
    (the pre-#487 fallback used the current iterate) leaves a residual of
    the size of the extrapolation step.  Also captures the accelerator
    objects the loop constructs, so a test can ask whether the anti-replay
    guard fired on the trajectory.

    The extrapolated call is the SECOND ``roothaan_effective_fock`` call of
    a cycle (the first is the raw pair); the ``fock_builder`` handed to
    ``run_roothaan_scf`` is captured to rebuild F(D_coupled).
    """
    import vibeqc.rohf as rohf_mod

    state = {"builder": None, "calls_this_cycle": 0, "residuals": [], "accel": []}

    real_run = rohf_mod.run_roothaan_scf

    def run_spy(s, hcore, e_nuc, n_alpha, n_beta, fock_builder, options,
                init_alpha, init_beta, **kw):
        def builder_spy(da, db):
            state["calls_this_cycle"] = 0
            return fock_builder(da, db)

        state["builder"] = fock_builder
        return real_run(s, hcore, e_nuc, n_alpha, n_beta, builder_spy,
                        options, init_alpha, init_beta, **kw)

    monkeypatch.setattr(rohf_mod, "run_roothaan_scf", run_spy)

    real_reff = rohf_mod.roothaan_effective_fock

    def reff_spy(fa, fb, da, db, s_):
        state["calls_this_cycle"] += 1
        if state["calls_this_cycle"] == 2:
            fa_chk, fb_chk, _ = state["builder"](da, db)
            state["residuals"].append(
                max(np.abs(fa - fa_chk).max(), np.abs(fb - fb_chk).max())
            )
        return real_reff(fa, fb, da, db, s_)

    monkeypatch.setattr(rohf_mod, "roothaan_effective_fock", reff_spy)

    for name in ("_EDIIS", "_ADIIS"):
        cls = getattr(rohf_mod, name)

        def factory(n, _cls=cls):
            obj = _cls(n)
            state["accel"].append(obj)
            return obj

        monkeypatch.setattr(rohf_mod, name, factory)
    return state


@pytest.mark.parametrize("accelerator", ["ediis", "adiis"])
def test_roothaan_coupling_projects_with_the_extrapolated_density(
    accelerator, monkeypatch
):
    """Every extrapolated cycle couples F(D_tilde) with D_tilde's projectors.

    GitLab #487 / IID #119: the EDIIS/ADIIS return is F(Σ_i c_i D_i); the
    Roothaan effective Fock must be assembled with that density, read back
    from the accelerator itself, never with the current iterate.  Synthetic
    HF system; the guard fires once here but re-solves onto the newest
    vertex, so this pins the identity on a clean trajectory -- the
    discriminating case is the CN fixture below.  (``ediis_diis`` takes the
    DIIS branch on every cycle of this well-conditioned system, so it is
    exercised on the CN fixture only.)
    """
    import vibeqc.rohf as rohf_mod

    state = _install_roothaan_coupling_spy(monkeypatch)
    s, h, eri = _synth(8, 6, gapped=True)
    bj, bk = _jk(eri)
    na, nb = 4, 2
    fbuild = _make_hf_fock_builder(h, bj, bk)
    x = _orthonormaliser(s, 1e-10)
    _eps, c = np.linalg.eigh(x.T @ h @ x)
    c = x @ c
    da, db = _aufbau_densities(c, na, nb)
    opts = ROHFOptions(max_iter=400, conv_tol_energy=1e-12, conv_tol_grad=1e-9)
    opts.scf_accelerator = accelerator
    res = rohf_mod.run_roothaan_scf(s, h, 0.0, na, nb, fbuild, opts, da, db)
    assert res.converged
    assert state["residuals"], "no cycle used an extrapolated Fock pair"
    assert max(state["residuals"]) < 1e-8, (
        f"{accelerator}: extrapolated Fock pair coupled with a density it "
        f"does not belong to (max |F_extrap - F(D_coupled)| = "
        f"{max(state['residuals']):.3e})"
    )


#: CN radical, ROHF/STO-3G from the SAD guess: the C++ anti-replay guard
#: erases an EDIIS/ADIIS history entry inside ``run_roothaan_scf`` on this
#: trajectory (measured 2026-09-02: 1 erasure under ediis, adiis and
#: ediis_diis), after which the pre-#487 Python density-history mirror was
#: one entry too long for 6 cycles (1 under ediis_diis) and the loop fell
#: back to the current density -- which differed from the extrapolated one
#: by up to 0.105 (1.08 under ediis_diis) in the density-matrix max norm.
_CN_ROHF_STO3G_E = -90.9974703508


@pytest.mark.parametrize("accelerator", ["ediis", "adiis", "ediis_diis"])
def test_roothaan_coupling_survives_a_guard_erasure(accelerator, monkeypatch, tmp_path):
    """The guard fires inside ``run_roothaan_scf`` and the coupling stays
    consistent afterwards (GitLab #487 closure criterion 1)."""
    from vibeqc import InitialGuess, run_job
    from vibeqc import ROHFOptions as _ROHFOptions

    monkeypatch.chdir(tmp_path)
    state = _install_roothaan_coupling_spy(monkeypatch)
    mol = Molecule(
        [Atom(6, [0.0, 0.0, 0.0]), Atom(7, [0.0, 0.0, 1.17 * ANGSTROM_TO_BOHR])],
        0,
        2,
    )
    opts = _ROHFOptions()
    opts.scf_accelerator = accelerator
    opts.max_iter = 80
    opts.initial_guess = InitialGuess.SAD
    res = run_job(mol, basis="sto-3g", method="rohf", rohf_options=opts,
                  output=f"cn_{accelerator}")
    assert res.converged
    erasures = sum(int(a.replay_guard_erasures()) for a in state["accel"])
    assert erasures >= 1, (
        f"{accelerator}: the anti-replay guard did not fire on the CN/STO-3G/"
        "SAD trajectory; this fixture no longer exercises #487"
    )
    assert state["residuals"], "no cycle used an extrapolated Fock pair"
    assert max(state["residuals"]) < 1e-8, (
        f"{accelerator}: after the guard erased history, an extrapolated Fock "
        f"pair was coupled with a density it does not belong to (max "
        f"|F_extrap - F(D_coupled)| = {max(state['residuals']):.3e})"
    )
    # Same stationary point as plain DIIS (and as every accelerator).
    assert float(res.energy) == pytest.approx(_CN_ROHF_STO3G_E, abs=1e-8)


def test_roothaan_extrapolated_density_pair_fails_loud_on_a_block_mismatch():
    """No silent fallback: a closed-shell (one-block) accelerator cannot
    feed the open-shell coupling (GitLab #487 closure criterion 2)."""
    from vibeqc import EDIIS
    from vibeqc.rohf import _extrapolated_density_pair

    e = EDIIS(max_subspace=4)
    e.extrapolate(np.eye(2), np.eye(2), energy=-1.0)
    with pytest.raises(RuntimeError, match="GitLab #487"):
        _extrapolated_density_pair(e)
    assert not hasattr(__import__("vibeqc.rohf", fromlist=["x"]),
                       "_hull_extrapolated_densities")


@pytest.mark.parametrize("method", ["rohf", "roks"])
def test_ro_restart_does_not_hide_invalid_guess(method):
    from vibeqc import run_roks
    from vibeqc.roks import ROKSOptions
    mol, basis, _, _, _ = _guess_probe_system()
    opts = ROHFOptions() if method == "rohf" else ROKSOptions()
    opts.initial_guess = "typo"
    opts.max_iter = 0
    seed = (np.eye(basis.nbasis), np.eye(basis.nbasis))
    run = run_rohf if method == "rohf" else run_roks
    with pytest.raises(ValueError, match="unknown initial_guess"):
        run(mol, basis, opts, initial_density=seed)


def test_rohf_auto_atom_reaches_patom_builder(monkeypatch):
    from vibeqc import _vibeqc_core as core
    mol = Molecule([Atom(8, [0., 0., 0.])], multiplicity=3)
    basis = BasisSet(mol, "sto-3g")
    original = core._guess_open_shell_density_with_jk
    reached = []
    def probe(*args):
        reached.append(args[4])
        return original(*args)
    monkeypatch.setattr(core, "_guess_open_shell_density_with_jk", probe)
    opts = ROHFOptions(initial_guess="auto", max_iter=0)
    result = run_rohf(mol, basis, opts)
    assert reached == [core.InitialGuess.PATOM]
    assert result.guess_selection.effective == core.InitialGuess.PATOM


def test_rohf_builder_error_propagates(monkeypatch):
    from vibeqc import _vibeqc_core as core
    def fail(*args):
        raise RuntimeError("atomic builder failed")
    monkeypatch.setattr(core, "_guess_open_shell_density_with_jk", fail)
    with pytest.raises(RuntimeError, match="atomic builder failed"):
        _density_for_guess("sad")


@pytest.mark.parametrize("method", ["rohf", "roks"])
@pytest.mark.parametrize("guess", ["auto", "hcore", "sad", "sap", "patom", "hueckel", "minao"])
def test_public_ro_guess_end_to_end(method, guess, tmp_path):
    from vibeqc import run_job
    from vibeqc.roks import ROKSOptions
    from vibeqc.runner import _detect_scf_guess
    from vibeqc.guess import resolve_initial_guess
    from vibeqc import _vibeqc_core as core
    mol = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 1.4])])
    options = ROHFOptions() if method == "rohf" else ROKSOptions()
    options.max_iter = 100
    kwargs = {method + "_options": options}
    if method == "roks":
        kwargs["functional"] = "lda"
    result = run_job(
        mol, basis="sto-3g", method=method, initial_guess=guess,
        output=tmp_path / "guess", write_molden_file=False, citations=False, **kwargs,
    )
    scf = result
    assert scf.converged
    basis = BasisSet(mol, "sto-3g")
    overlap = core.compute_overlap(basis)
    assert np.trace(scf.density_alpha @ overlap) == pytest.approx(1., abs=1e-10)
    assert np.trace(scf.density_beta @ overlap) == pytest.approx(1., abs=1e-10)
    effective = resolve_initial_guess(mol, guess, is_open_shell=True)
    assert scf.guess_selection.effective == effective
    cite = _detect_scf_guess(method, mol, options)
    assert (cite is None) == (effective == core.InitialGuess.HCORE)


@pytest.mark.parametrize("method", ["rohf", "roks"])
def test_public_ro_read_restart_preserves_selection(method, tmp_path):
    from vibeqc import run_job
    mol = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 1.4])])
    kwargs = dict(basis="sto-3g", method=method, write_molden_file=False,
                  citations=False)
    if method == "roks":
        kwargs["functional"] = "lda"
    first = run_job(mol, initial_guess="sap", output=tmp_path / "first", **kwargs)
    restarted = run_job(mol, initial_guess="read", read_from=first,
                        output=tmp_path / "restart", **kwargs)
    assert restarted.converged
    assert restarted.energy == pytest.approx(first.energy, abs=1e-9)
    assert restarted.guess_selection.effective.name == "READ"


def test_canonical_coercer_rejects_invalid_native_enum():
    from vibeqc import InitialGuess
    from vibeqc._initial_guess import coerce_initial_guess
    with pytest.raises(ValueError, match="unknown initial_guess"):
        coerce_initial_guess(InitialGuess(500))


@pytest.mark.parametrize("method", ["rohf", "roks"])
def test_public_ro_fragmo_reaches_fragment_builder(method, tmp_path, monkeypatch):
    from vibeqc import run_job
    import vibeqc.guess_fragmo as fragmo
    mol = Molecule([Atom(1, [0., 0., z]) for z in (0., 1.4, 6., 7.4)])
    fragments = [fragmo.Fragment([0, 1]), fragmo.Fragment([2, 3])]
    original = fragmo.resolve_fragmo_densities_open
    reached = []
    def probe(*args):
        reached.append(True)
        return original(*args)
    monkeypatch.setattr(fragmo, "resolve_fragmo_densities_open", probe)
    kw = {"functional": "lda"} if method == "roks" else {}
    result = run_job(
        mol, basis="sto-3g", method=method, initial_guess="fragmo",
        fragments=fragments, output=tmp_path / "fragmo",
        write_molden_file=False, citations=False, **kw,
    )
    assert reached == [True]
    assert result.converged
    assert result.guess_selection.effective.name == "FRAGMO"


@pytest.mark.parametrize("method", ["rohf", "roks"])
@pytest.mark.parametrize("guess", ["auto", "hcore", "sad", "sap", "patom", "hueckel", "minao"])
def test_ro_open_shell_guess_basin(method, guess):
    from vibeqc.roks import ROKSOptions, run_roks
    mol, basis, _, _, _ = _guess_probe_system()
    options = ROHFOptions() if method == "rohf" else ROKSOptions()
    options.initial_guess = guess
    options.max_iter = 300
    run = run_rohf if method == "rohf" else run_roks
    result = run(mol, basis, options)
    assert result.converged
    # Linear OH has two admitted ROKS occupation basins. SAD/SAP/MINAO
    # retain the degenerate 1.5/1.5 frontier ensemble; orbital guesses can
    # choose an integer 2/1 member. Preserve each existing basin rather than
    # forcing one by changing the initial-guess or occupation policy.
    integer_basin = method == "roks" and guess in ("hcore", "patom", "hueckel")
    options.initial_guess = "hcore" if integer_basin else "sad"
    baseline = run(mol, basis, options)
    assert baseline.converged
    assert result.energy == pytest.approx(baseline.energy, abs=1e-7)
    if method == "roks":
        assert np.allclose(result.mo_occupations, baseline.mo_occupations)
