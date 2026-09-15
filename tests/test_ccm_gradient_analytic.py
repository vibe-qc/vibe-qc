"""Analytic Γ-CCM RHF nuclear gradient (forces) — paper #3.

Gates :func:`vibeqc.periodic.ccm.run_ccm_rhf_gradient` against three independent
references:

* the finite-difference gate ``ccm_numerical_gradient`` on the H₂ chain (the
  full-energy derivative under the cyclic constraint), at the FD truncation floor;
* the molecular analytic RHF gradient ``compute_gradient`` in the isolated
  ``(1,1,1)`` big-box limit (analytic vs analytic — machine precision, no FD);
* the translational sum rule ``Σ_β grad = 0`` (Theorem 1.6.1).

Derivation: ``docs/manuscripts/aiccm_a_forces.md`` (§1.1–1.6). The analytic value
is exact; the residual vs FD is O(h²) truncation (validated to 1.1e-8 at h=1e-4).

Reference: Helgaker, Jørgensen & Olsen, *Molecular Electronic-Structure Theory*
(Eq. 12.5.7); Pulay, Mol. Phys. 17, 197 (1969); Peintinger & Bredow,
J. Comput. Chem. 35, 839 (2014).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    PeriodicSystem,
    RHFOptions,
    RKSOptions,
    UHFOptions,
    UKSOptions,
    compute_gradient,
    compute_gradient_rks,
    compute_gradient_uhf,
    compute_gradient_uks,
    run_rhf,
    run_rks,
    run_uhf,
    run_uks,
)
from vibeqc.periodic.ccm import (
    CCMSystem,
    ccm_numerical_gradient,
    run_ccm_ccsd_gradient,
    run_ccm_mp2_gradient,
    run_ccm_rhf_gradient,
    run_ccm_rks_gradient,
    run_ccm_uccsd_gradient,
    run_ccm_uhf_gradient,
    run_ccm_uks_gradient,
    run_ccm_ump2_gradient,
)
from vibeqc.periodic.ccm.dft import run_ccm_rks, run_ccm_uks
from vibeqc.periodic.ccm.scf import run_ccm_rhf
from vibeqc.periodic.ccm.uhf import run_ccm_uhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


def _h2_chain(cell=6.0, vac=15.0, d=1.3):
    return PeriodicSystem(
        3, np.diag([cell, vac, vac]),
        [Atom(1, [0, 0, 0]), Atom(1, [d, 0, 0])], charge=0, multiplicity=1,
    )


def _tight(ccm):
    # The analytic gradient needs a stationary SCF (first-order sensitive to F_ai).
    return run_ccm_rhf(ccm, method="aiccm2026dev-a", conv_tol=1e-12, max_iter=512)


@pytest.mark.parametrize("nrep", [(2, 1, 1), (3, 1, 1), (4, 1, 1)])
def test_analytic_matches_numerical_h2_chain(nrep):
    """Total analytic gradient == full-energy FD on the H₂ chain (FD floor)."""
    ccm = CCMSystem(_h2_chain(), nrep, "sto-3g")
    g_analytic = run_ccm_rhf_gradient(ccm)
    g_numeric = ccm_numerical_gradient(ccm, runner=_tight, h=1e-3)
    assert g_analytic.shape == (ccm.n_basis_atoms, 3)
    # h=1e-3 central-difference truncation is O(h²)~1e-6; the analytic value is exact.
    assert np.max(np.abs(g_analytic - g_numeric)) < 5e-6


def test_analytic_translational_sum_rule():
    """Σ_β grad = 0 (Theorem 1.6.1) — exact even at boundary ties."""
    ccm = CCMSystem(_h2_chain(), (3, 1, 1), "sto-3g")
    g = run_ccm_rhf_gradient(ccm)
    assert np.allclose(g.sum(axis=0), 0.0, atol=1e-10)


def test_mp2_numerical_gradient_translational_sum_rule():
    """CCM MP2 energy is translationally invariant: the FD MP2 gradient's per-cell
    forces sum to ≈ 0.

    **Step 0 of the post-HF forces line (paper #3):** the finite-difference MP2
    gradient is the reference the future *analytic* MP2 gradient will validate
    against (the ``ccm_numerical_gradient(runner=…)`` pattern generalises to any
    total-energy runner). Standalone, it is a translational-invariance regression
    gate for the CCM MP2 correlation energy: a symmetry-breaking bug (cf. the Γ
    pair-FT mirror bug, ~2e-2 Ha/cell) would wreck this ~10⁴× force cancellation,
    while the FD/SCF-noise floor at the h≈2e-3 sweet spot is ~1e-6. Dense
    ``union12`` four-center (cheap, exact — no cderi build); the physical chain
    force is ~0.08 Ha/bohr, so the sum rule is a nontrivial cancellation.
    """
    from types import SimpleNamespace

    from vibeqc.periodic.ccm.mp2 import run_ccm_mp2

    def mp2_runner(c):
        # tight SCF so the FD sees a smooth surface; e_total = e_hf + e_correlation
        scf = run_ccm_rhf(c, conv_tol=1e-12, max_iter=512)
        return SimpleNamespace(energy=run_ccm_mp2(c, scf).e_total)

    ccm = CCMSystem(_h2_chain(), (2, 1, 1), "sto-3g")
    g = ccm_numerical_gradient(ccm, runner=mp2_runner, h=2.0e-3)
    assert g.shape == (ccm.n_basis_atoms, 3)
    assert np.max(np.abs(g)) > 1e-2                       # nontrivial physical force
    assert np.max(np.abs(g.sum(axis=0))) < 1e-4          # sum rule (measured ~5.6e-6)


def test_analytic_isolated_limit_matches_molecular():
    """(1,1,1) in a huge box: analytic CCM gradient == molecular analytic gradient.

    Analytic vs analytic (no finite difference) — the Γ-CCM gradient reduces exactly
    to vibe-qc's molecular RHF gradient in the molecular limit (machine precision).
    """
    geom = [(1, [0, 0, 0]), (1, [1.3, 0, 0])]
    iso = PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]), [Atom(z, p) for z, p in geom])
    ccm = CCMSystem(iso, (1, 1, 1), "sto-3g")
    g_ccm = run_ccm_rhf_gradient(ccm)

    mol = Molecule([Atom(z, p) for z, p in geom], 0, 1)
    b = BasisSet(mol, "sto-3g")
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    g_mol = np.asarray(compute_gradient(mol, b, run_rhf(mol, b, opts)))
    assert np.max(np.abs(g_ccm - g_mol)) < 1e-8


def test_analytic_gradient_richardson_is_exact():
    """The analytic−FD residual is pure O(h²) truncation (halving h shrinks it ~4×)."""
    ccm = CCMSystem(_h2_chain(), (3, 1, 1), "sto-3g")
    g = run_ccm_rhf_gradient(ccm)
    e_big = np.max(np.abs(g - ccm_numerical_gradient(ccm, runner=_tight, h=1e-3)))
    e_small = np.max(np.abs(g - ccm_numerical_gradient(ccm, runner=_tight, h=5e-4)))
    # O(h²): halving h reduces the truncation error by ~4 (allow a loose band).
    assert e_small < e_big / 3.0


# --------------------------------------------------------------------------- #
# MP2 (relaxed-density) analytic gradient — §1.8 of the forces manuscript
# --------------------------------------------------------------------------- #
def _mp2_tight(ccm):
    """Tight-SCF MP2 total-energy runner for the FD gate (method-consistent)."""
    from types import SimpleNamespace

    from vibeqc.periodic.ccm.mp2 import run_ccm_mp2

    scf = run_ccm_rhf(ccm, method="aiccm2026dev-a", conv_tol=1e-12, max_iter=512)
    return SimpleNamespace(energy=run_ccm_mp2(ccm, scf, method="aiccm2026dev-a").e_total)


def test_mp2_analytic_matches_numerical_h2_chain():
    """MP2 analytic gradient == full-energy FD, with O(h²) Richardson (§1.8 val. 2).

    Gated on (3,1,1): the (2,1,1) chain's CCM-RHF ground state is symmetry-broken
    with a ~8e-12-split branch pair, so displaced SCFs branch-hop and the *FD* MP2
    reference is erratic there (E₂ differs ~1e-5 between branches while E_HF is
    branch-insensitive) — an FD-reference pathology, not a gradient defect
    (§1.8 caveat; same class as the UKS spin-branch FD caveat).
    """
    ccm = CCMSystem(_h2_chain(), (3, 1, 1), "sto-3g")
    g = run_ccm_mp2_gradient(ccm)
    assert g.shape == (ccm.n_basis_atoms, 3)
    assert np.max(np.abs(g)) > 1e-2                        # nontrivial physical force
    assert np.allclose(g.sum(axis=0), 0.0, atol=1e-10)     # sum rule (Thm 1.6.1)
    e_big = np.max(np.abs(g - ccm_numerical_gradient(ccm, runner=_mp2_tight, h=1e-3)))
    e_small = np.max(np.abs(g - ccm_numerical_gradient(ccm, runner=_mp2_tight, h=5e-4)))
    assert e_big < 5e-6                                    # FD floor at h=1e-3
    # O(h²): halving h shrinks the residual ~4× ⇒ the analytic value is exact and
    # the gap is pure FD truncation (measured 1.07e-6 → 2.66e-7).
    assert e_small < e_big / 3.0


def test_mp2_isolated_limit_matches_molecular_mp2_fd():
    """(1,1,1) big box: CCM-MP2 analytic gradient == FD of the *molecular* MP2.

    Cross-implementation check (§1.8 validation 3): vibe-qc's molecular
    ``run_rhf + run_mp2`` (C++ MP2 on canonical AO ERIs) shares no code with the
    CCM path; at h=1e-4 the central-difference floor is ~1e-9 (measured 3.6e-9).
    """
    from vibeqc import MP2Options, run_mp2

    geom = [(1, [0.0, 0.0, 0.0]), (1, [1.3, 0.0, 0.0])]
    iso = PeriodicSystem(3, np.diag([80.0] * 3), [Atom(z, p) for z, p in geom])
    ccm = CCMSystem(iso, (1, 1, 1), "sto-3g")
    g_ccm = run_ccm_mp2_gradient(ccm)

    def mol_e(coords):
        mol = Molecule([Atom(1, list(c)) for c in coords], 0, 1)
        b = BasisSet(mol, "sto-3g")
        o = RHFOptions()
        o.conv_tol_energy = 1e-13
        return float(run_mp2(mol, b, run_rhf(mol, b, o), MP2Options()).e_total)

    h = 1e-4
    base = [np.array(p, dtype=float) for _, p in geom]
    g_fd = np.zeros((2, 3))
    for a in range(2):
        for d in range(3):
            cp = [c.copy() for c in base]
            cm = [c.copy() for c in base]
            cp[a][d] += h
            cm[a][d] -= h
            g_fd[a, d] = (mol_e(cp) - mol_e(cm)) / (2 * h)
    assert np.max(np.abs(g_ccm - g_fd)) < 5e-8


def test_mp2_relaxed_density_response_identity():
    """dE_MP2/dλ for h^CCM → h^CCM + λV equals Tr[(P_SCF + P^Δ) V] (§1.8 val. 1).

    Isolates the relaxed one-particle density — the amplitude blocks d_oo/d_vv and
    the Z-vector (CCM-CPHF) ov block — from all geometric machinery: the FD side
    re-solves a perturbed-h CCM SCF + MP2 with no integral derivatives involved.
    Pinned at 1e-9 (measured 2.8e-11; the relaxation correction itself is ~7.7e-6,
    so an unrelaxed density fails this by ~4 orders of magnitude).
    """
    from vibeqc.periodic.ccm.gradient_analytic import (
        _g_ccm,
        _mp2_relaxed_densities,
    )
    from vibeqc.periodic.ccm.integrals import ccm_overlap
    from vibeqc.periodic.ccm.padded import ccm_hcore, ccm_nuclear_repulsion
    from vibeqc.periodic.ccm.scf import _ccm_eri_for_method

    ccm = CCMSystem(_h2_chain(), (2, 1, 1), "sto-3g")
    scf = _tight(ccm)
    eri = _ccm_eri_for_method(ccm, "aiccm2026dev-a")
    n_occ = ccm.supercell.n_electrons() // 2
    P_delta, _, _ = _mp2_relaxed_densities(
        ccm, scf.mo_coeffs, scf.mo_energies, n_occ, eri
    )
    P_relax = np.asarray(scf.density) + P_delta

    h0 = np.asarray(ccm_hcore(ccm)[0], dtype=float)
    S = np.asarray(ccm_overlap(ccm), dtype=float)
    vnn = float(ccm_nuclear_repulsion(ccm))
    rng = np.random.default_rng(7)
    V = rng.standard_normal(h0.shape)
    V = 0.005 * (V + V.T)

    def e_mp2_total(h):
        """Plain Roothaan RHF (fixed-point + damping) + RMP2 on (h, S, eri)."""
        w, U = np.linalg.eigh(S)
        Xo = U / np.sqrt(w)
        F, P = h.copy(), None
        for _ in range(2000):
            e, Cp = np.linalg.eigh(Xo.T @ F @ Xo)
            C = Xo @ Cp
            Pn = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
            P = Pn if P is None else 0.6 * Pn + 0.4 * P
            Fn = h + _g_ccm(eri, P)
            if np.max(np.abs(Fn - F)) < 1e-13:
                F = Fn
                break
            F = Fn
        e, Cp = np.linalg.eigh(Xo.T @ F @ Xo)
        C = Xo @ Cp
        P = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        e_hf = 0.5 * float(np.sum(P * (2.0 * h + _g_ccm(eri, P)))) + vnn
        g = np.einsum("mnls,mp,nq,lr,st->pqrt", eri, C, C, C, C, optimize=True)
        eo, ev = e[:n_occ], e[n_occ:]
        D = (eo[:, None, None, None] + eo[None, :, None, None]
             - ev[None, None, :, None] - ev[None, None, None, :])
        gov = g[:n_occ, n_occ:, :n_occ, n_occ:].transpose(0, 2, 1, 3)
        t = gov / D
        tt = 2.0 * t - t.transpose(0, 1, 3, 2)
        return e_hf + float(np.einsum("ijab,ijab->", tt, gov))

    lam = 1e-5
    fd = (e_mp2_total(h0 + lam * V) - e_mp2_total(h0 - lam * V)) / (2 * lam)
    assert abs(fd - float(np.sum(P_relax * V))) < 1e-9


# --------------------------------------------------------------------------- #
# CCSD (relaxed-density) analytic gradient — §1.9 of the forces manuscript
# --------------------------------------------------------------------------- #
def _ccsd_tight(ccm):
    """Tight-SCF CCSD total-energy runner for the FD gate (method-consistent, no (T))."""
    from types import SimpleNamespace

    from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd

    scf = run_ccm_rhf(ccm, method="aiccm2026dev-a", conv_tol=1e-12, max_iter=512)
    ccsd = run_ccm_ccsd(ccm, scf, method="aiccm2026dev-a", compute_triples=False,
                        conv_tol=1e-11)
    return SimpleNamespace(energy=ccsd.e_total)


def test_ccsd_analytic_matches_numerical_h2_chain():
    """CCSD relaxed-density gradient == full-energy FD, O(h²) Richardson (§1.9).

    Gated on (3,1,1): like MP2, the (2,1,1) chain's symmetry-broken branch pair makes
    the FD correlation reference erratic (§1.9 caveat). The relaxed density (Λ +
    Z-vector), the cumulant 2-PDM, and the energy-weighted density are all validated
    together here; the individual response identities (1-PDM, ∂E/∂mo, ∂E/∂S) hold to
    the FD floor in the derivation record.
    """
    ccm = CCMSystem(_h2_chain(), (3, 1, 1), "sto-3g")
    g = run_ccm_ccsd_gradient(ccm)
    assert g.shape == (ccm.n_basis_atoms, 3)
    assert np.max(np.abs(g)) > 1e-2                        # nontrivial physical force
    assert np.allclose(g.sum(axis=0), 0.0, atol=1e-10)     # sum rule (Thm 1.6.1)
    e_big = np.max(np.abs(g - ccm_numerical_gradient(ccm, runner=_ccsd_tight, h=1e-3)))
    e_small = np.max(np.abs(g - ccm_numerical_gradient(ccm, runner=_ccsd_tight, h=5e-4)))
    assert e_big < 5e-6                                    # FD floor at h=1e-3
    assert e_small < e_big / 3.0                           # O(h²) truncation


def test_ccsd_isolated_limit_matches_molecular_ccsd_fd():
    """(1,1,1) big box: CCM-CCSD analytic gradient == FD of the *molecular* CCSD.

    Cross-implementation check (§1.9): vibe-qc's molecular ``run_ccsd`` (C++ spin-
    adapted CCSD) shares no code with the CCM spin-orbital path; at h=1e-4 the FD
    floor is ~1e-9 (measured 3.6e-9). Both are all-electron CCSD (no (T)).
    """
    from vibeqc import CCSDOptions, run_ccsd

    geom = [(1, [0.0, 0.0, 0.0]), (1, [1.3, 0.0, 0.0])]
    iso = PeriodicSystem(3, np.diag([80.0] * 3), [Atom(z, p) for z, p in geom])
    ccm = CCMSystem(iso, (1, 1, 1), "sto-3g")
    g_ccm = run_ccm_ccsd_gradient(ccm)

    def mol_e(coords):
        mol = Molecule([Atom(1, list(c)) for c in coords], 0, 1)
        b = BasisSet(mol, "sto-3g")
        o = RHFOptions()
        o.conv_tol_energy = 1e-13
        r = run_rhf(mol, b, o)
        co = CCSDOptions()
        co.density_fit = False
        co.compute_triples = False
        co.conv_tol_energy = 1e-11
        return float(run_ccsd(mol, b, r, co).e_total)

    h = 1e-4
    base = [np.array(p, dtype=float) for _, p in geom]
    g_fd = np.zeros((2, 3))
    for a in range(2):
        for d in range(3):
            cp = [c.copy() for c in base]
            cm = [c.copy() for c in base]
            cp[a][d] += h
            cm[a][d] -= h
            g_fd[a, d] = (mol_e(cp) - mol_e(cm)) / (2 * h)
    assert np.max(np.abs(g_ccm - g_fd)) < 5e-8


# --------------------------------------------------------------------------- #
# UCCSD (open-shell relaxed-density) analytic gradient — §1.9.5
# --------------------------------------------------------------------------- #
def test_uccsd_closed_shell_reduction_matches_ccsd():
    """UCCSD gradient on a closed-shell system == the (validated) CCSD gradient.

    The primary UCCSD gate: on a closed-shell cluster the UHF reference collapses to
    RHF and every spin-resolved piece (the α/β cumulant blocks, the spin-coupled
    Z-vector, the per-spin W) must reduce exactly to the CCSD gradient — a
    machine-precision analytic-vs-analytic check that pins the entire open-shell
    assembly (measured 1.6e-13).
    """
    ccm = CCMSystem(_h2_chain(), (3, 1, 1), "sto-3g")
    g_u = run_ccm_uccsd_gradient(ccm)
    g_c = run_ccm_ccsd_gradient(ccm)
    assert np.max(np.abs(g_u - g_c)) < 1e-11
    assert np.allclose(g_u.sum(axis=0), 0.0, atol=1e-10)     # sum rule


def test_uccsd_isolated_limit_matches_molecular_uccsd_fd():
    """(1,1,1) big box: CCM-UCCSD analytic gradient == FD of the *molecular* UCCSD.

    Cross-implementation check (§1.9.5): vibe-qc's molecular ``run_uccsd`` shares no
    code with the CCM spin-orbital path. The tolerance (5e-6) is the open-shell FD
    floor — the UHF reference's energy noise (~3e-9) gives gradient FD noise ~ε/h,
    not analytic error; the analytic value is exact (machine-precision closed-shell
    reduction + ~1e-9 direct response identities in the derivation record).
    """
    from vibeqc import CCSDOptions, UHFOptions, run_uccsd, run_uhf

    geom = [(2, [0.0, 0.0, 0.0]), (1, [1.5, 0.0, 0.0])]
    iso = PeriodicSystem(3, np.diag([80.0] * 3),
                         [Atom(z, p) for z, p in geom], charge=0, multiplicity=2)
    ccm = CCMSystem(iso, (1, 1, 1), "6-31g")
    g_ccm = run_ccm_uccsd_gradient(ccm)
    assert np.allclose(g_ccm.sum(axis=0), 0.0, atol=1e-9)

    def mol_e(coords):
        mol = Molecule([Atom(z, list(c)) for (z, _), c in zip(geom, coords)], 0, 2)
        b = BasisSet(mol, "6-31g")
        o = UHFOptions()
        o.conv_tol_energy = 1e-13
        o.max_iter = 500
        r = run_uhf(mol, b, o)
        co = CCSDOptions()
        co.density_fit = False
        co.compute_triples = False
        co.conv_tol_energy = 1e-11
        return float(run_uccsd(mol, b, r, co).e_total)

    h = 1e-4
    base = [np.array(p, dtype=float) for _, p in geom]
    g_fd = np.zeros((2, 3))
    for a in range(2):
        for d in range(3):
            cp = [c.copy() for c in base]
            cm = [c.copy() for c in base]
            cp[a][d] += h
            cm[a][d] -= h
            g_fd[a, d] = (mol_e(cp) - mol_e(cm)) / (2 * h)
    assert np.max(np.abs(g_ccm - g_fd)) < 5e-6


# --------------------------------------------------------------------------- #
# UMP2 (open-shell relaxed-density) analytic gradient — §1.8.4
# --------------------------------------------------------------------------- #
def test_ump2_closed_shell_reduction_matches_rmp2():
    """UMP2 gradient on a closed-shell system == RMP2 gradient (machine ε).

    Pins every spin factor of §1.8.4 against the FD-validated closed-shell set:
    ``t^αβ = t``, ``t^σσ = t − t_swap``, ``X^σ = 2X_cs``, ``z^σ = ½z_cs``, the
    per-spin W blocks, and the Γ_ns/Γ_sep assemblies (measured 6.2e-15).
    """
    ccm = CCMSystem(_h2_chain(), (3, 1, 1), "sto-3g")
    g_u = run_ccm_ump2_gradient(ccm)
    g_r = run_ccm_mp2_gradient(ccm)
    assert np.max(np.abs(g_u - g_r)) < 1e-12


def _ump2_tight(ccm):
    """Tight-SCF UMP2 total-energy runner for the FD gate (method-consistent)."""
    from types import SimpleNamespace

    from vibeqc.periodic.ccm.ump2 import run_ccm_ump2

    scf = run_ccm_uhf(ccm, method="aiccm2026dev-a", conv_tol=1e-12, max_iter=512)
    return SimpleNamespace(
        energy=run_ccm_ump2(ccm, scf, method="aiccm2026dev-a").e_total)


def test_ump2_analytic_matches_numerical_heh_chain():
    """UMP2 analytic gradient == full-energy FD on the open-shell HeH chain.

    The (3,1,1) doublet chain (9 e⁻, ``n_occ_σ < nbf`` — off the fully-filled-spin
    edge). O(h²) Richardson: measured ``4.9e-6 → 1.2e-6 → 3.1e-7`` for
    ``h = 2e-3, 1e-3, 5e-4`` (§1.8.4 validation 3); sum rule 1.6e-14.
    """
    ccm = CCMSystem(_heh_chain(), (3, 1, 1), "sto-3g")
    g = run_ccm_ump2_gradient(ccm)
    assert g.shape == (ccm.n_basis_atoms, 3)
    assert np.max(np.abs(g)) > 1e-2                        # nontrivial physical force
    assert np.allclose(g.sum(axis=0), 0.0, atol=1e-10)     # sum rule (Thm 1.6.1)
    e_big = np.max(np.abs(g - ccm_numerical_gradient(ccm, runner=_ump2_tight, h=1e-3)))
    e_small = np.max(np.abs(g - ccm_numerical_gradient(ccm, runner=_ump2_tight, h=5e-4)))
    assert e_big < 5e-6
    assert e_small < e_big / 3.0                           # O(h²) truncation


def test_ump2_isolated_limit_matches_molecular_ump2_fd():
    """(1,1,1) big box: CCM-UMP2 analytic gradient == FD of the *molecular* UMP2.

    Cross-implementation check (§1.8.4 validation 4): vibe-qc's molecular
    ``run_uhf + run_ump2`` shares no code with the CCM path; HeH/6-31G keeps
    ``n_α = 2 < nbf`` (off the fully-filled-spin edge). Measured 1.1e-9 at h=1e-4.
    """
    from vibeqc import UMP2Options, run_ump2

    geom = [(2, [0.0, 0.0, 0.0]), (1, [1.5, 0.0, 0.0])]
    iso = PeriodicSystem(3, np.diag([80.0] * 3),
                         [Atom(z, p) for z, p in geom], charge=0, multiplicity=2)
    ccm = CCMSystem(iso, (1, 1, 1), "6-31g")
    g_ccm = run_ccm_ump2_gradient(ccm)

    def mol_e(coords):
        mol = Molecule([Atom(z, list(c)) for (z, _), c in zip(geom, coords)], 0, 2)
        b = BasisSet(mol, "6-31g")
        o = UHFOptions()
        o.conv_tol_energy = 1e-13
        o.conv_tol_grad = 1e-11
        o.max_iter = 500
        return float(run_ump2(mol, b, run_uhf(mol, b, o), UMP2Options()).e_total)

    h = 1e-4
    base = [np.array(p, dtype=float) for _, p in geom]
    g_fd = np.zeros((2, 3))
    for a in range(2):
        for d in range(3):
            cp = [c.copy() for c in base]
            cm = [c.copy() for c in base]
            cp[a][d] += h
            cm[a][d] -= h
            g_fd[a, d] = (mol_e(cp) - mol_e(cm)) / (2 * h)
    assert np.max(np.abs(g_ccm - g_fd)) < 5e-8


# --------------------------------------------------------------------------- #
# UHF (open-shell) analytic gradient
# --------------------------------------------------------------------------- #
def _heh_chain(cell=6.0, vac=15.0, d=1.5):
    """Open-shell HeH chain (3 electrons/cell; odd-N supercells stay doublet)."""
    return PeriodicSystem(
        3, np.diag([cell, vac, vac]),
        [Atom(2, [0, 0, 0]), Atom(1, [d, 0, 0])], charge=0, multiplicity=2,
    )


def _uhf_tight(ccm):
    return run_ccm_uhf(ccm, method="aiccm2026dev-a", conv_tol=1e-12, max_iter=512)


@pytest.mark.parametrize("nrep", [(3, 1, 1), (5, 1, 1)])
def test_uhf_analytic_matches_numerical_heh_chain(nrep):
    """UHF total analytic gradient == full-energy FD on the open-shell HeH chain.

    Odd-N supercells carry an odd electron count (genuine doublet) with
    ``n_occ_σ < nbf`` (away from the fully-filled-spin edge case).
    """
    ccm = CCMSystem(_heh_chain(), nrep, "sto-3g")
    g_analytic = run_ccm_uhf_gradient(ccm)
    g_numeric = ccm_numerical_gradient(ccm, runner=_uhf_tight, h=1e-3)
    assert g_analytic.shape == (ccm.n_basis_atoms, 3)
    assert np.max(np.abs(g_analytic - g_numeric)) < 5e-6
    assert np.allclose(g_analytic.sum(axis=0), 0.0, atol=1e-9)  # sum rule


def test_uhf_isolated_limit_matches_molecular():
    """(1,1,1) big box: UHF-CCM analytic gradient == molecular UHF analytic gradient.

    HeH/6-31G keeps ``n_α = 2 < nbf = 4`` (off the fully-filled-spin edge); the
    open-shell Γ-CCM force reduces to vibe-qc's molecular UHF force (machine
    precision, no finite difference).
    """
    geom = [(2, [0, 0, 0]), (1, [1.5, 0, 0])]
    iso = PeriodicSystem(
        3, np.diag([80.0, 80.0, 80.0]),
        [Atom(z, p) for z, p in geom], charge=0, multiplicity=2,
    )
    ccm = CCMSystem(iso, (1, 1, 1), "6-31g")
    g_ccm = run_ccm_uhf_gradient(ccm)

    mol = Molecule([Atom(z, p) for z, p in geom], 0, 2)
    b = BasisSet(mol, "6-31g")
    opts = UHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    opts.max_iter = 500
    g_mol = np.asarray(compute_gradient_uhf(mol, b, run_uhf(mol, b, opts)))
    assert np.max(np.abs(g_ccm - g_mol)) < 1e-7


# --------------------------------------------------------------------------- #
# RKS (closed-shell DFT) analytic gradient
# --------------------------------------------------------------------------- #
def _rks_tight(functional):
    def runner(ccm):
        return run_ccm_rks(ccm, functional, method="aiccm2026dev-a",
                           conv_tol=1e-11, max_iter=256)
    return runner


@pytest.mark.parametrize("functional", ["pbe", "pbe0"])
def test_rks_analytic_matches_numerical_h2_chain(functional):
    """RKS total analytic gradient == full-energy FD on the H₂ chain.

    Both the analytic and the molecular DFT reference are grid-fixed (neglect the
    Becke weight derivative), so analytic−FD lands at the grid-fixed floor (~1e-6
    here, not machine zero); the FD step's O(h²) truncation dominates at h=1e-3.
    """
    ccm = CCMSystem(_h2_chain(), (3, 1, 1), "sto-3g")
    g_analytic = run_ccm_rks_gradient(ccm, functional)
    g_numeric = ccm_numerical_gradient(ccm, runner=_rks_tight(functional), h=1e-3)
    assert g_analytic.shape == (ccm.n_basis_atoms, 3)
    assert np.max(np.abs(g_analytic - g_numeric)) < 5e-6
    # DFT sum rule holds only to the grid-fixed floor (~1e-6): the fixed-grid
    # XC-Pulay neglects the Becke weight derivative, the term that restores exact
    # translational invariance — vibe-qc's molecular DFT gradient has the same
    # property. (Exact for HF; ~machine-zero here only because H₂ is symmetric.)
    assert np.allclose(g_analytic.sum(axis=0), 0.0, atol=5e-6)


@pytest.mark.parametrize("functional", ["pbe", "pbe0", "b3lyp"])
def test_rks_isolated_limit_matches_molecular(functional):
    """(1,1,1) big box: RKS-CCM analytic gradient == molecular RKS analytic gradient.

    Analytic vs analytic (no FD), both grid-fixed — the Γ-CCM KS force reduces to
    vibe-qc's molecular ``compute_gradient_rks`` to machine precision for pure (PBE)
    and hybrid (PBE0, B3LYP) functionals; the only non-folded piece (the XC) is the
    same molecular functional on the same supercell density/grid.
    """
    geom = [(1, [0, 0, 0]), (1, [1.4, 0, 0])]
    iso = PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]),
                         [Atom(z, p) for z, p in geom], charge=0, multiplicity=1)
    ccm = CCMSystem(iso, (1, 1, 1), "sto-3g")
    g_ccm = run_ccm_rks_gradient(ccm, functional)

    mol = Molecule([Atom(z, p) for z, p in geom], 0, 1)
    b = BasisSet(mol, "sto-3g")
    opts = RKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-12
    opts.max_iter = 300
    g_mol = np.asarray(compute_gradient_rks(mol, b, run_rks(mol, b, opts)))
    assert np.max(np.abs(g_ccm - g_mol)) < 1e-7


# --------------------------------------------------------------------------- #
# UKS (open-shell DFT) analytic gradient
# --------------------------------------------------------------------------- #
def _uks_tight(functional):
    def runner(ccm):
        return run_ccm_uks(ccm, functional, method="aiccm2026dev-a",
                           conv_tol=1e-11, max_iter=256)
    return runner


@pytest.mark.parametrize("functional", ["pbe", "pbe0"])
def test_uks_analytic_matches_numerical_heh_chain(functional):
    """UKS total analytic gradient == full-energy FD on the open-shell HeH chain.

    Uses a *dilute* chain (cell=12 bohr): a dense HeH chain is metallic/spin-unstable
    under a GGA, so the FD's displaced-geometry UKS SCFs jump between near-degenerate
    spin-broken branches and the finite difference becomes ill-defined (the analytic
    reference itself is deterministic; it's the FD that is unreliable there). At
    cell=12 the UKS solution is gapped and unique, and analytic == FD at the
    grid-fixed floor.
    """
    ccm = CCMSystem(_heh_chain(cell=12.0), (3, 1, 1), "sto-3g")
    g_analytic = run_ccm_uks_gradient(ccm, functional)
    g_numeric = ccm_numerical_gradient(ccm, runner=_uks_tight(functional), h=1e-3)
    assert g_analytic.shape == (ccm.n_basis_atoms, 3)
    assert np.max(np.abs(g_analytic - g_numeric)) < 5e-6
    # DFT sum rule holds only to the grid-fixed floor (~1e-6; the neglected Becke
    # weight derivative is what makes it exact) — same as molecular compute_gradient_uks
    # on this asymmetric HeH system. HF sum rules stay exact.
    assert np.allclose(g_analytic.sum(axis=0), 0.0, atol=5e-6)


@pytest.mark.parametrize("functional", ["pbe", "pbe0", "b3lyp"])
def test_uks_isolated_limit_matches_molecular(functional):
    """(1,1,1) big box: UKS-CCM analytic gradient == molecular UKS analytic gradient.

    HeH/6-31G keeps ``n_α = 2 < nbf`` (off the fully-filled-spin edge); the open-shell
    Γ-CCM KS force reduces to vibe-qc's molecular ``compute_gradient_uks`` to machine
    precision for pure (PBE) and hybrid (PBE0, B3LYP) functionals.
    """
    geom = [(2, [0, 0, 0]), (1, [1.5, 0, 0])]
    iso = PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]),
                         [Atom(z, p) for z, p in geom], charge=0, multiplicity=2)
    ccm = CCMSystem(iso, (1, 1, 1), "6-31g")
    g_ccm = run_ccm_uks_gradient(ccm, functional)

    mol = Molecule([Atom(z, p) for z, p in geom], 0, 2)
    b = BasisSet(mol, "6-31g")
    opts = UKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-12
    opts.max_iter = 500
    g_mol = np.asarray(compute_gradient_uks(mol, b, run_uks(mol, b, opts)))
    assert np.max(np.abs(g_ccm - g_mol)) < 1e-7


# ---------------------------------------------------------------------------
# CCSD(T) gradient — the non-canonical (T) functional (§1.10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gradient_runner",
    [run_ccm_ccsd_gradient, run_ccm_uccsd_gradient],
    ids=["ccsd", "uccsd"],
)
@pytest.mark.parametrize("compute_triples", [False, True], ids=["sd", "sd_t"])
def test_ccsd_lambda_matrix_free_matches_dense_oracle(
    gradient_runner, compute_triples
):
    """The bounded-memory Lambda solve preserves the dense gradient oracle."""
    system = PeriodicSystem(
        3,
        np.diag([80.0, 80.0, 80.0]),
        [Atom(1, [0, 0, 0]), Atom(1, [1.3, 0, 0])],
    )
    ccm = CCMSystem(system, (1, 1, 1), "sto-3g")
    dense = gradient_runner(ccm, compute_triples=compute_triples)
    matrix_free = gradient_runner(
        ccm,
        compute_triples=compute_triples,
        lambda_solver="matrix-free",
        lambda_tol=1e-10,
    )
    assert np.allclose(matrix_free, dense, atol=1e-12, rtol=0.0)


def test_ccsd_lambda_dense_storage_regression():
    """Record the O(n_so^8) dense-Jacobian memory bug without allocating it."""
    from vibeqc.periodic.ccm.gradient_analytic import (
        _ccsd_lambda_jacobian_nbytes,
    )

    nso, nocc_so = 40, 20
    n_amplitudes = 20 * 20 + 20**2 * 20**2
    dense_bytes = _ccsd_lambda_jacobian_nbytes(nso, nocc_so)
    assert dense_bytes == 8 * n_amplitudes**2
    assert dense_bytes > 200_000_000_000


def test_ccsd_lambda_matrix_free_nonconvergence_fails_closed():
    """An exhausted iterative Lambda solve must not yield a gradient."""
    system = PeriodicSystem(
        3,
        np.diag([80.0, 80.0, 80.0]),
        [Atom(1, [0, 0, 0]), Atom(1, [1.3, 0, 0])],
    )
    ccm = CCMSystem(system, (1, 1, 1), "sto-3g")
    with pytest.raises(RuntimeError, match="Lambda solve did not converge"):
        run_ccm_ccsd_gradient(
            ccm,
            lambda_solver="matrix-free",
            lambda_tol=1e-10,
            lambda_max_iter=1,
        )


def test_ccsd_t_energy_noncanonical_reduction():
    """The non-canonical (T) functional (§1.10.2) reduces bit-identically to the
    canonical ``ccsd._triples`` at the CCM-RHF reference (A = D one-shot,
    f_ov = 0), and is invariant under finite occ-occ/vir-vir rotations (the
    off-diagonal A[f] solve is the correct analytic continuation)."""
    from scipy.linalg import expm

    from vibeqc.periodic.ccm.ccsd import _spin_block_eri_phys, _triples
    from vibeqc.periodic.ccm.gradient_analytic import (
        _ccsd_fs_from_spatial,
        _ccsd_solve_amplitudes,
        _ccsd_t_energy_so,
        _t3_A_offdiag,
        _t3_connected_numerator,
        _t3_denominator,
    )
    from vibeqc.periodic.ccm.scf import _ccm_eri_for_method

    cell = PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]),
                          [Atom(3, [0, 0, 0]), Atom(1, [3.0, 0, 0])], 0, 1)
    ccm = CCMSystem(cell, (1, 1, 1), "sto-3g")
    eri = _ccm_eri_for_method(ccm, "aiccm2026dev-a")
    scf = run_ccm_rhf(ccm, eri=eri, conv_tol=1e-12, max_iter=512)
    C = np.asarray(scf.mo_coeffs, float)
    eps = np.asarray(scf.mo_energies, float)
    nocc_so = int(ccm.supercell.n_electrons())
    mo = np.einsum("mnls,mp,nq,lr,st->pqrt", eri, C, C, C, C, optimize=True)
    spinints = _spin_block_eri_phys(mo)
    fs = _ccsd_fs_from_spatial(np.diag(eps))
    ts, td = _ccsd_solve_amplitudes(fs, spinints, nocc_so)

    e_ref = _triples(spinints, fs, ts, td, nocc_so)
    e_nc = float(np.real(_ccsd_t_energy_so(ts, td, fs, spinints, nocc_so)))
    assert e_nc == pytest.approx(e_ref, abs=1e-18)

    # finite occ/vir rotation: converge the A[f] solve manually and compare
    nso = fs.shape[0]
    no, nv = nocc_so, nso - nocc_so
    rng = np.random.default_rng(7)

    def small_orth(k):
        a = rng.normal(size=(k, k)) * 0.05
        return expm(a - a.T)

    Qo, Qv = small_orth(no), small_orth(nv)
    Q = np.zeros((nso, nso))
    Q[:no, :no] = Qo
    Q[no:, no:] = Qv
    fs_r = Q.T @ fs @ Q
    si_r = np.einsum("pqrs,pa,qb,rc,sd->abcd", spinints, Q, Q, Q, Q, optimize=True)
    ts_r = Qv.T @ ts @ Qo
    td_r = np.einsum("abij,ac,bd,ie,jf->cdef", td, Qv, Qv, Qo, Qo, optimize=True)
    # iterate the preconditioned sweep to convergence for the finite rotation
    W = _t3_connected_numerator(si_r, td_r, nocc_so)
    D = _t3_denominator(fs_r, nocc_so)
    t3 = W / D
    for _ in range(200):
        t3_new = (W - _t3_A_offdiag(fs_r, t3, nocc_so)) / D
        if np.max(np.abs(t3_new - t3)) < 1e-15:
            t3 = t3_new
            break
        t3 = t3_new
    from vibeqc.periodic.ccm.gradient_analytic import _t3_disconnected_numerator
    V = _t3_disconnected_numerator(fs_r, si_r, ts_r, td_r, nocc_so)
    e_rot = float((1.0 / 36.0) * np.einsum("abcijk,abcijk->", t3, W + V))
    assert e_rot == pytest.approx(e_ref, abs=1e-13)


def test_ccsd_t_isolated_limit_matches_molecular_ccsd_t_fd():
    """Heteronuclear LiH (1,1,1): the CCSD(T) analytic gradient == FD of the
    *molecular* ``run_ccsd(compute_triples=True)`` (C++ spin-adapted path, no
    shared code). THE decisive gate for the (T) semicanonical gap: the canonical
    (T) complex-step route had an h-independent ~2.5e-5 error here (off-diagonal
    triples densities missing, HANDOVER_AICCM_GRADIENT.md 2026-07-17); the
    non-canonical functional (§1.10) must reach the molecular FD floor."""
    from vibeqc import CCSDOptions, run_ccsd

    geom = [(3, [0.0, 0.0, 0.0]), (1, [3.0, 0.0, 0.0])]
    iso = PeriodicSystem(3, np.diag([80.0] * 3), [Atom(z, p) for z, p in geom])
    ccm = CCMSystem(iso, (1, 1, 1), "sto-3g")
    g_ccm = run_ccm_ccsd_gradient(ccm, compute_triples=True)
    assert np.allclose(g_ccm.sum(axis=0), 0.0, atol=1e-10)

    def mol_e(coords):
        mol = Molecule([Atom(z, list(c)) for (z, _), c in zip(geom, coords)], 0, 1)
        b = BasisSet(mol, "sto-3g")
        o = RHFOptions()
        o.conv_tol_energy = 1e-13
        r = run_rhf(mol, b, o)
        co = CCSDOptions()
        co.compute_triples = True
        co.density_fit = False
        cc = run_ccsd(mol, b, r, co)
        return r.energy + cc.e_ccsd_correlation + cc.e_t

    base = np.array([c for _, c in geom], dtype=float)
    h = 1e-4
    g_fd = np.zeros_like(base)
    for a in range(base.shape[0]):
        for k in range(3):
            p = base.copy()
            p[a, k] += h
            m = base.copy()
            m[a, k] -= h
            g_fd[a, k] = (mol_e(p) - mol_e(m)) / (2 * h)
    assert np.max(np.abs(g_ccm - g_fd)) < 1e-6


@pytest.mark.slow
def test_ccsd_t_analytic_matches_numerical_h2_chain():
    """(3,1,1) H₂ chain with (T): the periodic CCSD(T) gradient gates at the
    O(h²) FD floor (the symmetric case already gated on the canonical route:
    4.3e-6 → 1.05e-6 → 2.5e-7; the non-canonical functional must preserve it)."""
    from types import SimpleNamespace

    from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd

    def _ccsdt_tight(ccm):
        scf = run_ccm_rhf(ccm, method="aiccm2026dev-a", conv_tol=1e-12,
                          max_iter=512)
        r = run_ccm_ccsd(ccm, scf, method="aiccm2026dev-a",
                         compute_triples=True, conv_tol=1e-11)
        return SimpleNamespace(energy=r.e_total)

    ccm = CCMSystem(_h2_chain(), (3, 1, 1), "sto-3g")
    g = run_ccm_ccsd_gradient(ccm, compute_triples=True)
    assert np.allclose(g.sum(axis=0), 0.0, atol=1e-10)
    e_big = np.max(np.abs(g - ccm_numerical_gradient(ccm, runner=_ccsdt_tight, h=1e-3)))
    e_small = np.max(np.abs(g - ccm_numerical_gradient(ccm, runner=_ccsdt_tight, h=5e-4)))
    assert e_big < 5e-6
    assert e_small < e_big / 3.0


def test_ccsd_isolated_limit_matches_molecular_ccsd_fd_heteronuclear():
    """LiH (1,1,1): CCSD analytic gradient == FD of the *molecular* C++ CCSD.

    The heteronuclear regression for the W^Δ Z-vector constraint term (Eq.
    1.9c'/§1.9.3 erratum): without ¼C(Fz+Fzᵀ)Cᵀ the analytic gradient carries
    an h-independent 1.75e-5 error here while every homonuclear H₂-chain gate
    stays green (z is small by symmetry). Fixed 2026-07-17 to the FD floor."""
    from vibeqc import CCSDOptions, run_ccsd

    geom = [(3, [0.0, 0.0, 0.0]), (1, [3.0, 0.0, 0.0])]
    iso = PeriodicSystem(3, np.diag([80.0] * 3), [Atom(z, p) for z, p in geom])
    ccm = CCMSystem(iso, (1, 1, 1), "sto-3g")
    g_ccm = run_ccm_ccsd_gradient(ccm)
    assert np.allclose(g_ccm.sum(axis=0), 0.0, atol=1e-10)

    def mol_e(coords):
        mol = Molecule([Atom(z, list(c)) for (z, _), c in zip(geom, coords)], 0, 1)
        b = BasisSet(mol, "sto-3g")
        o = RHFOptions()
        o.conv_tol_energy = 1e-13
        r = run_rhf(mol, b, o)
        co = CCSDOptions()
        co.compute_triples = False
        co.density_fit = False
        cc = run_ccsd(mol, b, r, co)
        return r.energy + cc.e_ccsd_correlation

    base = np.array([c for _, c in geom], dtype=float)
    h = 1e-3
    g_fd = np.zeros_like(base)
    for a in range(base.shape[0]):
        for k in range(3):
            p = base.copy()
            p[a, k] += h
            m = base.copy()
            m[a, k] -= h
            g_fd[a, k] = (mol_e(p) - mol_e(m)) / (2 * h)
    assert np.max(np.abs(g_ccm - g_fd)) < 1e-6


def test_uccsd_t_closed_shell_reduction_matches_ccsd_t():
    """UCCSD(T) gradient on a closed-shell cluster == the CCSD(T) gradient at
    machine ε — pins the spin-resolved (T) assembly (the spin-agnostic
    non-canonical (T) functional + per-spin Fz constraint terms). Measured
    1.6e-13 on (3,1,1); the (2,1,1) chain is unusable here (symmetry-broken
    branch pair splits the RHF/UHF references)."""
    ccm = CCMSystem(_h2_chain(), (3, 1, 1), "sto-3g")
    g_u = run_ccm_uccsd_gradient(ccm, compute_triples=True)
    g_c = run_ccm_ccsd_gradient(ccm, compute_triples=True)
    assert np.max(np.abs(g_u - g_c)) < 1e-11
    assert np.allclose(g_u.sum(axis=0), 0.0, atol=1e-10)


@pytest.mark.slow
def test_uccsd_t_analytic_matches_numerical_heh_chain():
    """Open-shell heteronuclear (T) FD gate: HeH doublet chain (3,1,1), the
    analytic UCCSD(T) gradient vs the FD of the bare-four-center UHF-CCSD(T)
    energy (built from the same internals the gradient differentiates), at the
    O(h²) floor: measured 4.9e-6 → 1.2e-6 for h = 2e-3 → 1e-3 (ratio 4.0),
    sum rule 1.6e-14. Exercises BOTH 2026-07-17 landings open-shell:
    the per-spin Fz constraint W-term and the non-canonical (T) functional."""
    from types import SimpleNamespace

    from vibeqc.periodic.ccm.ccsd import _ccsd_energy_so
    from vibeqc.periodic.ccm.gradient_analytic import (
        _build_uhf_spinorbital,
        _ccsd_solve_amplitudes,
        _ccsd_t_energy_so,
    )
    from vibeqc.periodic.ccm.scf import _ccm_eri_for_method

    def _uccsdt_energy(c):
        scf = run_ccm_uhf(c, method="aiccm2026dev-a", conv_tol=1e-12,
                          conv_tol_grad=1e-10, max_iter=512)
        eri = _ccm_eri_for_method(c, "aiccm2026dev-a")
        Ca = np.asarray(scf.mo_coeffs_alpha, float)
        Cb = np.asarray(scf.mo_coeffs_beta, float)
        ea = np.asarray(scf.mo_energies_alpha, float)
        eb = np.asarray(scf.mo_energies_beta, float)
        na, nb = int(scf.n_alpha), int(scf.n_beta)
        g_aa = np.einsum("mnls,mp,nq,lr,st->pqrt", eri, Ca, Ca, Ca, Ca, optimize=True)
        g_bb = np.einsum("mnls,mp,nq,lr,st->pqrt", eri, Cb, Cb, Cb, Cb, optimize=True)
        g_ab = np.einsum("mnls,mp,nq,lr,st->pqrt", eri, Ca, Ca, Cb, Cb, optimize=True)
        fs, spinints, nocc_so = _build_uhf_spinorbital(
            np.diag(ea), np.diag(eb), g_aa, g_bb, g_ab, na, nb)
        ts, td = _ccsd_solve_amplitudes(fs, spinints, nocc_so, tol=1e-13)
        e_corr = _ccsd_energy_so(ts, td, fs, spinints, nocc_so)
        e_t = float(np.real(_ccsd_t_energy_so(ts, td, fs, spinints, nocc_so)))
        return SimpleNamespace(energy=scf.energy + e_corr + e_t)

    ccm = CCMSystem(_heh_chain(), (3, 1, 1), "sto-3g")
    g = run_ccm_uccsd_gradient(ccm, compute_triples=True)
    assert np.allclose(g.sum(axis=0), 0.0, atol=1e-10)
    e_big = np.max(np.abs(g - ccm_numerical_gradient(ccm, runner=_uccsdt_energy, h=2e-3)))
    e_small = np.max(np.abs(g - ccm_numerical_gradient(ccm, runner=_uccsdt_energy, h=1e-3)))
    assert e_big < 2e-5
    assert e_small < e_big / 3.0
