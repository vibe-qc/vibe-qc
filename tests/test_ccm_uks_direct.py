"""Open-shell KS-DFT on the direct-torus (real Γ-supercell) route — gates.

Gates for :func:`vibeqc.periodic.ccm.direct.run_ccm_uks_direct` (2026-07-16),
which completes the direct-route method matrix (RHF/UHF/RKS/UKS). The Fock is

    F_σ = h + J(D_a + D_b) + V_xc^σ − a_x [ K(D_σ) + ξ_N S D_σ S ]

— :func:`run_ccm_uhf_direct`'s per-spin exchange-q=0 seam composed with
:func:`run_ccm_rks_direct`'s ``a_x`` scaling and the spin-polarized periodic
XC kernel (``build_xc_periodic_uks`` on the same fold construction / periodic
Becke grid as the closed-shell route — the b3f74aa9-validated pairing).

The sharp gates:

1. **Closed-shell collapse** (the load-bearing one): UKS-direct == RKS-direct
   on a closed-shell cell, pure and hybrid (measured 0.0 / 1.8e-15) — so the
   open-shell route inherits every closed-shell parity gate (GDF, external
   PySCF KRKS, vacuum limit) transitively.
2. **Open-shell hybrid seam identity**: the ``exxdiv=None`` control sits above
   ``"ewald"`` by exactly ``a_x·ξ_N·N_e/2`` per supercell with the same
   per-spin densities (measured 6e-16).
3. **Open-shell direct-vs-GDF pure parity**: == the production multi-k KUKS
   (:func:`run_ccm_uks_gdf` → ``run_kuks_periodic_gdf``) at (1,1,1), where the
   spin conventions match by construction (measured 3.6e-13 — both routes ride
   the identical ``build_xc_periodic_uks`` + fit machinery).
4. **Vacuum limit**: an isolated H-atom doublet reproduces molecular
   ``run_uks`` per rung.
5. **Fail-loudly surface** + shell-aware route dispatch (the even-electron
   triplet no longer runs silently as closed shell on the real-gamma route).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import Atom, Molecule, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import (
    ccm_exchange_q0_madelung,
    run_ccm_rks_direct,
    run_ccm_uhf_direct,
    run_ccm_uks_direct,
)
from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi
from vibeqc.periodic.ccm.route import run_ccm_scf
from vibeqc.periodic.exchange_convention import BVK_EWALD, STRICT_ZERO

pytestmark = pytest.mark.experimental  # neutral finite-BvK-torus research lane


def _h2_cubic_ccm(nrep=(2, 1, 1)):
    """Compact 3-D H₂ cell (6-bohr cube) — the cheap closed-shell workhorse."""
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _li_cubic_ccm(nrep=(1, 1, 1), a=8.0):
    """Compact 3-D open-shell control: Li atom (3e doublet) in an a-bohr cube."""
    cell = PeriodicSystem(3, np.diag([a, a, a]), [Atom(3, [a / 2, a / 2, a / 2])],
                          0, 2)
    return CCMSystem(cell, nrep, "sto-3g")


def _h_triplet_ccm():
    """Even-electron OPEN-shell control: triplet H₂ (2e, multiplicity 3) —
    the fixture class the closed-shell loops used to swallow silently."""
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 3)
    return CCMSystem(cell, (1, 1, 1), "sto-3g")


# --- 1. Closed-shell collapse -------------------------------------------------


@pytest.mark.parametrize("functional", ["svwn", "pbe", "r2scan", "pbe0"])
def test_uks_direct_closed_shell_collapses_to_rks_direct(functional):
    """The load-bearing gate: with D_a = D_b = D/2 the per-spin seam and the
    a_x-scaled exchange collapse to the closed-shell −(a_x/2)[K + ξ_N S D S],
    and build_xc_periodic_uks(D/2, D/2) == build_xc_periodic(D), so UKS-direct
    must equal RKS-direct exactly. RKS-direct is gated against the multi-k GDF
    and external PySCF KRKS (test_ccm_rks_direct.py), so this collapse hands
    those parities to the open-shell route transitively."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    L = ccm_neutral_cderi(ccm)
    r = run_ccm_rks_direct(ccm, functional, cderi=L)
    u = run_ccm_uks_direct(ccm, functional, cderi=L)
    assert r.converged and u.converged
    assert u.open_shell is True and r.open_shell is False
    assert u.exchange_q0 == r.exchange_q0 == BVK_EWALD
    assert u.energy == pytest.approx(r.energy, abs=1e-9)
    assert np.max(np.abs(u.density_alpha - u.density_beta)) < 1e-8
    assert np.max(np.abs(u.density - r.density)) < 1e-8


# --- 2. Open-shell hybrid seam identity ---------------------------------------


def test_uks_direct_seam_off_offset_scales_with_ax_open_shell():
    """Open-shell hybrid seam identity: exxdiv=None sits above "ewald" by
    exactly a_x·ξ_N·N_e/2 per supercell with the SAME per-spin densities — the
    per-spin seam is occ/virt block-diagonal in each spin at convergence, and
    it enters only the exact-exchange channel (scaled by a_x). The open-shell
    twin of test_ccm_rks_direct.py's a_x identity."""
    ccm = _li_cubic_ccm()
    L = ccm_neutral_cderi(ccm)
    # The density pins need conv_tol_grad (the DIIS-residual criterion), not
    # just conv_tol: the KS energy is stationary, so |dE| is quadratic in the
    # density error and an energy-only gate leaves the minority-spin density
    # ~1e-5-loose nondeterministically (threaded-BLAS trajectory) — the
    # review-confirmed flake this knob was added for.
    r_on = run_ccm_uks_direct(ccm, "pbe0", cderi=L, exxdiv="ewald",
                              conv_tol=1e-10, conv_tol_grad=1e-8)
    r_off = run_ccm_uks_direct(ccm, "pbe0", cderi=L, exxdiv=None,
                               conv_tol=1e-10, conv_tol_grad=1e-8)
    assert r_on.converged and r_off.converged
    assert r_on.exchange_q0 == BVK_EWALD
    assert r_off.exchange_q0 == STRICT_ZERO

    xi = ccm_exchange_q0_madelung(ccm)
    n_e = ccm.supercell.n_electrons()
    a_x = 0.25  # PBE0 exact-exchange fraction (Adamo & Barone 1999)
    assert (r_off.energy - r_on.energy) == pytest.approx(
        a_x * xi * n_e / 2.0, abs=1e-8)
    # Same-state pins at 1e-5: conv_tol_grad=1e-8 bounds each run's density
    # error structurally at ~err/gap ≈ 1e-7 (measured across repeated runs:
    # dD_alpha ~1e-14, dD_beta 3e-12..1.2e-8 — the minority spin wanders with
    # the threaded-BLAS DIIS trajectory, which is why the pin carries margin
    # rather than asserting the instrumented best case).
    assert np.max(np.abs(r_off.density_alpha - r_on.density_alpha)) < 1e-5
    assert np.max(np.abs(r_off.density_beta - r_on.density_beta)) < 1e-5


# --- 3. Open-shell direct-vs-GDF parity ---------------------------------------


def test_uks_direct_matches_kuks_gdf_pure_gamma():
    """Open-shell pure-KS parity: direct == the production multi-k KUKS
    (run_ccm_uks_gdf → run_kuks_periodic_gdf) on the Li-box doublet at (1,1,1),
    where the per-unit-cell spin convention of the multi-k driver coincides
    with the CCM supercell multiplicity. Both sides evaluate XC through the
    identical build_xc_periodic_uks + periodic-Becke machinery, so the measured
    delta is 3.6e-13; gate at 1e-8, the HF-gate class. (Multi-cell open-shell
    parity is convention-laden — see run_ccm_uks_gdf's spin-bookkeeping
    warning — so this gate stays at (1,1,1) deliberately.)"""
    from vibeqc.periodic.ccm.ri import run_ccm_uks_gdf

    ccm = _li_cubic_ccm()
    u = run_ccm_uks_direct(ccm, "pbe")
    g = run_ccm_uks_gdf(ccm, "pbe")
    assert u.converged and g.converged
    assert u.energy / ccm.n_cells == pytest.approx(g.energy, abs=1e-8)


# --- 4. Vacuum limit ----------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["svwn", "pbe", "r2scan"])
def test_uks_direct_vacuum_limit_matches_molecular_uks(functional):
    """Equivalence theorem at N_c = 1 + vacuum: an isolated H-atom doublet in a
    20-bohr box reproduces molecular run_uks, per XC rung (LDA / GGA /
    meta-GGA — the sibling parametrization of the closed-shell vacuum
    gates)."""
    from vibeqc import BasisSet, Molecule, UKSOptions, run_uks

    a = 20.0
    cell = PeriodicSystem(3, np.diag([a, a, a]), [Atom(1, [a / 2] * 3)], 0, 2)
    ccm = CCMSystem(cell, (1, 1, 1), "sto-3g")
    u = run_ccm_uks_direct(ccm, functional)

    mol = Molecule([Atom(1, [a / 2] * 3)], multiplicity=2)
    opts = UKSOptions()
    opts.functional = functional
    ref = run_uks(mol, BasisSet(mol, "sto-3g"), opts)
    assert u.converged and ref.converged
    # Vacuum-limit tolerance: the periodic route still carries the finite
    # box's residual image terms + a different quadrature; measured
    # 1.47e-4 for the diffuse H-atom doublet at this box size (larger than
    # the closed-shell H₂ gates — a single H's density reaches further into
    # the images). Gate at 5e-4.
    assert u.energy == pytest.approx(ref.energy, abs=5e-4)


# --- 5. Fail-loudly surface + shell-aware dispatch ------------------------------


def test_uks_direct_fail_loudly_surface():
    ccm = _li_cubic_ccm()
    # Superseded 2026-07-17: HSE-class SR-only screened hybrids now run
    # (experimental; tests/test_ccm_rsh_direct.py). The fail-closed surface
    # is the nonzero full-range exact-exchange arm, per the shared resolver.
    with pytest.raises(NotImplementedError, match="full-range"):
        run_ccm_uks_direct(ccm, "wb97x")
    with pytest.raises(ValueError, match="exxdiv"):
        run_ccm_uks_direct(ccm, "pbe", exxdiv="bogus")
    with pytest.raises(ValueError, match="cderi_build"):
        run_ccm_uks_direct(ccm, "pbe", cderi_build="bogus")
    lowd = CCMSystem(
        PeriodicSystem(1, np.diag([6.0, 15.0, 15.0]),
                       [Atom(1, [0.0, 7.5, 7.5]), Atom(1, [1.4, 7.5, 7.5])],
                       0, 1), (2, 1, 1), "sto-3g")
    with pytest.raises(NotImplementedError, match="dim == 1"):
        run_ccm_uks_direct(lowd, "pbe")


def test_route_real_gamma_dispatches_by_shell():
    """run_ccm_scf(route='real-gamma') reaches the UHF/UKS direct drivers for
    open-shell clusters (2026-07-16). Load-bearing detail: an EVEN-electron
    open-shell cluster (triplet H₂) previously fell through to the
    closed-shell loop SILENTLY (n_occ = n_elec // 2 ignores multiplicity) and
    returned a wrong closed-shell number; it now reaches the spin-polarized
    routes, keyed on multiplicity AND electron parity."""
    # Odd-electron doublet → UHF/UKS direct.
    li = _li_cubic_ccm()
    hf = run_ccm_scf(li, route="real-gamma")
    ks = run_ccm_scf(li, route="real-gamma", functional="pbe")
    assert hf.n_alpha == 2 and hf.n_beta == 1          # CCMUHFResult
    assert ks.open_shell is True                        # CCMKSResult, UKS path
    assert hf.exchange_q0 == ks.exchange_q0 == BVK_EWALD

    # Even-electron triplet → still the open-shell routes (the former trap).
    trip = _h_triplet_ccm()
    hf3 = run_ccm_scf(trip, route="real-gamma")
    ks3 = run_ccm_scf(trip, route="real-gamma", functional="pbe")
    assert (hf3.n_alpha, hf3.n_beta) == (2, 0)
    assert ks3.open_shell is True
    assert np.max(np.abs(ks3.density_alpha - ks3.density_beta)) > 1e-3

    # Closed shell keeps the closed-shell drivers (no behaviour change).
    h2 = _h2_cubic_ccm((1, 1, 1))
    r = run_ccm_scf(h2, route="real-gamma", functional="pbe")
    assert r.open_shell is False


def test_uks_direct_open_shell_doublet_state():
    """The converged doublet has the exact spin counts, a genuinely
    spin-polarized density, and per-spin fields filled."""
    ccm = _li_cubic_ccm()
    u = run_ccm_uks_direct(ccm, "pbe")
    assert u.converged and u.open_shell
    assert u.density_alpha is not None and u.density_beta is not None
    assert u.mo_energies_beta is not None and u.mo_coeffs_beta is not None
    assert u.fock_beta is not None
    # 2 alpha / 1 beta electrons on the torus.
    S = u.overlap
    assert np.trace(u.density_alpha @ S) == pytest.approx(2.0, abs=1e-8)
    assert np.trace(u.density_beta @ S) == pytest.approx(1.0, abs=1e-8)
    assert np.max(np.abs(u.density_alpha - u.density_beta)) > 1e-2
    for fock, coeffs, energies in (
        (u.fock, u.mo_coeffs, u.mo_energies),
        (u.fock_beta, u.mo_coeffs_beta, u.mo_energies_beta),
    ):
        residual = fock @ coeffs - (S @ coeffs) * energies[None, :]
        assert np.max(np.abs(residual)) < 5e-9


def test_uks_direct_terminal_result_is_one_physical_density_generation(
    monkeypatch,
):
    """The terminal UKS record is physical, spin-complete, and canonical."""
    from vibeqc.periodic.ccm import direct as direct_module

    ccm = SimpleNamespace(
        unit_system=SimpleNamespace(dim=3),
        supercell=Molecule([Atom(3, [0., 0., 0.])], 0, 2),
        n_atoms=1,
        nrep=(1, 1, 1),
    )
    overlap = np.eye(3)
    hcore = np.diag([-1.2, -0.4, 0.6])
    e_nuclear = 0.15

    def j_of_density(_cderi, density):
        return np.array(
            [
                [0.10 * density[0, 0], 0.08 * density[0, 0], 0.0],
                [0.08 * density[0, 0], 0.07 * density[1, 1],
                 0.06 * density[1, 1]],
                [0.0, 0.06 * density[1, 1], 0.05 * density[2, 2]],
            ]
        )

    def xc_of_spins(density_alpha, density_beta):
        e_xc = 0.02 * float(
            np.sum(density_alpha**2) + np.sum(density_beta**2)
        ) + 0.01 * float(np.sum(density_alpha * density_beta))
        potential_alpha = 0.04 * density_alpha + 0.01 * density_beta
        potential_beta = 0.04 * density_beta + 0.01 * density_alpha
        return e_xc, potential_alpha, potential_beta

    monkeypatch.setattr(
        direct_module, "_reject_vacuum_padded_direct", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(direct_module, "_warn_experimental", lambda: None)
    monkeypatch.setattr(
        direct_module,
        "ccm_direct_oneelectron",
        lambda *args, **kwargs: (overlap, hcore, e_nuclear),
    )
    monkeypatch.setattr(direct_module, "ccm_ri_j_neutral", j_of_density)
    monkeypatch.setattr(
        direct_module,
        "_real_gamma_uks_xc_builder",
        lambda *args, **kwargs: xc_of_spins,
    )

    result = run_ccm_uks_direct(
        ccm,
        "pbe",
        cderi=np.zeros((1, 3, 3)),
        max_iter=1,
        conv_tol=1e-12,
        conv_tol_grad=1e-12,
        lindep_tol=1e-12,
    )
    expected_alpha = np.diag([1.0, 1.0, 0.0])
    expected_beta = np.diag([1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.density_alpha, expected_alpha, atol=1e-14)
    np.testing.assert_allclose(result.density_beta, expected_beta, atol=1e-14)
    np.testing.assert_allclose(
        result.density, result.density_alpha + result.density_beta, atol=1e-14
    )

    J = j_of_density(None, result.density)
    e_xc, V_alpha, V_beta = xc_of_spins(
        result.density_alpha, result.density_beta
    )
    physical_alpha = 0.5 * (
        hcore + J + V_alpha + (hcore + J + V_alpha).T
    )
    physical_beta = 0.5 * (
        hcore + J + V_beta + (hcore + J + V_beta).T
    )
    e_coulomb = 0.5 * float(np.sum(result.density * J))
    energy = (
        float(np.sum(result.density * hcore))
        + e_coulomb
        + e_xc
        + e_nuclear
    )
    np.testing.assert_allclose(result.fock, physical_alpha, atol=1e-14)
    np.testing.assert_allclose(result.fock_beta, physical_beta, atol=1e-14)
    assert result.e_coulomb == pytest.approx(e_coulomb, abs=1e-14)
    assert result.e_xc == pytest.approx(e_xc, abs=1e-14)
    assert result.e_hf_exchange == pytest.approx(0.0, abs=1e-14)
    assert result.energy == pytest.approx(energy, abs=1e-14)

    for fock, coeffs, energies in (
        (result.fock, result.mo_coeffs, result.mo_energies),
        (result.fock_beta, result.mo_coeffs_beta, result.mo_energies_beta),
    ):
        residual = fock @ coeffs - (overlap @ coeffs) * energies[None, :]
        assert np.max(np.abs(residual)) < 1e-12
