"""EXPERIMENTAL: screened (HSE-class) hybrids on the direct-torus route.

Gates for the erfc-attenuated neutral-cderi sibling (``omega_screen`` on
``build_lpq_bloch_native_fft`` / ``ccm_neutral_cderi{,_fold}``) and its
consumption by ``run_ccm_rks_direct`` / ``run_ccm_uks_direct`` (2026-07-17,
real-Γ development chat). Everything here is **experimental research
surface** (manual research lanes only, per ``scripts/test_gate`` policy):
the SCF method matrix (RHF/UHF/RKS/UKS with pure + global hybrids) is the
landed surface; screened hybrids are the 2a extension.

The convention (recorded, load-bearing):

* The SR exchange operator is assembled **directly** from the screened
  kernel (the CRYSTAL-style treatment recorded in
  ``vibeqc.periodic_screened_exchange``), NOT as full-minus-long-range:

      K_sr(D) = K[L_sr](D) + sigma_0 . S D S,
      sigma_0 = pi / (omega_screen^2 . V_sc)

  with ``L_sr`` the erfc-attenuated fitted cderi (its ``G+q=0`` point is
  excluded from the fit; the screened kernel's zero-mode weight is FINITE
  and restored analytically -- no Madelung / exxdiv seam attaches, and the
  runs are ``exxdiv``-independent by construction, eta = 0 in the
  aiccm2026dev-b binding contract).

* **Cross-code caveat (measured 2026-07-17, PySCF 2.13.1 out of process):**
  PySCF KRKS(HSE06, GDF) assembles full-minus-LR with exxdiv coupling. On
  rocksalt LiH/STO-3G at (1,1,1) the LR (erf) kernel has NO reciprocal
  support (first shell exp(-G^2/(4 w^2)) ~ e^-27), so PySCF@ewald
  degenerates to its PBE0 value (-8.291365331286 vs PBE0
  -8.291224931283) and its two exxdiv settings differ from each other by
  0.235 Ha (exxdiv=None: -8.056479159951). The direct route gives the
  exxdiv-independent SR-direct value (-9.179939311641; the sigma_0
  self-image term is large at omega*L_BvK ~ 0.6). These are *different
  finite-size conventions of the same TDL*; do not compare screened-hybrid
  numbers across codes without matching the assembly convention. The fair
  external comparison lives at meshes with omega*L_BvK >~ 1 (vq ladder).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, Molecule, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import (
    ccm_direct_oneelectron,
    run_ccm_rhf_direct,
    run_ccm_rks_direct,
    run_ccm_uks_direct,
)
from vibeqc.periodic.ccm.neutral import (
    ccm_neutral_cderi,
    ccm_neutral_cderi_fold,
    ccm_sr_exchange_zero_mode_constant,
    _supercell_wrap_lat_opts,
)
from vibeqc.periodic.ccm.ri import ccm_ri_k_neutral

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

OMEGA_HSE = 0.11  # HSE06 screening (Krukau et al., JCP 125, 224106 (2006))


def _h2_cubic_ccm(nrep=(2, 1, 1)):
    """Compact 3-D H₂ cell (6-bohr cube) — the cheap dim=3 workhorse."""
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _h2_box_pair():
    """Isolated H₂ (20-bohr box) as (CCMSystem, Molecule) for the vacuum limit."""
    atoms = [Atom(1, [10.0, 10.0, 9.3]), Atom(1, [10.0, 10.0, 10.7])]
    cell = PeriodicSystem(3, np.diag([20.0, 20.0, 20.0]), atoms, 0, 1)
    return CCMSystem(cell, (1, 1, 1), "sto-3g"), Molecule(atoms, 0, 1)


def _scf_density_and_overlap(ccm):
    scf = run_ccm_rhf_direct(ccm)
    S, _, _ = ccm_direct_oneelectron(ccm)
    return scf.density, S


# --- 1. Kernel-level ground truth: fit == direct G-sum ----------------------


def test_sr_cderi_matches_direct_g_sum_four_center():
    """The erfc-attenuated fitted four-center equals the direct reciprocal-sum
    ground truth (same mesh, same Bloch pair FT, no RI) to the RI floor.

    Measured 2026-07-17 on compact H₂ (2,1,1): 1.4e-4 four-center max /
    6.4e-5 on K. This is the decisive internal identity: it pins the kernel
    attenuation at every |G+q|, independent of any external comparator. The
    sigma_0 zero mode cancels between the two sides by construction, so this
    gate pins the G != 0 content; sigma_0 is pinned by the vacuum-box
    real-space gate and the vacuum-limit SCF gate below.
    """
    from vibeqc._aopair_ft import ao_pair_fourier_transform_bloch
    from vibeqc._vibeqc_core import direct_lattice_cells
    from vibeqc.aux_basis import (
        _ao_scales_for_rsgdf,
        _rsgdf_shifted_dense_g_mesh,
    )

    ccm = _h2_cubic_ccm((2, 1, 1))
    sysc = ccm.cluster_system
    V = float(abs(np.linalg.det(np.asarray(sysc.lattice, float))))
    omega = OMEGA_HSE

    opts = _supercell_wrap_lat_opts(ccm)
    Gq = _rsgdf_shifted_dense_g_mesh(sysc, np.zeros(3), 200.0)
    Gq2 = (Gq ** 2).sum(axis=1)
    nz = Gq2 > 1e-12
    Gq, Gq2 = Gq[nz], Gq2[nz]
    # FT[erfc(w r)/r](p) = (4 pi / p^2)(1 - exp(-p^2 / (4 w^2)))
    w = (4.0 * np.pi) / Gq2 / V * (-np.expm1(-Gq2 / (4 * omega * omega)))

    cells = direct_lattice_cells(sysc, float(opts.cutoff_bohr))
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    scales = _ao_scales_for_rsgdf(ccm.basis)
    pair_scales = np.outer(scales, scales)

    n = int(ccm.basis.nbasis)
    g_exact = np.zeros((n, n, n, n))
    chunk = 20000
    for lo in range(0, Gq.shape[0], chunk):
        pf = ao_pair_fourier_transform_bloch(
            ccm.basis, Gq[lo:lo + chunk], R_g, k_cart=np.zeros(3))
        pf = pf * pair_scales[:, :, None]
        g_exact += np.real(np.einsum(
            "mnk,k,rsk->mnrs", pf.conj(), w[lo:lo + chunk], pf,
            optimize=True))

    L_sr = ccm_neutral_cderi(ccm, omega_screen=omega)
    g_fit = np.einsum("Pmn,Prs->mnrs", L_sr, L_sr, optimize=True)
    assert np.max(np.abs(g_fit - g_exact)) < 1e-3


def test_sr_zero_mode_pinned_by_vacuum_box_real_space_k():
    """sigma_0 is load-bearing and correctly scaled: in the vacuum box the
    real-space erfc K (libint erfc integrals, image-summed) matches the
    fitted K only WITH the analytic zero mode.

    Measured 2026-07-17: with sigma_0 S D S the agreement is 1.2e-3 (image
    truncation at the default cell list); without it the gap is 0.053 ==
    max|sigma_0 S D S| — 40x the floor. NOTE this comparator is only used in
    the vacuum box: on compact multi-cell toruses
    ``build_jk_gamma_molecular_limit`` computes a different (molecular-limit)
    operator — a cutoff-independent O(1) gap, flagged to the periodic-SCF
    chat 2026-07-17 — while the fit is pinned by the G-sum gate above.
    """
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        build_jk_gamma_molecular_limit,
    )

    ccm, _ = _h2_box_pair()
    D, S = _scf_density_and_overlap(ccm)
    omega = OMEGA_HSE

    L_sr = ccm_neutral_cderi(ccm, omega_screen=omega)
    sigma0 = ccm_sr_exchange_zero_mode_constant(ccm, omega)
    K_fit_nz = ccm_ri_k_neutral(L_sr, D)
    K_fit = K_fit_nz + sigma0 * (S @ D @ S)

    K_rs = np.asarray(build_jk_gamma_molecular_limit(
        ccm.basis, ccm.cluster_system, LatticeSumOptions(), D, omega).K)

    assert np.max(np.abs(K_rs - K_fit)) < 5e-3
    # ... and the zero mode is what makes it match (not a loose tolerance):
    assert np.max(np.abs(K_rs - K_fit_nz)) > 10 * np.max(np.abs(K_rs - K_fit))


def test_sr_cderi_omega_to_zero_recovers_coulomb_contraction():
    """omega_screen -> 0 recovers the full-Coulomb neutral cderi at
    contraction level (the attenuation factor -> 1 on every mesh point;
    measured drift 0.0 at omega = 1e-3 on compact H₂ (2,1,1))."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    D, _ = _scf_density_and_overlap(ccm)
    K0 = ccm_ri_k_neutral(ccm_neutral_cderi(ccm), D)
    Keps = ccm_ri_k_neutral(ccm_neutral_cderi(ccm, omega_screen=1e-3), D)
    assert np.max(np.abs(K0 - Keps)) < 1e-10


def test_sr_fold_matches_supercell_build():
    """The fold-built SR cderi equals the supercell-fit SR cderi at
    contraction level (the attenuation is a function of |G+q| on the
    unfolded supercell reciprocal set, so the fold identity is untouched).
    Measured 2026-07-17: 3.2e-11 on K, 2.1e-10 on the HSE06 SCF."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    D, _ = _scf_density_and_overlap(ccm)
    Ks = ccm_ri_k_neutral(ccm_neutral_cderi(ccm, omega_screen=OMEGA_HSE), D)
    Kf = ccm_ri_k_neutral(
        ccm_neutral_cderi_fold(ccm, omega_screen=OMEGA_HSE), D)
    assert np.max(np.abs(Ks - Kf)) < 1e-8


# --- 2. SCF-level gates ------------------------------------------------------


@pytest.mark.slow
def test_hse06_direct_vacuum_limit_matches_molecular():
    """Equivalence-theorem vacuum limit: direct-route HSE06 on isolated H₂
    (20-bohr box) reproduces molecular ``run_rks`` HSE06. The sigma_0 energy
    term is -0.0065 Ha here, 30x the gate — this pins the zero mode at SCF
    level. Measured 2026-07-17: -2.1e-4 (periodic images + grid partition,
    the same class as the PBE vacuum-limit residual)."""
    from vibeqc import RKSOptions, make_basis, run_rks

    ccm, mol = _h2_box_pair()
    r = run_ccm_rks_direct(ccm, "hse06")
    opts = RKSOptions()
    opts.functional = "hse06"
    m = run_rks(mol, make_basis(mol, "sto-3g"), opts)
    assert r.converged and m.converged
    assert r.energy / ccm.n_cells == pytest.approx(m.energy, abs=5e-4)


def test_hse06_direct_is_exxdiv_independent():
    """No exchange-q=0 seam attaches to the screened kernel (finite zero
    mode), so the two declared conventions give the identical Hamiltonian —
    measured bit-identical 2026-07-17. This is the b-contract's eta = 0,
    and the sharp behavioural difference from global hybrids (PBE0 moves by
    exactly a_x ξ_N N_e/2)."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    r_ew = run_ccm_rks_direct(ccm, "hse06", exxdiv="ewald")
    r_no = run_ccm_rks_direct(ccm, "hse06", exxdiv=None)
    assert r_ew.converged and r_no.converged
    assert abs(r_ew.energy - r_no.energy) < 1e-10
    assert r_ew.exchange_q0_applicability == "inactive"
    assert r_no.exchange_q0_applicability == "inactive"


def test_hse06_uks_collapses_to_rks():
    """Closed-shell UKS-HSE06 == RKS-HSE06 (per-spin screened exchange
    collapses; measured 8.9e-16)."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    r = run_ccm_rks_direct(ccm, "hse06")
    u = run_ccm_uks_direct(ccm, "hse06")
    assert r.converged and u.converged
    assert abs(r.energy - u.energy) < 1e-10
    assert u.exchange_q0_applicability == "inactive"


def test_exchange_q0_applicability_records():
    """Applicability semantics across the functional classes: pure and
    screened-only carry no seam ("inactive"); a global hybrid's full-range
    arm makes the convention material ("active")."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    assert run_ccm_rks_direct(
        ccm, "pbe").exchange_q0_applicability == "inactive"
    assert run_ccm_rks_direct(
        ccm, "pbe0").exchange_q0_applicability == "active"
    assert run_ccm_rks_direct(
        ccm, "hse06").exchange_q0_applicability == "inactive"


# --- 3. Fail-closed boundary -------------------------------------------------


@pytest.mark.parametrize("functional", ["wb97x", "cam-b3lyp"])
@pytest.mark.parametrize("runner", [run_ccm_rks_direct, run_ccm_uks_direct])
def test_full_range_arm_rsh_still_fails_closed(runner, functional):
    """Range-separated functionals with a nonzero full-range exact-exchange
    arm keep failing closed (the shared resolver policy: their K_full needs
    an exxdiv treatment not validated on periodic routes)."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    with pytest.raises(NotImplementedError, match="full-range"):
        runner(ccm, functional)


def test_sr_zero_mode_constant_rejects_full_range():
    """sigma_0 is a screened-kernel object; omega = 0 (full Coulomb) has a
    divergent zero mode handled by the seam, not this constant."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    with pytest.raises(ValueError, match="omega_screen"):
        ccm_sr_exchange_zero_mode_constant(ccm, 0.0)
