"""Direct-torus (real Γ-supercell) neutral SCF — parity gates vs the multi-k GDF.

The D1 deliverable gates (:mod:`vibeqc.periodic.ccm.direct`):

* ``E(run_ccm_rhf_direct) − E(run_ccm_rhf_gdf) ≤ 1e-8`` Ha/cell on the 1-D
  H-chain, the vacuum-padded H₂ 3-D anchor (external KRHF
  ``−1.1182352381`` Ha/cell), and an ionic 3-D case (LiH) where the ξ_N seam
  is large.
* A deliberate ``exxdiv=None`` run reproduces the exchange-q0 offset
  *exactly* ``ξ_N·N_e/2`` per supercell — the seam term is what closes the
  documented gauge gap (the D2 finding), and it is block-diagonal in occ/virt
  so both conventions converge to the same density (the offset is an identity,
  not a first-order estimate).
* Every cross-route comparison asserts a matched exchange-``q=0`` convention
  (``assert_matched_exchange_q0``); mismatched labels must raise.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import (
    ccm_exchange_q0_madelung,
    run_ccm_rhf_direct,
    run_ccm_rhf_direct_rijcosx,
    run_ccm_uhf_direct,
)
from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf
from vibeqc.periodic.exchange_convention import (
    BVK_EWALD,
    STRICT_ZERO,
    assert_matched_exchange_q0,
)

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

# External multi-k KRHF (PySCF 2.13.1, out of process) on the H₂ anchor —
# docs/manuscripts/aiccm_comparison/data/h2_mp2_2026-06-21.json.
E_KRHF_H2_ANCHOR = -1.1182352381008094


def _h2_anchor_ccm():
    """The converged-theory anchor: H₂/STO-3G, 20×20×6 bohr, (1,1,2) mesh."""
    cell = PeriodicSystem(
        3, np.array([[20.0, 0, 0], [0, 20.0, 0], [0, 0, 6.0]]),
        [Atom(1, [10.0, 10.0, 2.3]), Atom(1, [10.0, 10.0, 3.7])], 0, 1)
    return CCMSystem(cell, (1, 1, 2), "sto-3g")


def _h_chain_ccm(nrep=(3, 1, 1)):
    """1-D H₂ chain, 6-bohr cell, 15-bohr transverse vacuum (dim=1)."""
    cell = PeriodicSystem(
        1, np.diag([6.0, 15.0, 15.0]),
        [Atom(1, [0.0, 7.5, 7.5]), Atom(1, [1.4, 7.5, 7.5])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _h2_cubic_ccm(nrep=(2, 1, 1)):
    """Compact 3-D H₂ cell (6-bohr cube) — the cheap dim=3 workhorse."""
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _padded_h_chain_ccm(nrep=(1, 1, 1)):
    """IID 291 repro class: a 1-D H₂ chain along z, declared 3-D with 40 bohr
    transverse vacuum -- the configuration that used to evade the declared-
    dimension gate and produce a diverging neutral energy ladder."""
    cell = PeriodicSystem(
        3, np.diag([40.0, 40.0, 2.8]),
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _lih_ccm(nrep=(1, 1, 1)):
    """Ionic 3-D control: rocksalt LiH, fcc primitive cell (a = 7.72 bohr)."""
    a = 7.72
    lat = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]).T
    cell = PeriodicSystem(
        3, lat, [Atom(3, [0, 0, 0]), Atom(1, [0.5 * a, 0.5 * a, 0.5 * a])],
        0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _diamond_ccm(nrep=(1, 1, 1), basis="sto-3g"):
    """TIGHT-CORE 3-D control: diamond, fcc primitive cell (a = 6.74 bohr).

    The cheap member of the class GitLab #307/#308 was about. Its C STO-3G
    ``zeta_max`` is 71.617, so ``10 x zeta_max = 716.2 Ha`` exceeds the 200 Ha
    default rsgdf mesh and the tight-core classifier
    (:func:`~vibeqc.pbc_gdf._gamma_dense_core_gdf_parity_held`) fires -- the
    same class as the calibrated P01 MgO/STO-3G cell (``zeta_max`` 299.237) at
    a fraction of the tailed-build cost (measured 2026-08-25: MgO's tailed
    fold cderi 403 s, diamond's 20 s).

    Every pre-existing control in this file is OUTSIDE that class -- H2/STO-3G
    ``zeta_max`` ~ 3.4, LiH/STO-3G 16.12 -- which is precisely why they all
    passed while MgO/STO-3G at ``nrep=(1,1,1)`` was off by 0.49 Ha/cell.
    """
    a = 6.74
    lat = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]).T
    cell = PeriodicSystem(
        3, lat, [Atom(6, [0, 0, 0]), Atom(6, [0.25 * a, 0.25 * a, 0.25 * a])],
        0, 1)
    return CCMSystem(cell, nrep, basis)


#: The tail ``ccm_neutral_tail_ke_cutoff`` resolves for ``_diamond_ccm``:
#: 1.1 x 10 x zeta_max = 1.1 x 10 x 71.6168373. Asserted against the live
#: resolver below rather than trusted, so a change to the ratio or the
#: classifier breaks HERE instead of silently retuning the parity gate.
DIAMOND_STO3G_AUTO_TAIL = 787.785207


def _li_cubic_ccm(nrep=(1, 1, 1), a=8.0):
    """Compact 3-D open-shell control: single Li atom (3e doublet) in an
    a-bohr cube. Converges cleanly with ⟨S²⟩ = 0.75 and a Madelung-large seam."""
    cell = PeriodicSystem(3, np.diag([a, a, a]), [Atom(3, [a / 2, a / 2, a / 2])],
                          0, 2)
    return CCMSystem(cell, nrep, "sto-3g")


def _assert_direct_gdf_parity(ccm, tol=1e-8):
    """Shared gate: direct-torus == multi-k GDF per unit cell at matched
    exchange-q=0 convention AND matched reciprocal support. Returns
    (direct, gdf) results."""
    r = run_ccm_rhf_direct(ccm)
    g = run_ccm_rhf_gdf(ccm)
    assert r.converged and g.converged
    # The GDF route is exxdiv='ewald' throughout (multi-k branch and the
    # dim=3 Γ fast-path both); assert before comparing energies.
    #
    # Read BOTH sides. Until 2026-08-28 this compared the DIRECT result's
    # label against the module constant BVK_EWALD and never read
    # ``g.exchange_q0`` at all, so its negative branch was unreachable:
    # ``_ccm_gdf`` sets ``exchange_q0=""`` whenever its ``active`` read is
    # false, and this assertion could not have noticed. A decoration, not a
    # gate (GitLab #307; LEARNINGS L95/L99 class).
    assert_matched_exchange_q0([r.exchange_q0, g.exchange_q0],
                               context="direct-vs-gdf")
    assert_matched_exchange_q0([r.exchange_q0, BVK_EWALD],
                               context="direct-vs-ewald-literal")
    # The RECIPROCAL-SUPPORT convention is the second half of "same
    # Hamiltonian" and is exactly what #307 turned on: the two routes only
    # evaluate the same operator once they resolved the same high-|G| tail.
    assert r.rsgdf_tail_ke_cutoff == g.rsgdf_tail_ke_cutoff, (
        "direct-vs-gdf: mismatched high-|G| tail completion -- "
        f"direct={r.rsgdf_tail_ke_cutoff!r} gdf={g.rsgdf_tail_ke_cutoff!r}; "
        "these are different Hamiltonians, not a tolerance question"
    )
    # A held absolute energy must never be compared as if it were converged.
    assert not r.parity_held and not g.parity_held
    assert r.normalization == "total_cyclic_cluster"
    assert r.total_cyclic_energy == pytest.approx(r.energy, abs=0.0)
    e_direct_cell = r.total_cyclic_energy / ccm.n_cells
    assert e_direct_cell == pytest.approx(g.energy, abs=tol)
    return r, g


@pytest.mark.slow
def test_direct_matches_gdf_h2_anchor_and_external_krhf():
    """Anchor gate: direct == GDF == external KRHF on H₂ 20×20×6 (1,1,2)."""
    ccm = _h2_anchor_ccm()
    r, _ = _assert_direct_gdf_parity(ccm, tol=1e-8)
    assert r.energy / ccm.n_cells == pytest.approx(E_KRHF_H2_ANCHOR, abs=1e-8)


def test_direct_and_gdf_h_chain_1d_both_fail_closed():
    """1-D gate (rewritten 2026-07-10). This used to assert ``direct == GDF`` on the
    H₂ chain. It passed because **both** sides shared the same broken ``dim < 3``
    Coulomb kernel: bare, conditionally-convergent V_ne/E_nn against a neutral cderi
    whose G-mesh collapses every transverse axis to ``G_perp = 0``. Their agreement
    measured only that the two routes broke identically -- the common total drifted
    -6.917 -> -6.568 Ha as the vacuum box went 12 -> 30 bohr, and the multi-k GDF
    drifted -3.458 -> -3.324 Ha on the same sweep.

    Neither route has a gauge for dim<3, so both now refuse. The gauge-consistent
    low-D Hamiltonians (``run_ccm_rhf_wire``, the four-center route) are reached by
    name; the physics gates live in ``tests/test_ccm_lowd_gauge_consistency.py``.
    """
    ccm = _h_chain_ccm((3, 1, 1))
    with pytest.raises(NotImplementedError, match="dim == 1"):
        run_ccm_rhf_direct(ccm)
    with pytest.raises(NotImplementedError, match="no consistent Coulomb gauge"):
        run_ccm_rhf_gdf(ccm)


def test_direct_matches_gdf_lih_ionic_3d():
    """Ionic 3-D gate: direct == GDF on rocksalt LiH, where the ξ_N seam is
    Madelung-large (asserted below, not assumed)."""
    ccm = _lih_ccm((1, 1, 1))
    r, _ = _assert_direct_gdf_parity(ccm, tol=1e-8)
    # Seam magnitude per supercell = ξ_N·N_e/2 — genuinely large here.
    xi = ccm_exchange_q0_madelung(ccm)
    n_e = ccm.supercell.n_electrons()
    assert xi * n_e / 2.0 > 0.1  # Ha — Madelung-scale, not a rounding term


@pytest.mark.slow
def test_direct_matches_gdf_lih_multicell_fold():
    """LiH (2,1,1) — tight gate on the (default) fold-built cderi.

    The per-q unit-cell fits ARE the multi-k GDF's own objects, so parity
    holds by construction even on the ultra-diffuse Li/sto-3g basis:
    measured 4e-14 Ha/cell (2026-07-02)."""
    ccm = _lih_ccm((2, 1, 1))
    r = run_ccm_rhf_direct(ccm)  # cderi_build="fold" default
    g = run_ccm_rhf_gdf(ccm)
    assert r.converged and g.converged
    assert_matched_exchange_q0([r.exchange_q0, BVK_EWALD], context="lih-211")
    assert r.total_cyclic_energy == pytest.approx(r.energy, abs=0.0)
    assert abs(r.total_cyclic_energy / ccm.n_cells - g.energy) < 1e-8


@pytest.mark.slow
def test_supercell_cderi_lih_documented_residual():
    """LiH (2,1,1) with the SUPERCELL-FFT cderi — documented residual (canary).

    Measured 2026-07-02: direct(supercell-L) − GDF = −1.35e-4 Ha/cell,
    independent of the cderi lattice cutoff (26/30/45 bohr identical) and
    thus NOT the (fixed) cross-torus wrap truncation: the supercell-Γ FFT
    fit's J/K break torus translation invariance at ~1e-4 on this
    ultra-diffuse basis (vs machine ε on H/sto-3g), a defect of the
    supercell-fit path specifically — the fold-built cderi on the same torus
    matches the GDF to 4e-14 (test above), which is what localizes it.
    Li/sto-3g's 42–56-bohr 2sp extents are the documented problem class of
    the shared rsgdf machinery (handovers/HANDOVER_RIJCOSX_M3A.md § M3b-5,
    escalated).
    Pinned at 5e-4 so a builder fix is noticed and this gate tightened.
    """
    ccm = _lih_ccm((2, 1, 1))
    r = run_ccm_rhf_direct(ccm, cderi_build="supercell")
    g = run_ccm_rhf_gdf(ccm)
    assert r.converged and g.converged
    d = abs(r.energy / ccm.n_cells - g.energy)
    assert 1e-6 < d < 5e-4  # residual present (lower bound: it's a real defect)


def test_fold_cderi_matches_supercell_cderi():
    """The fold-built cderi is the same RI factorization as the supercell fit:
    identical four-center contraction (≈1e-10) on compact toruses, including
    a multi-axis mesh — the case that requires frame-consistent (unfolded-kb)
    per-q builds."""
    from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi_fold

    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    for nrep in ((3, 1, 1), (2, 2, 1)):
        ccm = CCMSystem(cell, nrep, "sto-3g")
        L_f = ccm_neutral_cderi_fold(ccm)
        L_s = ccm_neutral_cderi(ccm)
        g_f = np.einsum("Pmn,Prs->mnrs", L_f, L_f, optimize=True)
        g_s = np.einsum("Pmn,Prs->mnrs", L_s, L_s, optimize=True)
        assert np.max(np.abs(g_f - g_s)) < 1e-8


def test_seam_off_offset_is_exactly_half_xi_ne():
    """The exxdiv=None control sits above the ewald run by exactly
    ξ_N·N_e/2 per supercell.

    Exactness, not first order: the seam Fock term −(ξ_N/2)·S D S is
    block-diagonal in occ/virt at convergence (S D S has no occ-virt element
    for an idempotent D), so both conventions converge to the *same* density
    and the total-energy offset is the seam energy identity
    −(ξ_N/4)·Tr[(DS)²] = −(ξ_N/2)·N_e. This is the demonstration that the
    derived seam operator is what closes the documented strict-zero-vs-ewald
    gauge gap (neutral.py / the D2 finding).

    Runs on a compact 3-D cell. It used to use the dim=1 H₂ chain, which the
    direct-torus route no longer accepts (2026-07-10: no gauge exists there, see
    tests/test_ccm_lowd_gauge_consistency.py). The seam identity is independent of
    the periodic dimension — it follows from idempotency of D and the linearity of
    the seam in D — so the demonstration is unchanged by the move.
    """
    ccm = _h2_cubic_ccm((2, 1, 1))
    L = ccm_neutral_cderi(ccm)  # build once, share across both runs
    r_on = run_ccm_rhf_direct(ccm, cderi=L, exxdiv="ewald")
    r_off = run_ccm_rhf_direct(ccm, cderi=L, exxdiv=None)
    assert r_on.converged and r_off.converged
    assert r_on.exchange_q0 == BVK_EWALD
    assert r_off.exchange_q0 == STRICT_ZERO

    xi = ccm_exchange_q0_madelung(ccm)
    n_e = ccm.supercell.n_electrons()
    offset = r_off.energy - r_on.energy
    assert offset == pytest.approx(xi * n_e / 2.0, abs=1e-9)
    # Same converged density (block-diagonal seam ⇒ no orbital relaxation).
    assert np.max(np.abs(r_off.density - r_on.density)) < 1e-7
    # And the mismatch gate refuses to compare the two conventions.
    with pytest.raises(ValueError, match="exchange-q=0 convention mismatch"):
        assert_matched_exchange_q0(
            [r_on.exchange_q0, r_off.exchange_q0], context="seam test")


def test_direct_matches_gdf_non_trim_mesh():
    """Non-TRIM (complex-q) folding gate: (3,1,1) on a fully periodic cell.

    Regression for the 2026-07-02 cross-torus truncation bug: with the flat
    ~15-bohr lattice cutoff, the Γ-supercell cderi on this 18-bohr supercell
    lost the image cell that wraps edge AO pairs, breaking torus translation
    invariance (J/K asymmetry 4e-2) and mismatching the GDF by 1.1e-4 Ha/cell
    on exactly the meshes with complex-q channels. TRIM-only meshes
    ((2,1,1) / (1,1,2), supercell ≤ 12 bohr) masked it. The wrap-aware
    cutoff (`_supercell_wrap_lat_opts`) restores 1e-11-level parity."""
    cell3 = PeriodicSystem(
        3, np.diag([6.0, 15.0, 15.0]),
        [Atom(1, [0.0, 7.5, 7.5]), Atom(1, [1.4, 7.5, 7.5])], 0, 1)
    _assert_direct_gdf_parity(CCMSystem(cell3, (3, 1, 1), "sto-3g"), tol=1e-8)


# --- D2: neutral RIJCOSX on the direct route ---------------------------------
#
# Gate calibration (2026-07-02): the composed SR+LR periodic COSX exchange
# (KPointCosxK, M3b) carries an intrinsic ~22-30 µHa/cell composition/DF floor
# vs the exact neutral-cderi exchange — invariant under ω, LR cutoff, and grid
# tier, and equal to the engine's own validated backend parity (0.024-0.027
# mHa, handovers/HANDOVER_RIJCOSX_M3A.md § M3b-4c). The gates below pin the route at
# 5e-5 Ha/cell (floor + margin); tightening to the mission's ~10 µHa target
# requires a floor improvement in the shared engine, not CCM-side tuning.


def _h2_compact_3d(nrep=(2, 2, 2)):
    """Compact genuine-3-D control: H₂ in a 6-bohr cubic cell."""
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def test_direct_rijcosx_matches_direct_anchor():
    """Neutral RIJCOSX == direct route on the dim=3 vacuum-padded H₂ anchor
    (measured −29.6 µHa/cell — the engine floor), at matched exchange-q0."""
    ccm = _h2_anchor_ccm()
    L = ccm_neutral_cderi(ccm)
    r = run_ccm_rhf_direct(ccm, cderi=L)
    c = run_ccm_rhf_direct_rijcosx(ccm, cderi=L)
    assert r.converged and c.converged
    assert_matched_exchange_q0(
        [r.exchange_q0, c.exchange_q0], context="cosx-vs-direct")
    assert abs(c.energy - r.energy) / ccm.n_cells < 5e-5


@pytest.mark.slow
def test_direct_rijcosx_matches_direct_compact_3d():
    """Neutral RIJCOSX == direct route on the compact 3-D H₂ (2,2,2) torus
    (measured −28.7 µHa/cell at the default COSX tier)."""
    ccm = _h2_compact_3d()
    L = ccm_neutral_cderi(ccm)
    r = run_ccm_rhf_direct(ccm, cderi=L)
    c = run_ccm_rhf_direct_rijcosx(ccm, cderi=L)
    assert r.converged and c.converged
    assert abs(c.energy - r.energy) / ccm.n_cells < 5e-5


def test_direct_rijcosx_seam_off_offset():
    """The exchange-q0 seam identity holds on the COSX route too: the
    exxdiv=None control sits above the ewald run by exactly ξ_N·N_e/2 per
    supercell (same block-diagonal argument as the exact-K route)."""
    ccm = _h2_anchor_ccm()
    L = ccm_neutral_cderi(ccm)
    c_on = run_ccm_rhf_direct_rijcosx(ccm, cderi=L, exxdiv="ewald")
    c_off = run_ccm_rhf_direct_rijcosx(ccm, cderi=L, exxdiv=None)
    assert c_on.exchange_q0 == BVK_EWALD
    assert c_off.exchange_q0 == STRICT_ZERO
    xi = ccm_exchange_q0_madelung(ccm)
    n_e = ccm.supercell.n_electrons()
    assert c_off.energy - c_on.energy == pytest.approx(
        xi * n_e / 2.0, abs=1e-7)


def test_direct_rijcosx_low_d_raises():
    """dim<3 refuses loudly: the LR erf complement on the dimension-collapsed
    reciprocal mesh cannot compose with the exact real-space SR exchange
    (measured K error 0.08–0.6 on dim=1 chain toruses, growing with the SR
    share). A canary — when the low-D composition is fixed (D3 mixed-boundary
    kernels), replace this with a parity gate."""
    ccm = _h_chain_ccm((3, 1, 1))
    with pytest.raises(NotImplementedError, match="dim < 3"):
        run_ccm_rhf_direct_rijcosx(ccm)


def test_seam_constant_equals_supercell_madelung_3d():
    """For dim=3, ξ_N (the _madelung_for_kmesh delegate) == the cluster
    supercell's own madelung_constant_for_cell — the seam constant is the
    BvK-supercell probe-charge Ewald constant, not the primitive-cell one."""
    from vibeqc.madelung import madelung_constant_for_cell

    ccm = _lih_ccm((2, 1, 1))
    xi = ccm_exchange_q0_madelung(ccm)
    assert xi == pytest.approx(
        float(madelung_constant_for_cell(ccm.cluster_system)), rel=1e-12)
    # And it is smaller than the primitive-cell ξ (k-mesh-aware, the
    # _madelung_for_kmesh docstring's LiH story).
    assert xi < float(madelung_constant_for_cell(ccm.unit_system))


# --- Open-shell direct-torus route (run_ccm_uhf_direct) -----------------------


def test_uhf_direct_closed_shell_collapses_to_rhf_direct():
    """The load-bearing correctness gate: the per-spin seam ``ξ_N S D_σ S``
    reduces to the closed-shell ``½(K + ξ_N S D S)``, so on an even-electron
    closed-shell cell ``run_ccm_uhf_direct == run_ccm_rhf_direct`` (D_a = D_b =
    D/2). Since ``run_ccm_rhf_direct`` is itself gated to the multi-k GDF / KRHF
    at 1e-8, the open-shell route inherits that parity transitively for closed
    shells."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    r = run_ccm_rhf_direct(ccm)
    u = run_ccm_uhf_direct(ccm)
    assert r.converged and u.converged
    assert u.exchange_q0 == r.exchange_q0 == BVK_EWALD
    assert u.energy == pytest.approx(r.energy, abs=1e-9)
    assert abs(u.s_squared) < 1e-8
    # Spin-symmetric density, equal to half the RHF (factor-2) density.
    assert np.max(np.abs(u.density_alpha - u.density_beta)) < 1e-8
    assert np.max(np.abs(u.density_alpha + u.density_beta - r.density)) < 1e-8


def test_uhf_direct_seam_off_offset_is_exact_open_shell():
    """Open-shell seam identity: the ``exxdiv=None`` control sits above the
    ``"ewald"`` run by exactly ``ξ_N·N_e/2`` per supercell with the SAME per-spin
    densities — the per-spin seam is occ/virt block-diagonal in each spin at
    convergence, so it shifts the energy, not the state (the open-shell twin of
    the closed-shell seam-off control)."""
    ccm = _li_cubic_ccm()
    L = ccm_neutral_cderi(ccm)  # share across both runs
    r_on = run_ccm_uhf_direct(ccm, cderi=L, exxdiv="ewald")
    r_off = run_ccm_uhf_direct(ccm, cderi=L, exxdiv=None)
    assert r_on.converged and r_off.converged
    assert r_on.exchange_q0 == BVK_EWALD
    assert r_off.exchange_q0 == STRICT_ZERO
    assert (r_on.n_alpha, r_on.n_beta) == (2, 1)   # genuine open shell
    assert r_on.s_squared == pytest.approx(0.75, abs=1e-6)

    xi = ccm_exchange_q0_madelung(ccm)
    n_e = ccm.supercell.n_electrons()
    assert (r_off.energy - r_on.energy) == pytest.approx(xi * n_e / 2.0, abs=1e-8)
    assert np.max(np.abs(r_off.density_alpha - r_on.density_alpha)) < 1e-6
    assert np.max(np.abs(r_off.density_beta - r_on.density_beta)) < 1e-6


def test_uhf_direct_fails_closed_below_3d():
    """``dim < 3`` fails closed, exactly like the closed-shell route (the
    3-D-torus construction has no gauge-consistent Coulomb kernel below three
    dimensions)."""
    ccm = _h_chain_ccm((3, 1, 1))
    with pytest.raises(NotImplementedError, match="dim == 1"):
        run_ccm_uhf_direct(ccm)


def test_factored_k_matches_dense_reference():
    """The occupied-rank factored exchange (efficiency-program K lever,
    2026-08-21) equals the dense reference at machine precision on an
    aufbau density, including the seam composition, and the PSD guard
    rejects indefinite input. Measured on LiH pob-TZVP-REV2 (2,2,2):
    2.7e-14 max element at 2.3x (local) / projected ~3x at the (3,3,3)
    ladder rung where K is ~80% of the per-iteration cost."""
    from vibeqc.periodic.ccm.direct import _k_neutral_fast
    from vibeqc.periodic.ccm.ri import (
        ccm_ri_k_neutral,
        ccm_ri_k_neutral_factored,
        psd_density_factor,
    )

    ccm = _h2_cubic_ccm((2, 1, 1))
    L = ccm_neutral_cderi(ccm)
    scf = run_ccm_rhf_direct(ccm, cderi=L)
    D = scf.density
    S = np.asarray(scf.overlap)

    X = psd_density_factor(D)
    assert X.shape[1] == ccm.supercell.n_electrons() // 2  # aufbau rank
    K_dense = ccm_ri_k_neutral(L, D)
    K_fact = ccm_ri_k_neutral_factored(L, X)
    scale = float(np.max(np.abs(K_dense)))
    assert np.max(np.abs(K_fact - K_dense)) < 1e-12 * max(scale, 1.0)

    # Seam composition through the same factor.
    xi = 0.371
    ref = K_dense + xi * (S @ D @ S)
    fast = _k_neutral_fast(L, D, S, xi)
    assert np.max(np.abs(fast - ref)) < 1e-12 * max(float(np.max(np.abs(ref))), 1.0)

    # The asymmetric-L trap is covered: L blocks here are the real fold
    # stack whose per-P blocks are NOT symmetric in general; a factored
    # form that drops the second-factor transpose was measured 5.6e-6
    # off on pob-class blocks and must not come back.
    with pytest.raises(ValueError, match="indefinite"):
        psd_density_factor(D - 2.0 * np.eye(D.shape[0]))


def test_direct_prebuilt_cderi_rejects_aux_basis():
    """IID 344: aux_basis together with a prebuilt cderi is a contradiction
    on the direct route -- the cderi already fixes the fitting basis, so the
    route fails closed instead of silently ignoring the aux basis the caller
    asked for."""
    from vibeqc.periodic.ccm.direct import _neutral_cderi_for

    ccm = _h2_cubic_ccm((1, 1, 1))
    L = _neutral_cderi_for(ccm, None, "fold", 200.0, None)
    with pytest.raises(ValueError, match="aux_basis is ignored"):
        run_ccm_rhf_direct(ccm, cderi=L, aux_basis="def2-svp-jk")


def test_direct_result_carries_backend_identity():
    """IID 344: the real-Γ neutral direct route identifies its executing
    operator on the result and positively reports the parity-hold state
    (this route never touches the multi-k GDF hold class, so held=False is
    an affirmative verdict, not a default of ignorance)."""
    from vibeqc.periodic.ccm.direct import _neutral_cderi_for

    ccm = _h2_cubic_ccm((1, 1, 1))
    L = _neutral_cderi_for(ccm, None, "fold", 200.0, None)
    r = run_ccm_rhf_direct(ccm, cderi=L)
    assert r.converged
    assert r.backend == "ccm-neutral-direct-rhf"
    assert r.parity_held is False


def test_vacuum_direction_classifier_measures_not_declares():
    """IID 291: the physical dimensionality decision must come from the atom
    positions, not the declared dim. A vacuum-padded 1-D chain is measured
    as carrying vacuum on its two padded axes, while the externally validated
    dilute anchor cell and uniform boxes stay measured 3-D."""
    from vibeqc.periodic.ccm.system import ccm_vacuum_directions

    assert ccm_vacuum_directions(_padded_h_chain_ccm()) == [0, 1]
    assert ccm_vacuum_directions(_h2_anchor_ccm()) == []
    assert ccm_vacuum_directions(_h2_cubic_ccm((1, 1, 1))) == []
    # A uniformly dilute 3-D molecular crystal (80-bohr cube) has no
    # needle-cell vacuum direction and must stay un-flagged.
    big = PeriodicSystem(
        3, np.diag([80.0, 80.0, 80.0]),
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])], 0, 1)
    assert ccm_vacuum_directions(CCMSystem(big, (1, 1, 1), "sto-3g")) == []


def test_padded_1d_chain_fails_closed_on_neutral_routes():
    """IID 291: a 1-D chain padded to a declared-3-D cell must be treated
    like the declared-1-D chain -- every neutral-route entry point refuses
    it with a message naming the affected directions."""
    from vibeqc.periodic.ccm.ri import (
        run_ccm_rhf_gdf,
        run_ccm_rhf_ri_neutral,
    )

    ccm = _padded_h_chain_ccm()
    for fn in (run_ccm_rhf_direct, run_ccm_rhf_gdf, run_ccm_rhf_ri_neutral):
        with pytest.raises(NotImplementedError, match="vacuum") as exc:
            fn(ccm)
        assert "1, 2" in str(exc.value)  # names the two padded directions


def test_positive_closed_shell_energy_fails_closed():
    """IID 291 second arm, independent of the gate: a closed-shell neutral
    CCM SCF that converges to a positive total energy must not report
    success. The compressed padded chain converges to +0.03 Ha (garbage)."""
    from vibeqc.periodic.ccm.scf import run_ccm_rhf

    with pytest.raises(ValueError, match="positive total energy"):
        run_ccm_rhf(_padded_h_chain_ccm(), method="aiccm2026dev-a")


# ---------------------------------------------------------------------------
# GitLab #307 / #308 -- the tight-core high-|G| tail.
#
# run_ccm_rhf_gdf at nrep=(1,1,1) does not enter the multi-k SCF loop: a
# single-Γ mesh dispatches to run_pbc_gdf_rhf, which auto-sizes an RSGDF
# high-|G+q| tail at 1.1 x 10 x zeta_max on the tight-core class. The direct
# route builds its cderi through ccm_neutral_cderi_fold, which had no tail
# parameter at all, so on that class the two routes silently evaluated
# DIFFERENT Hamiltonians and no result field said so.
#
# Measured 2026-08-25 before the fix landed:
#   MgO fcc a=7.956 / STO-3G N=1: untailed direct -271.5490049276 Ha vs
#     Γ GDF -271.0496165976 Ha = -4.9939e-01 Ha/cell; with the tail matched,
#     +4.6e-11 Ha.
#   diamond a=6.74 / STO-3G N=1: -1.5884e-02 Ha -> -7.0e-11 Ha.
# The TAILED side is the accurate one: it sits 0.3 mHa from the PySCF value
# quoted in _gamma_dense_core_gdf_parity_held.
#
# ccm_neutral_tail_ke_cutoff now resolves the tail for BOTH routes.
#
# THE CANONICAL GATE FOR THAT IS tests/test_ccm_neutral_tail_parity.py (16
# tests, landed with the fix): end-to-end direct-vs-GDF parity on fcc Be and
# diamond, each asserting that the UNTAILED control still fails the same
# tolerance, so the negative branch is exercised. Do not duplicate it here.
#
# What this file adds is coverage of the shared _assert_direct_gdf_parity
# helper on a tight-core cell -- that helper is used by every other
# direct-vs-GDF gate here and the tail-parity file does not call it -- plus
# the LiH negative control that explains why every pre-existing control in
# THIS file passed while the defect was live.
# ---------------------------------------------------------------------------


def test_tail_classifier_pins_the_tight_core_gate_cell():
    """The gate cell is in the tight-core class and its resolved tail is the
    pinned constant -- and the pre-existing LiH control is NOT in the class.

    The negative control is the point: it is why every gate in this file
    passed while the defect was live.
    """
    from vibeqc import make_basis
    from vibeqc.pbc_gdf import (
        _gamma_dense_core_gdf_parity_held,
        _max_ao_primitive_exponent,
    )
    from vibeqc.periodic.ccm.neutral import ccm_neutral_tail_ke_cutoff

    ccm = _diamond_ccm()
    basis = make_basis(ccm.unit_system.unit_cell_molecule(), ccm.basis_name)
    assert _max_ao_primitive_exponent(basis) == pytest.approx(71.6168373, rel=1e-6)

    # Untailed: held. Tailed at the resolved value: lifted. Both branches
    # reachable, so neither assertion is a decoration.
    assert _gamma_dense_core_gdf_parity_held(
        ccm.unit_system, "rsgdf", ao_basis=basis) is True
    tail = ccm_neutral_tail_ke_cutoff(ccm, ke_cutoff=200.0, tail_ke_cutoff=None)
    assert tail == pytest.approx(DIAMOND_STO3G_AUTO_TAIL, rel=1e-9)
    assert _gamma_dense_core_gdf_parity_held(
        ccm.unit_system, "rsgdf", ao_basis=basis, tail_ke_cutoff=tail) is False

    # NEGATIVE CONTROL: LiH/STO-3G resolves no tail, so the pre-existing
    # nrep=(1,1,1) gate below could never have caught #307.
    lih = _lih_ccm()
    lih_basis = make_basis(
        lih.unit_system.unit_cell_molecule(), lih.basis_name)
    assert _gamma_dense_core_gdf_parity_held(
        lih.unit_system, "rsgdf", ao_basis=lih_basis) is False
    assert ccm_neutral_tail_ke_cutoff(
        lih, ke_cutoff=200.0, tail_ke_cutoff=None) is None


@pytest.mark.slow
def test_direct_matches_gdf_tight_core_diamond_3d():
    """_assert_direct_gdf_parity on a TIGHT-CORE cell at nrep=(1,1,1).

    The end-to-end numerics for the #307 tail are gated in
    ``tests/test_ccm_neutral_tail_parity.py``, which also pins the falsifying
    control (untailed must still fail). This case exists because every OTHER
    direct-vs-GDF gate in this file routes through
    :func:`_assert_direct_gdf_parity`, and until this case existed that helper
    was only ever exercised on loose-core cells -- so its new matched-tail and
    parity_held assertions had no tight-core coverage at all.

    With the tail plumbing reverted the direct side returns -74.0179352366
    against the GDF's -74.0020515730, a 1.6e-2 Ha/cell gap this fails on.
    """
    ccm = _diamond_ccm()
    r, g = _assert_direct_gdf_parity(ccm, tol=1e-8)

    # Not vacuously satisfied by "both None": this cell must actually tail.
    assert r.rsgdf_tail_ke_cutoff == pytest.approx(
        DIAMOND_STO3G_AUTO_TAIL, rel=1e-9)
    assert g.rsgdf_tail_ke_cutoff == pytest.approx(
        DIAMOND_STO3G_AUTO_TAIL, rel=1e-9)
    # The Γ fast path is what runs at a single-Γ mesh; pin that too, since the
    # whole defect was this route switch going unnoticed.
    assert "gamma" in str(g.raw.backend).lower()
    # Absolute anchor, so a future change that moves BOTH routes together
    # still fails here rather than passing on self-consistency alone.
    assert r.energy / ccm.n_cells == pytest.approx(-74.0020515731, abs=1e-8)
