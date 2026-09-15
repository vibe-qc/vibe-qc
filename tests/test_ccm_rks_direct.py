"""KS-DFT on the direct-torus (real Γ-supercell) route — external + identity gates.

Gates for :func:`vibeqc.periodic.ccm.direct.run_ccm_rks_direct` (the M1/M2
method-matrix deliverables of the neutral real-Gamma control line).

**Relation to the ``run_ccm_rks_gdf`` reference** (the way the HF gates in
``test_ccm_direct.py`` compare against ``run_ccm_rhf_gdf``): the multi-k / Γ
GDF **KS** reference was found internally inconsistent on 2026-07-09 (the
KRKS-side finding; HF unaffected) — on rocksalt LiH/STO-3G (1,1,1) PBE the
production KRKS reported ``-7.930753392700`` Ha where external PySCF 2.13.1
KRKS(GDF) gives ``-8.238879724246`` and ``run_ccm_rks_direct``
``-8.233910335588`` (the ~5 mHa gap to PySCF is the documented HF-level
ultra-diffuse-Li shared-fitting deviation; XC-attributable ~0.26 mHa). Three
GDF-side defects were fixed in sequence, and each canary that pinned them is
now folded into the direct-vs-GDF parity gates of section 3:

* the Γ-path **pure** KS density/grid-convention mismatch (``b3f74aa9``:
  home-cell-only density evaluated on the periodic-Becke grid),
* the Γ-path **hybrid** KS exchange convention (same-day follow-up fix): the
  Γ KS fast path delegated to the legacy molecular-limit gamma driver, whose
  full-range real-space K carries NO exxdiv convention — +8.32e-2 Ha above
  the exxdiv-matched direct route on LiH/PBE0 (neither the strict-zero-mode
  offset 0.297 Ha = a_x·ξ_N·N_e/2 nor the ~5 mHa fitting floor). Γ
  closed-shell KS now rides the PySCF-µHa-validated ``run_pbc_gdf_rks``
  (a_x-scaled GDF K + the ``exxdiv='ewald'`` Madelung shift — the identical
  convention to ``apply_exxdiv_ewald_to_K`` and the direct route's seam),
* the **multi-k dim<3** KS Hartree (same follow-up fix): pure functionals
  kept the EWALD_3D J branch, which is 3D-only and silently degrades to the
  diagnostic FFT-Poisson grid backend on dim<3 — reported -3.0897 Ha/cell on
  the (3,1,1) H₂-chain PBE fixture while its own stored density evaluates to
  -3.7814 under the same functional and the variational minimum sits at
  -3.8947. Pure multi-k KS on dim<3 was rerouted to the cached-Lpq GDF J,
  like HF/hybrids always did — and then superseded by the 2026-07-10 dim<3
  fail-closed correction (``ad7bdcd3`` line): the collapsed reciprocal mesh
  is not a Coulomb kernel at all, so both KS routes now refuse dim<3 and the
  section-3 gate for this class asserts the refusal.

The sharp correctness gates are:

1. **Vacuum limit** (equivalence theorem at N_c=1 + vacuum): direct KS ==
   molecular ``run_rks`` on an isolated H₂ in a 20-bohr box, per XC rung.
2. **External anchors**: pinned out-of-process PySCF KRKS values on ionic 3-D
   LiH for PBE and PBE0 at the shared-machinery tolerance.
3. **Direct-vs-GDF KS parity** for the Γ defect classes (pure Γ / hybrid Γ)
   at the floor each pairing supports; the multi-k dim<3 class instead
   asserts that both routes fail closed (see the bullet above).
4. **Exact identities**: the hybrid ``exxdiv=None`` control sits above the
   ewald run by exactly ``a_x·ξ_N·N_e/2`` per supercell, with the same
   converged density.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import (
    _external_xc_grid_options,
    _real_gamma_rks_xc_builder,
    _real_gamma_uks_xc_builder,
    _xc_difference_closed_cells,
    _xc_lattice_options,
    ccm_exchange_q0_madelung,
    run_ccm_rks_direct,
)
from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi
from vibeqc.periodic.ccm.ri import run_ccm_rks_gdf
from vibeqc.periodic.ccm.route import run_ccm_scf
from vibeqc.periodic.exchange_convention import BVK_EWALD, STRICT_ZERO

pytestmark = pytest.mark.experimental  # neutral finite-BvK-torus research lane

_EXTERNAL_NAMES = itertools.count()


class _QuadraticFullGridProvider:
    """Analytic nonlocal test functional E = 1/2 (integral rho)^2."""

    def __init__(self):
        self.calls = []

    def __call__(self, features):
        self.calls.append(features)
        weights = np.asarray(features["grid_weights"], dtype=float)
        rho_a = np.asarray(features["rho_alpha"], dtype=float)
        rho_b = np.asarray(features["rho_beta"], dtype=float)
        integral = float(weights @ (rho_a + rho_b))
        q = integral * weights
        zeros = np.zeros(weights.size)
        zeros_grad = np.zeros((weights.size, 3))
        return {
            "energy": 0.5 * integral * integral,
            "v_rho_alpha": q,
            "v_rho_beta": q,
            "v_grad_alpha": zeros_grad,
            "v_grad_beta": zeros_grad,
            "v_tau_alpha": zeros,
            "v_tau_beta": zeros,
        }


def _external_functional(provider, *, required_grid_profile=""):
    name = f"test-real-gamma-full-grid-{next(_EXTERNAL_NAMES)}"
    vq.define_external_functional(
        name,
        provider,
        required_grid_profile=required_grid_profile,
    )
    return name

# --- External references (PySCF 2.13.1, out of process, 2026-07-09) ---------
# Rocksalt LiH, fcc primitive cell a = 7.72 bohr, STO-3G, (1,1,1) mesh.
# pyscf.pbc dft.KRKS(cell, kpts).density_fit(); pure PBE carries no
# exchange-q=0 convention; PBE0 ran at pyscf's default exxdiv='ewald' == this
# route's exxdiv="ewald" (matched convention).
E_PYSCF_KRKS_PBE_LIH = -8.238879724246488
E_PYSCF_KRKS_PBE0_LIH = -8.291224931282631
# Shared-machinery baseline on the same fixture: PySCF KRHF -8.334998435128
# vs vibe-qc direct/GDF HF -8.330292527860 => 4.71 mHa is the documented
# ultra-diffuse Li/STO-3G rsgdf fitting deviation, NOT an XC error. Gate the
# KS anchors at 6 mHa (baseline + margin); the XC-attributable part measured
# 0.26 mHa (PBE) / 0.51 mHa (PBE0).
TOL_EXTERNAL_LIH = 6e-3


def _h_chain_ccm(nrep=(2, 1, 1)):
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


def _lih_ccm(nrep=(1, 1, 1)):
    """Ionic 3-D control: rocksalt LiH, fcc primitive cell (a = 7.72 bohr)."""
    a = 7.72
    lat = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]).T
    cell = PeriodicSystem(
        3, lat, [Atom(3, [0, 0, 0]), Atom(1, [0.5 * a, 0.5 * a, 0.5 * a])],
        0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _h2_box_pair():
    """Isolated H₂ (20-bohr box) as (CCMSystem, Molecule) for the vacuum limit."""
    from vibeqc import Molecule

    atoms = [Atom(1, [10.0, 10.0, 9.3]), Atom(1, [10.0, 10.0, 10.7])]
    cell = PeriodicSystem(3, np.diag([20.0, 20.0, 20.0]), atoms, 0, 1)
    return CCMSystem(cell, (1, 1, 1), "sto-3g"), Molecule(atoms, 0, 1)


def _molecular_rks(mol, functional):
    from vibeqc import RKSOptions, make_basis, run_rks

    opts = RKSOptions()
    opts.functional = functional
    return run_rks(mol, make_basis(mol, "sto-3g"), opts)


# --- 1. Vacuum limit: direct KS == molecular RKS, per XC rung ----------------


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["svwn", "pbe", "r2scan"])
def test_rks_direct_vacuum_limit_matches_molecular(functional):
    """Equivalence-theorem N_c=1 vacuum limit (CCM_THEORY.md § 10 remark 1):
    the real-Γ KS on an isolated H₂ in a 20-bohr box reproduces molecular
    ``run_rks`` for each XC rung (LDA / GGA / meta-GGA). Residual = periodic
    images + periodic-Becke-vs-molecular grid partition (measured 4.5e-5 Ha
    for PBE); gate at 2e-4."""
    ccm, mol = _h2_box_pair()
    r = run_ccm_rks_direct(ccm, functional)
    m = _molecular_rks(mol, functional)
    assert r.converged and m.converged
    assert r.energy / ccm.n_cells == pytest.approx(m.energy, abs=2e-4)


# --- 2. External anchors (ionic 3-D, out-of-process PySCF) -------------------


def test_rks_direct_pbe_lih_matches_external_pyscf():
    """Pure-GGA ionic 3-D anchor: direct KS == external PySCF KRKS to the
    shared-machinery floor (see module docstring; XC-attributable ~0.26 mHa).
    Pure functional => convention-free comparison."""
    ccm = _lih_ccm()
    r = run_ccm_rks_direct(ccm, "pbe")
    assert r.converged
    assert r.e_hf_exchange == 0.0  # pure: no exact-exchange channel
    assert r.energy / ccm.n_cells == pytest.approx(
        E_PYSCF_KRKS_PBE_LIH, abs=TOL_EXTERNAL_LIH)


def test_rks_direct_pbe0_hybrid_lih_matches_external_pyscf():
    """Global-hybrid anchor: PBE0 direct == external PySCF KRKS
    (exxdiv='ewald' both sides). The a_x-scaled seam is Madelung-large here
    (asserted below) — a seam/convention error would miss the anchor by
    ~a_x·ξ_N·N_e/2 ≈ 0.1 Ha, not the 6 mHa machinery floor."""
    ccm = _lih_ccm()
    r = run_ccm_rks_direct(ccm, "pbe0")
    assert r.converged
    assert r.e_hf_exchange < 0.0  # hybrid: exact-exchange channel active
    assert r.exchange_q0 == BVK_EWALD
    assert r.energy / ccm.n_cells == pytest.approx(
        E_PYSCF_KRKS_PBE0_LIH, abs=TOL_EXTERNAL_LIH)
    xi = ccm_exchange_q0_madelung(ccm)
    n_e = ccm.supercell.n_electrons()
    assert 0.25 * xi * n_e / 2.0 > 0.025  # Ha — the a_x-scaled seam is large


# --- 3. Direct-vs-GDF KS parity gates (the 2026-07-09 finding, all fixed) ----


def test_rks_direct_matches_gdf_pure_gamma_post_fix():
    """The first flipped canary (the 2026-07-09 finding, fixed same day for
    the Γ-path pure KS by the b3f74aa9 XC-density/grid pairing): direct ==
    GDF KS on ionic LiH/PBE. On the legacy Γ KS path the post-b3f74aa9 delta
    was -1.04e-4 Ha (the quadrature floor between two independent XC
    grid/convention pairings); since the Γ-hybrid fix rerouted ALL Γ
    closed-shell KS to run_pbc_gdf_rks, both sides evaluate XC through the
    identical build_xc_periodic + periodic-Becke + Γ-torus-density machinery
    and the measured delta is +5.6e-13 Ha. Gate at 1e-8, the HF-gate class
    (LiH (1,1,1) D1 row of test_ccm_direct.py)."""
    ccm = _lih_ccm()
    r = run_ccm_rks_direct(ccm, "pbe")
    g = run_ccm_rks_gdf(ccm, "pbe")
    assert r.converged and g.converged
    assert r.energy / ccm.n_cells == pytest.approx(g.energy, abs=1e-8)


def test_rks_direct_matches_gdf_hybrid_gamma_post_fix():
    """The second flipped canary (Γ-path HYBRID KS): pre-fix the Γ KS fast
    path delegated to the legacy gamma driver whose real-space K carries no
    exxdiv convention (+8.32e-2 Ha above the direct route on LiH/PBE0 —
    neither the strict-zero-mode offset 0.297 Ha = a_x·ξ_N·N_e/2 nor the
    ~5 mHa fitting floor). Post-fix, Γ closed-shell KS rides
    run_pbc_gdf_rks: the a_x-scaled GDF exchange with the exxdiv='ewald'
    Madelung shift — the identical convention to the direct route's
    ξ_N·S·D·S seam — and the measured delta is +7.0e-13 Ha. Gate at 1e-8
    (HF-gate class). Both sides also anchor to external PySCF KRKS PBE0
    (exxdiv='ewald') at the shared-machinery tolerance (section 2)."""
    ccm = _lih_ccm()
    r = run_ccm_rks_direct(ccm, "pbe0")
    g = run_ccm_rks_gdf(ccm, "pbe0")
    assert r.converged and g.converged
    assert r.exchange_q0 == BVK_EWALD
    assert g.raw.e_hf_exchange < 0.0  # hybrid channel active on the GDF side
    assert r.energy / ccm.n_cells == pytest.approx(g.energy, abs=1e-8)


def test_rks_direct_and_gdf_multik_lowd_both_fail_closed():
    """The third flipped canary, rewritten 2026-07-10 for the dim<3
    fail-closed correction (``ad7bdcd3`` line). Its first rewrite pinned
    direct == GDF at +2.2e-15 Ha on this dim=1 chain after the multi-k KS
    Hartree fix — but that parity measured two routes sharing the same
    collapsed-mesh Coulomb kernel, whose transverse axes are pinned at
    ``G_perp = 0`` (the sibling KRKS pin was converted the same way in
    ``9e6bbaf2``, and the HF twin is
    ``test_ccm_direct.py::test_direct_and_gdf_h_chain_1d_both_fail_closed``;
    the H-chain sits 0.512 Ha/atom over-bound against the four-center value
    on that kernel). Mutual agreement between two routes sharing a defect is
    not validation, so both KS routes now refuse dim<3 loudly. The
    gauge-consistent low-D physics gates live with the wire/four-center
    routes (``tests/test_ccm_lowd_gauge_consistency.py``)."""
    ccm = _h_chain_ccm((3, 1, 1))
    with pytest.raises(NotImplementedError, match="dim == 1"):
        run_ccm_rks_direct(ccm, "pbe")
    with pytest.raises(NotImplementedError, match="no consistent Coulomb gauge"):
        run_ccm_rks_gdf(ccm, "pbe")


# --- 4. Exact identities + plumbing ------------------------------------------


def test_real_gamma_external_grid_capability_and_image_radius_contract():
    """Provider requirements resolve early and one image radius reaches XC."""
    provider = _QuadraticFullGridProvider()
    name = _external_functional(
        provider, required_grid_profile="pyscf-level3")
    func = vq.Functional(name, 1)

    resolved = _external_xc_grid_options(
        func, None, who="test_real_gamma")
    assert resolved.atomic_grid_profile == vq.AtomicGridProfile.PySCFLevel3

    incompatible = vq.GridOptions()
    with pytest.raises(ValueError, match="requires grid profile"):
        _external_xc_grid_options(
            func, incompatible, who="test_real_gamma")
    assert provider.calls == []

    lattice = vq.LatticeSumOptions()
    lattice.cutoff_bohr = 6.25
    lattice.becke_image_radius_bohr = 1.0
    copied = _xc_lattice_options(lattice, 5.5)
    assert copied.cutoff_bohr == pytest.approx(6.25)
    assert copied.becke_image_radius_bohr == pytest.approx(5.5)
    # The adapter must not mutate a caller-owned lattice-options object.
    assert lattice.becke_image_radius_bohr == pytest.approx(1.0)
    with pytest.raises(ValueError, match="must be at least"):
        _xc_lattice_options(lattice, 7.5)


def test_real_gamma_xc_cell_support_is_difference_closed():
    """Every active AO-image pair has an output density block g = s-a."""
    from vibeqc.pair_resolved_truncation import pair_resolved_domain

    ccm = _h2_cubic_ccm((2, 1, 1))
    cutoff = 6.1
    active = list(pair_resolved_domain(ccm.unit_system, cutoff).cells)
    support = _xc_difference_closed_cells(
        ccm.unit_system,
        cutoff,
        cutoff,
    )
    support_indices = {
        tuple(int(value) for value in np.asarray(cell.index))
        for cell in support
    }
    for bra in active:
        bra_index = np.asarray(bra.index, dtype=int)
        for ket in active:
            difference = tuple(
                int(value)
                for value in np.asarray(ket.index, dtype=int) - bra_index
            )
            assert difference in support_indices


def test_real_gamma_external_xc_projection_is_variational():
    """The exact SCF closure obeys d(Nc Exc)/dDsc = folded Vxc."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    provider = _QuadraticFullGridProvider()
    func = vq.Functional(_external_functional(provider), 1)
    grid = vq.GridOptions()
    grid.n_radial = 4
    grid.n_theta = 3
    grid.n_phi = 4
    lattice = vq.LatticeSumOptions()
    lattice.cutoff_bohr = 6.1
    lattice.nuclear_cutoff_bohr = 6.1
    xc_of_density = _real_gamma_rks_xc_builder(
        ccm,
        func,
        grid_options=grid,
        becke_image_radius_bohr=6.1,
        lat_opts=lattice,
    )

    nbf = int(ccm.basis.nbasis)
    density = 0.32 * np.eye(nbf)
    direction = np.arange(1, nbf * nbf + 1, dtype=float).reshape(nbf, nbf)
    direction = 0.5 * (direction + direction.T)
    direction /= np.linalg.norm(direction)

    _, potential = xc_of_density(density)
    assert len(provider.calls) == 1
    step = 2.0e-7
    e_plus, _ = xc_of_density(density + step * direction)
    e_minus, _ = xc_of_density(density - step * direction)
    finite_difference = (e_plus - e_minus) / (2.0 * step)
    projected = float(np.sum(potential * direction))
    assert projected == pytest.approx(
        finite_difference, rel=3.0e-6, abs=3.0e-7)
    # AO batching is internal: each complete energy build calls the provider
    # exactly once, with periodic metadata and one atom-major unit-cell grid.
    assert len(provider.calls) == 3
    payload = provider.calls[0]
    assert payload["periodic"] is True
    assert payload["periodic_dimension"] == 3
    assert len(payload["atomic_grid_sizes"]) == len(ccm.unit_system.unit_cell)


def test_real_gamma_external_xc_equal_spin_uks_matches_rks():
    """Equal-spin UKS and restricted builders use one density convention."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    provider = _QuadraticFullGridProvider()
    name = _external_functional(provider)
    grid = vq.GridOptions()
    grid.n_radial = 4
    grid.n_theta = 3
    grid.n_phi = 4
    lattice = vq.LatticeSumOptions()
    lattice.cutoff_bohr = 6.1
    rks = _real_gamma_rks_xc_builder(
        ccm,
        vq.Functional(name, 1),
        grid_options=grid,
        becke_image_radius_bohr=6.1,
        lat_opts=lattice,
    )
    uks = _real_gamma_uks_xc_builder(
        ccm,
        vq.Functional(name, 2),
        grid_options=grid,
        becke_image_radius_bohr=6.1,
        lat_opts=lattice,
    )
    density = 0.32 * np.eye(int(ccm.basis.nbasis))
    e_rks, v_rks = rks(density)
    e_uks, va, vb = uks(0.5 * density, 0.5 * density)
    assert e_uks == pytest.approx(e_rks, abs=2.0e-11)
    np.testing.assert_allclose(va, v_rks, atol=2.0e-10, rtol=2.0e-10)
    np.testing.assert_allclose(vb, v_rks, atol=2.0e-10, rtol=2.0e-10)


def test_rks_direct_hybrid_seam_off_offset_scales_with_ax():
    """The exxdiv=None hybrid control sits above the ewald run by exactly
    a_x·ξ_N·N_e/2 per supercell — the HF seam identity scaled by the
    exact-exchange fraction (PBE0: a_x = 1/4), with the same converged
    density (the seam Fock term is occ/virt block-diagonal at convergence).

    Runs on the compact 3-D cell (fixture moved off the dim=1 chain
    2026-07-10, dim<3 fail-closed): the identity follows from idempotency of
    D and the linearity of the seam in D, not from the periodic dimension —
    the HF twin in test_ccm_direct.py moved the same way."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    L = ccm_neutral_cderi(ccm)  # build once, share across both runs
    r_on = run_ccm_rks_direct(ccm, "pbe0", cderi=L, exxdiv="ewald")
    r_off = run_ccm_rks_direct(ccm, "pbe0", cderi=L, exxdiv=None)
    assert r_on.converged and r_off.converged
    assert r_on.exchange_q0 == BVK_EWALD
    assert r_off.exchange_q0 == STRICT_ZERO

    xi = ccm_exchange_q0_madelung(ccm)
    n_e = ccm.supercell.n_electrons()
    a_x = 0.25  # PBE0 exact-exchange fraction (Adamo & Barone 1999)
    offset = r_off.energy - r_on.energy
    assert offset == pytest.approx(a_x * xi * n_e / 2.0, abs=1e-8)
    assert np.max(np.abs(r_off.density - r_on.density)) < 1e-6


def test_rks_direct_route_keyword_dispatch():
    """run_ccm_scf(route='real-gamma', functional=...) reaches the KS direct
    driver (the route selector's former NotImplementedError gap). Compact
    3-D fixture (moved off the dim=1 chain 2026-07-10, dim<3 fail-closed)."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    via_route = run_ccm_scf(ccm, route="real-gamma", functional="pbe")
    direct = run_ccm_rks_direct(ccm, "pbe")
    assert via_route.converged and direct.converged
    assert via_route.energy == pytest.approx(direct.energy, abs=1e-10)
    assert via_route.exchange_q0 == BVK_EWALD


def test_rks_direct_range_separated_full_range_arm_raises():
    """Full-range-arm RSH refuses loudly (shared resolver policy). Compact
    3-D fixture so the RSH guard, not the dim<3 guard, is what this asserts.
    Superseded boundary (2026-07-17): HSE-class SR-only screened hybrids now
    RUN on this route (experimental erfc-attenuated cderi + sigma_0 zero
    mode -- gates in ``tests/test_ccm_rsh_direct.py``); the fail-closed
    surface is the nonzero full-range arm (wb97x, cam-b3lyp, ...)."""
    ccm = _h2_cubic_ccm((2, 1, 1))
    with pytest.raises(NotImplementedError, match="full-range"):
        run_ccm_rks_direct(ccm, "wb97x")
