"""Ewald exchange split (option (b) energy assembly, 2026-06-10).

Covers the corrected BIPOLE Γ exchange convention::

    K = K_SR(erfc ω, direct) + K_LR(erf ω, reciprocal, K≠0)
        + (ξ_M − π/(Vω²))·S·D·S        [exchange_exxdiv='ewald']

and the accompanying gauge changes (full-Bloch density — no Γ-locality
projection — and no EXT EL-SPHEROPOLE term in the total).

External references (PySCF 2.13 GDF, run OUT-OF-PROCESS per
CLAUDE.md §10; export scripts in examples/regression/bipole_parity/):

* MgO fcc primitive (a = 4.21 Å), STO-3G, Γ, RHF exxdiv='ewald':
  E = −271.0496549 Ha. At the converged PySCF density the split-K
  reproduces PySCF's vk element-wise to |ΔK|_max 8e-4 / ΔE_K −0.32 mHa
  (cutoff 14) and is ω-invariant (2026-06-10 session probe).
* H₂/STO-3G in a 12-bohr cubic box, Γ, RHF exxdiv='ewald':
  E = −1.1225839666 Ha. BIPOLE at cutoff 12 / precision 1e-8
  reproduces it to ~1 µHa (measured −1.12258512).
* PySCF ``tools.pbc.madelung`` on the MgO cell: ξ = 0.576295611599.

**Validation status — method-family policy (docs/periodic_jk_routes.md).**
These PySCF ``exxdiv='ewald'`` pins are a **secondary cross-family check** of
BIPOLE's Γ exchange-split *convention* (the ξ_M·S·D·S finite-size exchange
correction), not BIPOLE's primary parity gate. BIPOLE belongs to the CRYSTAL
method family (bipolar/multipole expansion + TOLINTEG truncation + SHRINK
k-sampling); its **primary** gate is the same-family BIPOLE↔CRYSTAL
total-energy parity at a matched k-net, in
``examples/regression/crystal_parity/`` (MgO anchor, SHRINK 8 8). The PySCF Γ
comparison stays here because it is still informative for the
exchange-divergence convention — but its tolerance is the method spread, not
a bug bound, so it is not the gating same-family assertion.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import (
    InitialGuess,
    PeriodicRHFOptions,
    monkhorst_pack,
)
from vibeqc.bipole_fock_ewald import (
    compute_K_long_range_gamma,
    probe_charge_madelung,
    probe_charge_madelung_supercell,
)
from vibeqc.pbc_bipole import run_pbc_bipole_rhf
from vibeqc.pbc_bipole_common import home_cell_block

ANG2BOHR = 1.0 / 0.529177210903

# Published / cross-code reference targets (provenance in module docstring).
# PySCF 2.13 pbc RHF(cell).density_fit(), exxdiv='ewald', conv_tol 1e-10.
E_PYSCF_MGO_GAMMA = -271.04965486
E_PYSCF_H2_12BOHR_GAMMA = -1.1225839666
XI_PYSCF_MGO = 0.576295611599
# Simple-cubic probe-charge Madelung constant ξ·L (Wigner lattice value).
XI_TIMES_L_CUBIC = 2.837297479481
# With a converged fold at cutoff 14, the ω → 1.3ω fixed-density total
# drifts by 0.4 mHa from the residual SR-lattice truncation asymmetry.
OMEGA_INVARIANCE_TOL = 1e-3


def _mgo_primitive():
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    return vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a / 2, a / 2, a / 2])],
    )


def _h2_box(L: float = 12.0, sep: float = 1.4):
    half = 0.5 * sep
    return vq.PeriodicSystem(
        3,
        np.diag([L, L, L]),
        [vq.Atom(1, [0, 0, -half]), vq.Atom(1, [0, 0, half])],
    )


def _rhf_opts(cutoff: float, max_iter: int = 60):
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = cutoff
    opts.lattice_opts.nuclear_cutoff_bohr = cutoff
    opts.max_iter = max_iter
    opts.use_diis = True
    opts.initial_guess = InitialGuess.SAD
    return opts


# ---------------------------------------------------------------------------
# probe_charge_madelung
# ---------------------------------------------------------------------------


def test_probe_charge_madelung_pyscf_pin_and_alpha_invariance():
    sysp = _mgo_primitive()
    xi = probe_charge_madelung(sysp)
    assert xi == pytest.approx(XI_PYSCF_MGO, abs=1e-9)
    # Any α > 0 must give the same constant.
    for alpha in (0.3, 0.9, 1.7):
        assert probe_charge_madelung(sysp, alpha=alpha) == pytest.approx(
            xi, abs=1e-9
        )


def test_probe_charge_madelung_cubic_literature_value():
    for L in (8.0, 12.0):
        sysp = vq.PeriodicSystem(3, np.diag([L, L, L]), [vq.Atom(2, [0, 0, 0])])
        assert probe_charge_madelung(sysp) * L == pytest.approx(
            XI_TIMES_L_CUBIC, abs=1e-8
        )


def test_probe_charge_madelung_rejects_low_dim():
    sysp = vq.PeriodicSystem(
        2,
        np.array([[8.0, 0, 0], [0, 8.0, 0], [0, 0, 30.0]]),
        [vq.Atom(2, [0, 0, 0])],
    )
    with pytest.raises(ValueError, match="dim=3"):
        probe_charge_madelung(sysp)


# ---------------------------------------------------------------------------
# compute_K_long_range_gamma
# ---------------------------------------------------------------------------


def test_k_long_range_gamma_symmetric_and_psd_quadratic_form():
    """K_LR built from a symmetric density is symmetric, and its exchange
    quadratic form Tr[D·K_LR(D)] = Σ_K kernel(K)·|ρ̃-sandwich|² ≥ 0 for any
    PSD density (each K term is kernel(K)·Tr[(ρ̃†Dρ̃)·D] with PSD factors)."""
    from vibeqc._vibeqc_core import LatticeSumOptions, compute_overlap_lattice
    from vibeqc.bipole_fock_ewald import _build_j_long_range_cache

    sysp = _mgo_primitive()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 6.0
    lat.nuclear_cutoff_bohr = 6.0
    S_lat = compute_overlap_lattice(basis, sysp, lat)
    cells_r_cart = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in S_lat.cells], dtype=float
    )
    cache = _build_j_long_range_cache(basis, sysp, cells_r_cart, 0.5587, 1e-6)
    rng = np.random.default_rng(7)
    A = rng.standard_normal((basis.nbasis, basis.nbasis))
    D = A @ A.T  # symmetric PSD
    K_LR = compute_K_long_range_gamma(cache, D)
    assert np.abs(K_LR - K_LR.T).max() < 1e-12
    assert float(np.einsum("ij,ji->", D, K_LR)) > 0.0


# ---------------------------------------------------------------------------
# home_cell_block
# ---------------------------------------------------------------------------


def test_home_cell_block_reads_origin_cell():
    from vibeqc._vibeqc_core import LatticeSumOptions, compute_overlap_lattice

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 4.0
    lat.nuclear_cutoff_bohr = 4.0
    tmpl = compute_overlap_lattice(basis, sysp, lat)
    marker = None
    for i, c in enumerate(tmpl.cells):
        val = float(i + 1)
        tmpl.set_block(i, np.full((basis.nbasis, basis.nbasis), val))
        if not np.asarray(c.index, dtype=int).any():
            marker = val
    assert marker is not None
    blk = home_cell_block(tmpl)
    assert np.all(blk == marker)


# ---------------------------------------------------------------------------
# Driver surface: flags, validation, provenance
# ---------------------------------------------------------------------------


def test_exchange_split_auto_on_at_gamma_and_flagged_on_result():
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    r = run_pbc_bipole_rhf(
        sysp,
        basis,
        monkhorst_pack(sysp, [1, 1, 1]),
        _rhf_opts(4.0, max_iter=30),
        ewald_precision=1e-6,
        progress=False,
    )
    assert r.converged
    assert r.exchange_ewald_split is True
    assert r.exchange_exxdiv == "ewald"
    assert r.e_ext_el_spheropole is None


def test_exchange_split_legacy_optout_keeps_old_gauge():
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    r = run_pbc_bipole_rhf(
        sysp,
        basis,
        monkhorst_pack(sysp, [1, 1, 1]),
        _rhf_opts(4.0, max_iter=30),
        ewald_precision=1e-6,
        use_exchange_ewald_split=False,
        progress=False,
    )
    assert r.converged
    assert r.exchange_ewald_split is False
    assert r.exchange_exxdiv is None
    # Legacy gauge keeps the spheropole term in the total.
    assert r.e_ext_el_spheropole is not None


def test_exchange_split_multik_rejects_adhoc_kmesh():
    """Multi-k split needs MP mesh metadata (BvK torus + supercell ξ_M);
    ad-hoc k-lists carry mesh = (1,1,1) placeholders and are rejected."""
    from vibeqc._vibeqc_core import bloch_kmesh_from_lists

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    mp = monkhorst_pack(sysp, [2, 1, 1])
    adhoc = bloch_kmesh_from_lists(
        [np.asarray(k, dtype=float) for k in mp.kpoints],
        [float(w) for w in mp.weights],
    )
    with pytest.raises(ValueError, match="Monkhorst-Pack"):
        run_pbc_bipole_rhf(
            sysp,
            basis,
            adhoc,
            _rhf_opts(4.0, max_iter=5),
            use_exchange_ewald_split=True,
            ewald_precision=1e-6,
            progress=False,
        )


def test_exchange_split_multik_rejects_torus_uncovering_cutoff():
    """At multi-k the (2×-cutoff) density list must span the BvK torus;
    a 12-bohr box at cutoff 4 (wide list radius 8) cannot host the
    (−1,0,0) torus cell of a [2,1,1] mesh and must fail loudly rather
    than fold a wrong density."""
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    with pytest.raises(ValueError, match="BvK torus"):
        run_pbc_bipole_rhf(
            sysp,
            basis,
            monkhorst_pack(sysp, [2, 1, 1]),
            _rhf_opts(4.0, max_iter=5),
            use_exchange_ewald_split=True,
            ewald_precision=1e-6,
            progress=False,
        )


def test_exchange_exxdiv_validated():
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    with pytest.raises(ValueError, match="exchange_exxdiv"):
        run_pbc_bipole_rhf(
            sysp,
            basis,
            monkhorst_pack(sysp, [1, 1, 1]),
            _rhf_opts(4.0, max_iter=5),
            exchange_exxdiv="vcut_sph",
            progress=False,
        )


def test_exxdiv_convention_identity_gamma():
    """E(none) − E(ewald) == ½·N_e·ξ_M, exactly, at the shared minimizer.

    On the idempotent closed-shell manifold the exxdiv term contributes
    ΔE = −¼·ξ·Σ_k w_k·Tr[D(k)S(k)D(k)S(k)] = −½·N_e·ξ for ANY
    closed-shell density (Tr[DSDS] = 2·N_e by idempotency), i.e. the
    term is constant on the manifold, both conventions share the same
    SCF minimizer, and the two totals differ by exactly the constant.
    This is the conversion identity quoted in
    docs/user_guide/from_crystal.md § "Comparing total energies" —
    pinned here so the doc's arithmetic can never drift from the code.
    """
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    r_ewald = run_pbc_bipole_rhf(
        sysp,
        basis,
        kmesh,
        _rhf_opts(4.0, max_iter=60),
        exchange_exxdiv="ewald",
        ewald_precision=1e-8,
        progress=False,
    )
    r_none = run_pbc_bipole_rhf(
        sysp,
        basis,
        kmesh,
        _rhf_opts(4.0, max_iter=60),
        exchange_exxdiv="none",
        ewald_precision=1e-8,
        progress=False,
    )
    assert r_ewald.converged and r_none.converged
    xi = probe_charge_madelung(sysp)
    n_e = 2.0
    assert r_none.energy - r_ewald.energy == pytest.approx(
        0.5 * n_e * xi, abs=1e-6
    )


def test_exxdiv_convention_identity_multik():
    """Multi-k variant of the convention identity: the constant is
    ½·N_e·ξ_M(BvK supercell) — one supercell Madelung constant for the
    whole mesh, matching PySCF's ``_ewald_exxdiv_for_G0``."""
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    mesh = [2, 1, 1]
    kmesh = monkhorst_pack(sysp, mesh)
    r_ewald = run_pbc_bipole_rhf(
        sysp,
        basis,
        kmesh,
        _rhf_opts(6.0, max_iter=60),
        exchange_exxdiv="ewald",
        ewald_precision=1e-8,
        progress=False,
    )
    r_none = run_pbc_bipole_rhf(
        sysp,
        basis,
        kmesh,
        _rhf_opts(6.0, max_iter=60),
        exchange_exxdiv="none",
        ewald_precision=1e-8,
        progress=False,
    )
    assert r_ewald.converged and r_none.converged
    xi_sc = probe_charge_madelung_supercell(sysp, mesh)
    n_e = 2.0
    assert r_none.energy - r_ewald.energy == pytest.approx(
        0.5 * n_e * xi_sc, abs=1e-6
    )


def test_analytic_gradient_accepts_gamma_split_gauge_result():
    from vibeqc.bipole_gradient import compute_bipole_gradient_rhf

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _rhf_opts(4.0, max_iter=30)
    r = run_pbc_bipole_rhf(
        sysp,
        basis,
        monkhorst_pack(sysp, [1, 1, 1]),
        opts,
        ewald_precision=1e-6,
        # The analytic kernel accepts the corrected exchange gauge, but it
        # does not yet differentiate M5's padded/pair-resolved Fock domain.
        # Keep this gauge-specific contract on the historical diagnostic
        # domain; M5 fail-closed coverage lives in test_bipole_gradient.py.
        sr_image_precision=None,
        use_fock_symmetry_reduce=False,
        progress=False,
    )
    assert r.exchange_ewald_split is True
    with pytest.warns(UserWarning, match="maintained preview"):
        grad = compute_bipole_gradient_rhf(
            sysp, basis, r, lattice_opts=opts.lattice_opts
        )
    assert grad.shape == (2, 3)
    assert np.all(np.isfinite(grad))


# ---------------------------------------------------------------------------
# Energy parity (PySCF cross-code targets)
# ---------------------------------------------------------------------------


def test_h2_12bohr_gamma_matches_pyscf_gdf():
    """M5 padded-domain cutoff-4 result stays within 0.5 mHa of PySCF.

    The historical truncated traversal sat 0.17 mHa from PySCF.  M5 recovers
    the interaction-resolved SR image mass and moves this secondary
    cross-family comparison to +0.445 mHa; the tighter snapshot below pins
    the production default itself.
    """
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _rhf_opts(4.0, max_iter=30)
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    r = run_pbc_bipole_rhf(
        sysp,
        basis,
        monkhorst_pack(sysp, [1, 1, 1]),
        opts,
        ewald_precision=1e-6,
        progress=False,
    )
    assert r.converged
    assert r.energy == pytest.approx(E_PYSCF_H2_12BOHR_GAMMA, abs=5e-4)
    # Snapshot pin (tighter than the PySCF distance) to catch drift.
    #
    # Moved by the #478 alpha-consistent Ewald real cutoff, and this is the
    # cross-code evidence that the fix points the right way: the distance to
    # PySCF's own converged Ewald nuclear repulsion falls from +0.4454 mHa
    # (snapshot -1.122138533948) to +0.1973 mHa, a factor of 2.26. The shift
    # is -2.4818e-4 Ha rather than the -2.4846e-4 seen on the sibling H2
    # fixtures because this one runs at ewald_precision=1e-6 (real cutoff
    # 15.93 bohr) while the 1e-8 default gives 18.39 bohr.
    #
    # Moved again by #674: the corrected split's default alpha is bounded by
    # the 4-bohr exchange cutoff (0.929 at ewald_precision=1e-6, over
    # CRYSTAL's 0.233), so the image exchange the truncated erfc arm dropped
    # is back. -1.122386713788 -> -1.122526209887 (-1.3950e-4 Ha); the
    # distance to PySCF falls from +0.1973 mHa to +0.0578 mHa.
    assert r.energy == pytest.approx(-1.122526209887, abs=1e-6)


def test_s_fold_truncation_drift_mgo_pins():
    """The fold-convergence diagnostic that guards the corrected gauge:
    MgO/STO-3G's diffuse Mg 3sp tails (outermost exponent ≈ 0.046) keep
    the S(Γ) fold badly under-converged at kernel-ish cutoffs. The
    2026-06-10 diagnosis values; at 8 bohr the SCF can land in spurious
    metric-artifact states (converged 0.70 Ha below PySCF with the
    electron count off by 0.43 in the reference metric)."""
    from vibeqc._vibeqc_core import LatticeSumOptions
    from vibeqc.pbc_bipole_common import s_fold_truncation_drift

    sysp = _mgo_primitive()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    def drift(cut):
        lat = LatticeSumOptions()
        lat.cutoff_bohr = cut
        lat.nuclear_cutoff_bohr = cut
        return s_fold_truncation_drift(basis, sysp, lat, extend_factor=2.0)

    d8, d12, d16 = drift(8.0), drift(12.0), drift(16.0)
    assert d8 > 0.1, f"c8 drift should be catastrophic; got {d8:.2e}"
    assert 1e-4 < d12 < 5e-2, f"c12 drift mid-scale; got {d12:.2e}"
    assert d16 < 1e-4, f"c16 drift converged; got {d16:.2e}"


def test_h2_box_fold_drift_negligible():
    """The molecular-limit cell is immune (overlaps die inside the box) —
    the same property that makes its Γ energies hit PySCF to ~µHa.
    Measured: 1.9e-5 at cutoff 4 (the 1s tails at the 12-bohr image are
    not literally zero), 3e-10 at cutoff 12."""
    from vibeqc._vibeqc_core import LatticeSumOptions
    from vibeqc.pbc_bipole_common import s_fold_truncation_drift

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 4.0
    lat.nuclear_cutoff_bohr = 4.0
    assert s_fold_truncation_drift(basis, sysp, lat, extend_factor=4.0) < 1e-4
    lat.cutoff_bohr = 12.0
    lat.nuclear_cutoff_bohr = 12.0
    assert s_fold_truncation_drift(basis, sysp, lat, extend_factor=1.5) < 1e-8


# ---------------------------------------------------------------------------
# RKS Γ propagation (option (b) Phase 4a, 2026-06-11)
# ---------------------------------------------------------------------------
# PySCF 2.13 pbc RKS(cell).density_fit() references, H₂/STO-3G 12-bohr
# box, Γ, conv_tol 1e-10 (out-of-process per CLAUDE.md §10):
E_PYSCF_H2_SVWN = -1.1212772541
E_PYSCF_H2_PBE0 = -1.1558353538
# PySCF 2.13.1 pbc RKS(cell).density_fit(), exxdiv='ewald', water/6-31G
# in an 18-bohr molecular-limit cube, Gamma, conv_tol 1e-10. Generated
# out-of-process for the v0.15 BIPOLE RKS larger-basis residual audit.
E_PYSCF_H2O_PBE0_631G_GAMMA = -76.302434126215


def _rks_opts(functional: str, cutoff: float = 6.0):
    from vibeqc._vibeqc_core import PeriodicKSOptions

    opts = PeriodicKSOptions()
    opts.functional = functional
    opts.lattice_opts.cutoff_bohr = cutoff
    opts.lattice_opts.nuclear_cutoff_bohr = cutoff
    opts.max_iter = 40
    opts.use_diis = True
    opts.initial_guess = InitialGuess.SAD
    return opts


def _h2o_box():
    return vq.PeriodicSystem(
        3,
        18.0 * np.eye(3),
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 1.4305, 1.1070]),
            vq.Atom(1, [0.0, -1.4305, 1.1070]),
        ],
        0,
        1,
    )


def test_rks_h2_svwn_gamma_matches_pyscf():
    """Pure functional under the corrected gauge: the projection removal
    fixes the XC grid density (cross-cell AO products) and the 1e/J
    contraction — H₂/SVWN lands on PySCF to ~0.05 mHa at cutoff 6
    (measured 2026-06-11; legacy gauge sat 24 mHa off)."""
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    r = run_pbc_bipole_rks(
        sysp,
        basis,
        monkhorst_pack(sysp, [1, 1, 1]),
        _rks_opts("svwn"),
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        progress=False,
    )
    assert r.converged
    assert r.exchange_ewald_split is True
    assert r.e_ext_el_spheropole is None
    assert r.energy == pytest.approx(E_PYSCF_H2_SVWN, abs=5e-4)


def test_rks_h2_pbe0_hybrid_split_matches_pyscf():
    """Hybrid path: the α_HF-scaled Ewald exchange split (K_SR fused +
    K_LR + Madelung SDS) matches PySCF PBE0 to ~0.08 mHa at cutoff 6
    (measured 2026-06-11). PBE0 is the discriminator with no B3LYP-style
    composite-definition ambiguity."""
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    r = run_pbc_bipole_rks(
        sysp,
        basis,
        monkhorst_pack(sysp, [1, 1, 1]),
        _rks_opts("pbe0"),
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        progress=False,
    )
    assert r.converged
    assert r.exchange_ewald_split is True
    assert r.energy == pytest.approx(E_PYSCF_H2_PBE0, abs=5e-4)


def test_run_periodic_job_bipole_rks_h2o_defaults_use_periodic_ewald(tmp_path):
    """A home-only short-range ball remains periodic with the reciprocal split."""
    from vibeqc.periodic_runner import run_periodic_job

    sysp = _h2o_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "6-31g")
    out_stem = tmp_path / "h2o-pbe0-bipole"
    result = run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="pbe0",
        jk_method="bipole",
        kpoints=(1, 1, 1),
        output=out_stem,
        output_qvf=False,
        citations=False,
        max_iter=100,
        conv_tol_energy=1e-9,
        initial_guess="SAD",
        convergence="off",
        damping=0.0,
        fock_mixing=0.0,
        level_shift=0.0,
        use_exchange_ewald_split=True,
        exchange_exxdiv="ewald",
        ewald_precision=1e-8,
        write_density=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        record_hostname=False,
        progress=False,
    )

    assert result.converged
    assert result.exchange_ewald_split is True
    assert result.energy == pytest.approx(-76.30241760934, abs=1e-8)
    assert result.energy == pytest.approx(E_PYSCF_H2O_PBE0_631G_GAMMA, abs=3e-3)
    assert out_stem.with_suffix(".system").exists()
    assert out_stem.with_suffix(".out").exists()


def test_bipole_rks_h2o_pbe0_exact_operator_tracks_pyscf():
    """The direct driver retains the #478 fixed point and #514 exact tail."""
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    sysp = _h2o_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "6-31g")
    opts = _rks_opts("pbe0", cutoff=15.0)
    opts.max_iter = 100
    opts.conv_tol_energy = 1e-9
    opts.damping = 0.0
    opts.fock_mixing = 0.0
    opts.level_shift = 0.0

    result = run_pbc_bipole_rks(
        sysp,
        basis,
        monkhorst_pack(sysp, [1, 1, 1]),
        opts,
        functional="pbe0",
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        use_exchange_ewald_split=True,
        exchange_exxdiv="ewald",
        sr_image_precision=1e-6,
        progress=False,
    )

    assert result.converged
    assert result.energy == pytest.approx(E_PYSCF_H2O_PBE0_631G_GAMMA, abs=3e-3)
    # #674: on this 18-bohr box at the 15-bohr cutoff the corrected split's
    # default alpha is the erfc bound 0.286 (CRYSTAL's 0.156 left the image
    # exchange beyond 15 bohr out), -76.30237318987 -> -76.30241760934; an
    # explicit ewald_omega=0.5 gives -76.30241762749 (split-invariant), and
    # the distance to PySCF falls from +6.09e-5 to +1.65e-5 Ha.
    assert result.energy == pytest.approx(-76.30241760934, abs=1e-8)


def test_rks_legacy_gauge_optout_keeps_spheropole():
    """use_exchange_ewald_split=False keeps the legacy RKS gauge
    (projection + spheropole) with the default M5 padded J_SR domain.

    The fixed-point pin uses an immutable overlap lattice.  Before #102,
    the Gamma XC adapter replaced ``S_lat`` with the density after the first
    iteration and shifted this energy by +18.3 mHa.
    """
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    r = run_pbc_bipole_rks(
        sysp,
        basis,
        monkhorst_pack(sysp, [1, 1, 1]),
        _rks_opts("svwn"),
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
        progress=False,
    )
    assert r.converged
    assert r.exchange_ewald_split is False
    assert r.sr_image_extent_bohr is not None
    assert r.e_ext_el_spheropole is not None
    # Restoring only the historical grid partition recovers the old native
    # pin (-1.1156375078921 Ha) within 1.6e-10 Ha, with the same spheropole.
    # The point-centered partition review is in docs/bipole_erfc_resume.md.
    assert r.e_ext_el_spheropole == pytest.approx(
        0.005559511758728281, abs=1e-11, rel=0.0
    )
    assert r.energy == pytest.approx(-1.1156334486137565, abs=1e-9, rel=0.0)


def test_rks_multik_split_rejects_adhoc_kmesh():
    """Multi-k split (all four drivers) needs MP mesh metadata; the RKS
    driver rejects ad-hoc k-lists like the RHF one."""
    from vibeqc._vibeqc_core import bloch_kmesh_from_lists
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    mp = monkhorst_pack(sysp, [2, 1, 1])
    adhoc = bloch_kmesh_from_lists(
        [np.asarray(k, dtype=float) for k in mp.kpoints],
        [float(w) for w in mp.weights],
    )
    with pytest.raises(ValueError, match="Monkhorst-Pack"):
        run_pbc_bipole_rks(
            sysp,
            basis,
            adhoc,
            _rks_opts("svwn"),
            use_exchange_ewald_split=True,
            ewald_precision=1e-8,
            progress=False,
        )


def test_rks_routes_through_shared_restricted_fock_builder(monkeypatch):
    """M2b: RKS delegates its two-electron Fock build to pbc_bipole_fock."""
    from vibeqc._vibeqc_core import PeriodicKSOptions, make_lattice_matrix_set
    from vibeqc.pbc_bipole_fock import BipoleRestrictedFockBuild
    import vibeqc.pbc_bipole_rks as rks_mod

    calls = []

    def fake_builder(
        ctx,
        density,
        *,
        coeffs_for_rho,
        alpha_hf,
        exchange_assembly=None,
        use_incremental=True,
        reseed_incremental=False,
    ):
        calls.append(
            {
                "alpha_hf": float(alpha_hf),
                "exact_j_for_pure_rks": bool(ctx.exact_j_for_pure_rks),
                "n_cells": len(density.cells),
                "use_incremental": bool(use_incremental),
                "is_screened": bool(
                    exchange_assembly is not None
                    and exchange_assembly.is_screened
                ),
            }
        )
        zero_blocks = [
            np.zeros_like(np.asarray(block, dtype=float))
            for block in density.blocks
        ]
        return BipoleRestrictedFockBuild(
            f2e_real=make_lattice_matrix_set(
                int(density.nbf),
                list(density.cells),
                zero_blocks,
            ),
            k_corr_per_k=None,
        )

    monkeypatch.setattr(rks_mod, "build_bipole_restricted_fock", fake_builder)

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    # Exact-FT retirement (2026-07-18): pure functionals default onto the
    # padded SR+LR composition (exact_j_for_pure_rks=False); the
    # analytic-FT builder engages only via the explicit use_exact_ft_j
    # cross-check oracle.
    expected = {
        "svwn": (0.0, False, False),
        "pbe0": (0.25, False, False),
        "svwn-oracle": (0.0, True, True),
    }
    for key, (alpha_hf, exact_j, use_exact_ft_j) in expected.items():
        functional = key.split("-")[0]
        opts = PeriodicKSOptions()
        opts.functional = functional
        opts.lattice_opts.cutoff_bohr = 4.0
        opts.lattice_opts.nuclear_cutoff_bohr = 4.0
        opts.max_iter = 1
        opts.use_diis = False
        opts.initial_guess = InitialGuess.SAD
        start = len(calls)
        rks_mod.run_pbc_bipole_rks(
            sysp,
            basis,
            monkhorst_pack(sysp, [1, 1, 1]),
            opts,
            use_ewald_j_split=True,
            use_exchange_ewald_split=True,
            use_exact_ft_j=use_exact_ft_j,
            ewald_precision=1e-6,
            progress=False,
        )
        new_calls = calls[start:]
        assert new_calls, functional
        assert {call["alpha_hf"] for call in new_calls} == {alpha_hf}
        assert {call["exact_j_for_pure_rks"] for call in new_calls} == {exact_j}
        assert all(call["n_cells"] > 0 for call in new_calls)
        # Iteration builds retain the incremental path, while the post-loop
        # consistency pass deliberately performs one full build on the exact
        # density returned to callers.
        assert new_calls[0]["use_incremental"]
        assert not new_calls[-1]["use_incremental"]
        # Neither functional is a screened hybrid.
        assert not any(call["is_screened"] for call in new_calls)


# ---------------------------------------------------------------------------
# UHF/UKS Γ propagation (option (b) Phase 4b, 2026-06-11)
# ---------------------------------------------------------------------------
# PySCF 2.13 pbc UHF/UKS(cell).density_fit() references, H₂/STO-3G
# triplet (spin=2) 12-bohr box, Γ, conv_tol 1e-10 (out-of-process):
E_PYSCF_H2_TRIPLET_UHF = -0.5358505084
E_PYSCF_H2_TRIPLET_UKS_SVWN = -0.4840745700
E_PYSCF_H2_TRIPLET_UKS_PBE0 = -0.5420559941


def _h2_triplet_box(L: float = 12.0, sep: float = 1.4):
    half = 0.5 * sep
    return vq.PeriodicSystem(
        3,
        np.diag([L, L, L]),
        [vq.Atom(1, [0, 0, -half]), vq.Atom(1, [0, 0, half])],
        multiplicity=3,
    )


def test_uhf_h2_triplet_gamma_matches_pyscf():
    """Per-spin Ewald exchange split (UHF): +0.07 mHa from PySCF at
    cutoff 6 (measured 2026-06-11; legacy gauge sat +7.1 mHa off)."""
    from vibeqc._vibeqc_core import PeriodicRHFOptions
    from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

    sysp = _h2_triplet_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.lattice_opts.nuclear_cutoff_bohr = 6.0
    opts.max_iter = 40
    opts.use_diis = True
    opts.initial_guess = InitialGuess.SAD
    r = run_pbc_bipole_uhf(
        sysp,
        basis,
        monkhorst_pack(sysp, [1, 1, 1]),
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        progress=False,
    )
    assert r.converged
    assert r.exchange_ewald_split is True
    assert r.e_ext_el_spheropole is None
    assert r.s_squared == pytest.approx(2.0, abs=1e-8)
    assert r.energy == pytest.approx(E_PYSCF_H2_TRIPLET_UHF, abs=5e-4)


def test_uks_h2_triplet_gamma_matches_pyscf_svwn_and_pbe0():
    """UKS under the corrected gauge: SVWN +1.0 mHa / PBE0 +0.8 mHa from
    PySCF at cutoff 6 (measured 2026-06-11). The PBE0 pin also certifies
    the per-spin hybrid exchange factor fix: the pre-2026-06-11 code
    applied −½α·K_σ to PER-SPIN densities (the ½ belongs to the
    closed-shell doubled-total-density convention), i.e. UKS hybrids ran
    with α/2 exchange in any gauge."""
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    sysp = _h2_triplet_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    for fn, ref in (
        ("svwn", E_PYSCF_H2_TRIPLET_UKS_SVWN),
        ("pbe0", E_PYSCF_H2_TRIPLET_UKS_PBE0),
    ):
        r = run_pbc_bipole_uks(
            sysp,
            basis,
            monkhorst_pack(sysp, [1, 1, 1]),
            _rks_opts(fn),
            use_ewald_j_split=True,
            ewald_precision=1e-8,
            progress=False,
        )
        assert r.converged, fn
        assert r.exchange_ewald_split is True
        assert r.e_ext_el_spheropole is None
        assert r.energy == pytest.approx(ref, abs=2e-3), fn


@pytest.mark.slow
def test_exchange_split_historical_domain_omega_drift_pinned():
    """Pin the pre-M5 SR/LR bookkeeping diagnostic across Ewald ω.

    The repaired padded SR domain intentionally exposes the separately
    diagnosed finite-ke reciprocal core tail (the M5 split-gap test pins it
    directly), so it is not an ω-invariance oracle at ke=200.  Keep this
    unpadded-domain redistribution check at the fold-safe cutoff 14.
    """
    sysp = _mgo_primitive()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha

    V = float(abs(np.linalg.det(np.asarray(sysp.lattice, dtype=float))))
    w0 = crystal_default_ewald_alpha(V)

    # Fixed density: converge once at ω₀, then 1-iter warm starts.
    r0 = run_pbc_bipole_rhf(
        sysp,
        basis,
        kmesh,
        _rhf_opts(14.0),
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        sr_image_precision=None,
        use_fock_symmetry_reduce=False,
        progress=False,
    )
    assert r0.converged
    blocks = [
        np.asarray(r0.density.blocks[i], dtype=float)
        for i in range(len(r0.density.cells))
    ]
    totals = []
    for w in (w0, 1.3 * w0):
        opts = _rhf_opts(14.0, max_iter=1)
        opts.use_diis = False
        rw = run_pbc_bipole_rhf(
            sysp,
            basis,
            kmesh,
            opts,
            use_ewald_j_split=True,
            ewald_omega=w,
            ewald_precision=1e-8,
            sr_image_precision=None,
            use_fock_symmetry_reduce=False,
            progress=False,
            initial_density=blocks,
        )
        totals.append(float(rw.energy_components[0].e_total))
    assert totals[0] == pytest.approx(totals[1], abs=OMEGA_INVARIANCE_TOL)


# ---------------------------------------------------------------------------
# Multi-k q≠0 LR-exchange channels (option (b) Phase 3, 2026-06-11)
# ---------------------------------------------------------------------------
# PySCF 2.13 references (out-of-process per CLAUDE.md §10; generator
# script /tmp-pattern recorded in examples/regression/bipole_parity/):
# tools.pbc.madelung(cell, kpts) and KRHF(cell, kpts).density_fit()
# with exxdiv='ewald', conv_tol 1e-10 on the 12-bohr H₂ box.
XI_PYSCF_H2_BOX12_211 = 0.150486817543
XI_PYSCF_H2_BOX12_222 = 0.118220728315
XI_PYSCF_MGO_222 = 0.288147805800
E_PYSCF_H2_12BOHR_KRHF_211 = -1.1201398824


def _h2_jx_caches(cutoff: float = 6.0, omega: float = 0.2333):
    from vibeqc._vibeqc_core import LatticeSumOptions, compute_overlap_lattice
    from vibeqc.bipole_fock_ewald import (
        _build_j_long_range_cache,
        build_k_exchange_long_range_cache,
    )

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    lat = LatticeSumOptions()
    lat.cutoff_bohr = cutoff
    lat.nuclear_cutoff_bohr = cutoff
    S_lat = compute_overlap_lattice(basis, sysp, lat)
    cells_r_cart = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in S_lat.cells], dtype=float
    )
    j_cache = _build_j_long_range_cache(basis, sysp, cells_r_cart, omega, 1e-6)
    K_max = float(np.linalg.norm(j_cache.K_vectors, axis=1).max()) + 1e-9
    x_cache = build_k_exchange_long_range_cache(
        basis, sysp, j_cache, K_max=K_max
    )
    return sysp, basis, j_cache, x_cache


def test_k_long_range_at_k_gamma_channel_matches_gamma_function():
    """At n_k = 1 the multi-k entry point routes through the q ≡ 0
    channel, which shares the J-cache tables — machine-precision
    agreement with compute_K_long_range_gamma."""
    from vibeqc.bipole_fock_ewald import compute_K_long_range_at_k

    _, basis, j_cache, x_cache = _h2_jx_caches()
    rng = np.random.default_rng(11)
    A = rng.standard_normal((basis.nbasis, basis.nbasis))
    D = A @ A.T
    K_gamma = compute_K_long_range_gamma(j_cache, D)
    K_multik = compute_K_long_range_at_k(
        x_cache,
        np.zeros(3),
        [np.zeros(3)],
        [1.0],
        [D.astype(complex)],
    )
    assert np.abs(np.real(K_multik) - K_gamma).max() < 1e-13
    assert np.abs(np.imag(K_multik)).max() < 1e-13


def test_shifted_reciprocal_vectors_properties():
    from vibeqc.bipole_ext_el_pole import compute_reciprocal_lattice_vectors
    from vibeqc.bipole_fock_ewald import (
        compute_shifted_reciprocal_lattice_vectors,
    )

    sysp = _h2_box()
    K_max = 2.0
    b = 2.0 * np.pi * np.linalg.inv(np.asarray(sysp.lattice, dtype=float)).T

    def _sorted_set(K):
        return np.array(sorted(map(tuple, np.round(K, 9))))

    # q = 0 reduces to the unshifted enumeration (K = 0 excluded).
    K0 = compute_shifted_reciprocal_lattice_vectors(sysp, np.zeros(3), K_max)
    K_ref = compute_reciprocal_lattice_vectors(sysp, K_max)
    assert np.array_equal(_sorted_set(K0), _sorted_set(K_ref))

    # A genuine q ≠ 0 channel: the [2,1,1] mesh difference q = ½·b₁.
    q = 0.5 * b[:, 0]
    Kq = compute_shifted_reciprocal_lattice_vectors(sysp, q, K_max)
    norms = np.linalg.norm(Kq, axis=1)
    assert norms.min() > 1e-6 and norms.max() <= K_max + 1e-12
    # The G = 0 member (the vector q itself) is included.
    assert np.abs(Kq - q[None, :]).sum(axis=1).min() < 1e-9

    # Set-invariance under q → q + G₀ (channel identity mod G).
    Kq_shift = compute_shifted_reciprocal_lattice_vectors(
        sysp,
        q + b[:, 1],
        K_max,
    )
    assert np.array_equal(_sorted_set(Kq), _sorted_set(Kq_shift))


def test_shifted_reciprocal_vectors_and_q_canonicalisation_on_skew_cell():
    from vibeqc.bipole_fock_ewald import (
        KExchangeLongRangeCache,
        compute_shifted_reciprocal_lattice_vectors,
    )

    lattice = np.array(
        [
            [7.0, 0.4, 0.2],
            [0.3, 8.0, 0.5],
            [0.1, 0.6, 9.0],
        ]
    )
    sysp = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(2, [0.0, 0.0, 0.0])],
    )
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    fractional_q = np.array([0.61, -0.72, 0.20])
    q = reciprocal @ fractional_q
    K_max = 2.0

    actual = compute_shifted_reciprocal_lattice_vectors(sysp, q, K_max)
    grid = np.arange(-8, 9)
    n1, n2, n3 = np.meshgrid(grid, grid, grid, indexing="ij")
    idx = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=-1)
    oracle = q[None, :] + idx @ reciprocal.T
    oracle_norm_sq = np.einsum("ij,ij->i", oracle, oracle)
    oracle = oracle[
        (oracle_norm_sq > 1e-12) & (oracle_norm_sq <= K_max**2)
    ]

    def _sorted_set(vectors):
        return np.array(sorted(map(tuple, np.round(vectors, 12))))

    assert np.array_equal(_sorted_set(actual), _sorted_set(oracle))
    shifted = compute_shifted_reciprocal_lattice_vectors(
        sysp,
        q + reciprocal[:, 1],
        K_max,
    )
    assert np.array_equal(_sorted_set(actual), _sorted_set(shifted))

    cache = KExchangeLongRangeCache(
        basis=None,
        system=sysp,
        cells_r_cart=np.empty((0, 3)),
        omega=0.3,
        K_max=K_max,
        j_cache=None,
    )
    key, q_canonical = cache._canonical_q(q)
    shifted_key, shifted_canonical = cache._canonical_q(q + reciprocal[:, 1])
    wrapped = np.array([-0.39, 0.28, 0.20])
    assert key == shifted_key
    assert q_canonical == pytest.approx(reciprocal @ wrapped, abs=1e-12)
    assert shifted_canonical == pytest.approx(q_canonical, abs=1e-12)


def test_probe_charge_madelung_supercell_pyscf_pins():
    from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell

    h2 = _h2_box()
    assert probe_charge_madelung_supercell(h2, (2, 1, 1)) == pytest.approx(
        XI_PYSCF_H2_BOX12_211, abs=1e-9
    )
    assert probe_charge_madelung_supercell(h2, (2, 2, 2)) == pytest.approx(
        XI_PYSCF_H2_BOX12_222, abs=1e-9
    )
    mgo = _mgo_primitive()
    assert probe_charge_madelung_supercell(mgo, (2, 2, 2)) == pytest.approx(
        XI_PYSCF_MGO_222, abs=1e-9
    )
    # mesh (1,1,1) is the unit-cell constant.
    assert probe_charge_madelung_supercell(mgo, (1, 1, 1)) == pytest.approx(
        probe_charge_madelung(mgo), abs=1e-12
    )


def test_bvk_torus_density_matrices_inverts_bloch_fold():
    """Synthesize TR-consistent per-k densities on the actual mesh,
    store the real-space fold D(g) = Σ_k w_k·Re[e^{−ik·R_g}·D(k)] (the
    exact real_space_density_from_kpoints convention), and check the
    torus fold recovers every D(k) to machine precision.

    vibe-qc's monkhorst_pack meshes are Γ-centered ({m/n}·b), so a
    [4,1,1] mesh carries both self-paired points (k ≡ −k mod G: Γ and
    the zone boundary ½·b₁, where a physical TR-symmetric density is
    real) and a genuine ± pair (¼·b₁ ↔ ¾·b₁ ≡ −¼·b₁, where
    D(−k) = D(k)*). The fold must be exact for both."""
    from vibeqc._vibeqc_core import LatticeSumOptions, compute_overlap_lattice
    from vibeqc.pbc_bipole_common import bvk_torus_density_matrices

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    mesh = (4, 1, 1)
    mp = monkhorst_pack(sysp, list(mesh))
    k_points = [np.asarray(k, dtype=float) for k in mp.kpoints]
    weights = np.asarray(mp.weights, dtype=float)
    nbf = basis.nbasis
    a = np.asarray(sysp.lattice, dtype=float)

    def _frac(k):
        return np.round((a.T @ k) / (2.0 * np.pi), 9)

    # Pair indices under time reversal mod G.
    partner = {}
    for i, k in enumerate(k_points):
        fi = _frac(-k) % 1.0
        matches = [
            j
            for j, kj in enumerate(k_points)
            if np.allclose(_frac(kj) % 1.0, fi % 1.0, atol=1e-9)
        ]
        assert len(matches) == 1, "mesh must be TR-symmetric mod G"
        partner[i] = matches[0]

    rng = np.random.default_rng(23)
    D_k_ref: list = [None] * len(k_points)
    n_self_paired = 0
    n_pairs = 0
    for i in range(len(k_points)):
        if D_k_ref[i] is not None:
            continue
        j = partner[i]
        if j == i:
            A = rng.standard_normal((nbf, nbf))
            D_k_ref[i] = (A + A.T).astype(complex)  # real symmetric
            n_self_paired += 1
        else:
            M = rng.standard_normal((nbf, nbf)) + 1j * rng.standard_normal(
                (nbf, nbf)
            )
            D = M @ M.conj().T  # Hermitian with genuine imag part
            D_k_ref[i] = D
            D_k_ref[j] = D.conj()
            n_pairs += 1
    assert n_self_paired >= 2 and n_pairs >= 1  # both cases exercised

    lat = LatticeSumOptions()
    # The [4,1,1] torus spans cells out to (−2,0,0)·a₁ = 24 bohr.
    lat.cutoff_bohr = 26.0
    lat.nuclear_cutoff_bohr = 26.0
    density = compute_overlap_lattice(basis, sysp, lat)
    for g_idx in range(len(density.cells)):
        R_g = np.asarray(density.cells[g_idx].r_cart, dtype=float)
        block = np.zeros((nbf, nbf), dtype=float)
        for k_idx, k in enumerate(k_points):
            phase = np.exp(-1j * float(np.dot(k, R_g)))
            block += float(weights[k_idx]) * np.real(phase * D_k_ref[k_idx])
        density.set_block(g_idx, block)

    recovered = bvk_torus_density_matrices(density, k_points, mesh)
    for D_rec, D_ref in zip(recovered, D_k_ref):
        assert np.abs(D_rec - D_ref).max() < 1e-12


def test_exchange_split_multik_h2_211_matches_pyscf_krhf():
    """Integration: the corrected gauge at [2,1,1] on the 12-bohr H₂
    box against out-of-process PySCF KRHF GDF (exxdiv='ewald'). Cutoff
    7 (wide density list radius 14 covers the BvK torus); residual
    +0.121 mHa = real-space truncation (cutoff 12 lands −2.2 µHa)."""
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _rhf_opts(7.0, max_iter=40)
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    r = run_pbc_bipole_rhf(
        sysp,
        basis,
        monkhorst_pack(sysp, [2, 1, 1]),
        opts,
        use_exchange_ewald_split=True,
        ewald_precision=1e-6,
        progress=False,
    )
    assert r.converged
    assert r.exchange_ewald_split is True
    assert r.e_ext_el_spheropole is None
    assert r.energy == pytest.approx(E_PYSCF_H2_12BOHR_KRHF_211, abs=5e-4)


def test_exchange_split_multik_matches_explicit_supercell_gamma():
    """The strongest internal check of the q ≠ 0 channels: a [2,1,1]
    mesh on the unit cell IS the Γ-point SCF of the doubled supercell,
    exactly unfolded. Run both with the SAME Ewald ω (the supercell
    would otherwise auto-derive a different split) and compare energy
    per unit cell. Residual = the volume-derived K_max envelope
    difference + real-space truncation asymmetry.  M5 uses the same generous
    explicit absolute radius in both representations so the test remains an
    identity check rather than comparing two precision-derived ball sizes.
    #704/D131 removes the historical +44.572 microhartree/cell nuclear
    residual by covering displaced Gaussian products in the V_ne image sum.
    Pin the restored energy identity, not the former known-defect magnitude;
    retain the independent two-electron and exchange component controls.
    """
    from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    V = float(abs(np.linalg.det(np.asarray(sysp.lattice, dtype=float))))
    omega = crystal_default_ewald_alpha(V)

    opts = _rhf_opts(12.0, max_iter=40)
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    r_multik = run_pbc_bipole_rhf(
        sysp,
        basis,
        monkhorst_pack(sysp, [2, 1, 1]),
        opts,
        use_exchange_ewald_split=True,
        ewald_omega=omega,
        ewald_precision=1e-8,
        sr_image_extent_bohr=40.0,
        progress=False,
    )
    assert r_multik.converged

    L = 12.0
    half = 0.7
    super_sys = vq.PeriodicSystem(
        3,
        np.diag([2.0 * L, L, L]),
        [
            vq.Atom(1, [0, 0, -half]),
            vq.Atom(1, [0, 0, half]),
            vq.Atom(1, [L, 0, -half]),
            vq.Atom(1, [L, 0, half]),
        ],
    )
    super_basis = vq.BasisSet(super_sys.unit_cell_molecule(), "sto-3g")
    opts_sc = _rhf_opts(12.0, max_iter=40)
    opts_sc.conv_tol_energy = 1e-9
    opts_sc.conv_tol_grad = 1e-7
    r_super = run_pbc_bipole_rhf(
        super_sys,
        super_basis,
        monkhorst_pack(super_sys, [1, 1, 1]),
        opts_sc,
        use_exchange_ewald_split=True,
        ewald_omega=omega,
        ewald_precision=1e-8,
        sr_image_extent_bohr=40.0,
        progress=False,
    )
    assert r_super.converged
    residual = r_super.energy / 2.0 - r_multik.energy
    assert abs(residual) < 1e-8

    comp_multik = r_multik.energy_components[-1]
    comp_super = r_super.energy_components[-1]
    assert comp_super.e_two_electron / 2.0 == pytest.approx(
        comp_multik.e_two_electron, abs=1e-7
    )
    assert comp_super.e_exchange / 2.0 == pytest.approx(
        comp_multik.e_exchange, abs=1e-7
    )


# PySCF 2.13 KRKS/KUHF/KUKS multi-k references (out-of-process; same
# generator script). OPEN-SHELL TRAP: PySCF interprets cell.spin at
# multi-k as the TOTAL spin imbalance over the BvK supercell — KUHF
# with spin=2 at [2,1,1] fills nelec=(3,1), a mixed state, NOT the
# uniform triplet lattice. The references below pin the per-cell
# triplet by setting mf.nelec = (4, 0) explicitly.
E_PYSCF_H2_KRKS_SVWN_211 = -1.1212703585
E_PYSCF_H2_KRKS_PBE0_211 = -1.1552192298
E_PYSCF_H2_KUHF_TRIPLET_211 = -0.5337291457
E_PYSCF_H2_KUHF_TRIPLET_GAMMA = -0.5358505084
E_PYSCF_H2_KUKS_PBE0_TRIPLET_211 = -0.5415218581
E_PYSCF_H2_KUKS_PBE0_TRIPLET_GAMMA = -0.5420559972


def _ks_opts(cutoff: float, functional: str, max_iter: int = 60):
    from vibeqc._vibeqc_core import PeriodicKSOptions

    opts = PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = cutoff
    opts.lattice_opts.nuclear_cutoff_bohr = cutoff
    opts.max_iter = max_iter
    opts.use_diis = True
    opts.initial_guess = InitialGuess.SAD
    opts.functional = functional
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    return opts


def test_exchange_split_multik_rks_matches_pyscf_krks():
    """RKS multi-k split: pure SVWN exercises the gauge change + torus
    fold + ρ̂(K); PBE0 exercises the α_HF-scaled q≠0 channels +
    supercell correction. Both land at truncation scale from
    out-of-process PySCF KRKS GDF (measured +0.043 / +0.062 mHa at
    cutoff 7, 2026-06-12)."""
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    for functional, ref in (
        ("svwn", E_PYSCF_H2_KRKS_SVWN_211),
        ("pbe0", E_PYSCF_H2_KRKS_PBE0_211),
    ):
        r = run_pbc_bipole_rks(
            sysp,
            basis,
            monkhorst_pack(sysp, [2, 1, 1]),
            _ks_opts(7.0, functional),
            use_exchange_ewald_split=True,
            ewald_precision=1e-6,
            progress=False,
        )
        assert r.converged, functional
        assert r.exchange_ewald_split is True
        assert r.energy == pytest.approx(ref, abs=5e-4), functional


def test_exchange_split_multik_uhf_triplet_matches_pyscf_kuhf():
    """UHF multi-k split on the uniform triplet lattice: absolute
    +0.034 mHa at cutoff 7, and the k-sampling SHIFT Γ→[2,1,1]
    matches PySCF KUHF to 0.04 mHa (measured +2.084 vs +2.121 mHa,
    2026-06-12) — the q≠0 channels + supercell-ξ compensation.
    GEOMETRY NOTE: the PySCF refs are at exactly 1.4 bohr (the
    fixture geometry); on the steep triplet curve a 0.74-Å (1.39839
    bohr) cell shifts E by ~1 mHa."""
    from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

    sysp = _h2_triplet_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    energies = {}
    for label, mesh in (("gamma", [1, 1, 1]), ("211", [2, 1, 1])):
        opts = _rhf_opts(7.0, max_iter=50)
        opts.conv_tol_energy = 1e-9
        opts.conv_tol_grad = 1e-7
        r = run_pbc_bipole_uhf(
            sysp,
            basis,
            monkhorst_pack(sysp, mesh),
            opts,
            use_exchange_ewald_split=True,
            ewald_precision=1e-6,
            progress=False,
        )
        assert r.converged, label
        energies[label] = float(r.energy)
    assert energies["211"] == pytest.approx(
        E_PYSCF_H2_KUHF_TRIPLET_211, abs=5e-4
    )
    shift_pyscf = E_PYSCF_H2_KUHF_TRIPLET_211 - E_PYSCF_H2_KUHF_TRIPLET_GAMMA
    shift_bipole = energies["211"] - energies["gamma"]
    assert shift_bipole == pytest.approx(shift_pyscf, abs=5e-4)


def test_exchange_split_multik_uks_pbe0_matches_pyscf_kuks():
    """UKS multi-k split (per-spin α_HF-scaled channels) on the
    uniform PBE0 triplet lattice: absolute +0.77 mHa at cutoff 7 (the
    documented Γ grid/truncation offset — mesh-independent) and
    k-sampling shift agreement to 0.013 mHa vs PySCF KUKS
    (+0.521 vs +0.534 mHa, 2026-06-12)."""
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    sysp = _h2_triplet_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    energies = {}
    for label, mesh in (("gamma", [1, 1, 1]), ("211", [2, 1, 1])):
        r = run_pbc_bipole_uks(
            sysp,
            basis,
            monkhorst_pack(sysp, mesh),
            _ks_opts(7.0, "pbe0"),
            use_exchange_ewald_split=True,
            ewald_precision=1e-6,
            progress=False,
        )
        assert r.converged, label
        energies[label] = float(r.energy)
    assert energies["211"] == pytest.approx(
        E_PYSCF_H2_KUKS_PBE0_TRIPLET_211, abs=2e-3
    )
    shift_pyscf = (
        E_PYSCF_H2_KUKS_PBE0_TRIPLET_211 - E_PYSCF_H2_KUKS_PBE0_TRIPLET_GAMMA
    )
    shift_bipole = energies["211"] - energies["gamma"]
    assert shift_bipole == pytest.approx(shift_pyscf, abs=3e-4)


def test_bvk_torus_fold_uses_minimal_norm_residue_reps_on_fcc():
    """On non-orthogonal cells the residue-class representatives must
    be chosen by minimal |R|, not by index box: for MgO-shaped fcc at
    [2,2,2] the naive box rep (−1,−1,−1) sits at |a₁+a₂+a₃| = a√3 ≈
    13.8 bohr while the same class is covered by (−1,1,1) ↔ (a,0,0)
    at 7.96 bohr. At cutoff 9 the fold must succeed and invert the
    Bloch transform exactly (any complete residue system is exact)."""
    from vibeqc._vibeqc_core import LatticeSumOptions, compute_overlap_lattice
    from vibeqc.pbc_bipole_common import bvk_torus_density_matrices

    sysp = _mgo_primitive()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    mesh = (2, 2, 2)
    mp = monkhorst_pack(sysp, list(mesh))
    k_points = [np.asarray(k, dtype=float) for k in mp.kpoints]
    weights = np.asarray(mp.weights, dtype=float)
    nbf = basis.nbasis

    # Γ-centered [2,2,2]: every point is TR-self-paired mod G, so all
    # physical densities are real symmetric.
    rng = np.random.default_rng(31)
    D_k_ref = []
    for _ in k_points:
        A = rng.standard_normal((nbf, nbf))
        D_k_ref.append((A + A.T).astype(complex))

    lat = LatticeSumOptions()
    lat.cutoff_bohr = 9.0  # < a√3 ≈ 13.8: the naive index box FAILS here
    lat.nuclear_cutoff_bohr = 9.0
    density = compute_overlap_lattice(basis, sysp, lat)
    for g_idx in range(len(density.cells)):
        R_g = np.asarray(density.cells[g_idx].r_cart, dtype=float)
        block = np.zeros((nbf, nbf), dtype=float)
        for k_idx, k in enumerate(k_points):
            phase = np.exp(-1j * float(np.dot(k, R_g)))
            block += float(weights[k_idx]) * np.real(phase * D_k_ref[k_idx])
        density.set_block(g_idx, block)

    recovered = bvk_torus_density_matrices(density, k_points, mesh)
    for D_rec, D_ref in zip(recovered, D_k_ref):
        assert np.abs(D_rec - D_ref).max() < 1e-12


def test_s_fold_truncation_drift_k_aware():
    """k_points=None reproduces the historical Γ-only measure; a mesh
    takes the max over per-k fold drifts (the multi-k split contracts
    S(k) at every mesh point — zone-boundary folds truncate
    differently from Γ). A Γ-containing mesh can only raise the max."""
    from vibeqc._vibeqc_core import LatticeSumOptions
    from vibeqc.pbc_bipole_common import s_fold_truncation_drift

    sysp = _h2_box(L=8.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 6.0
    lat.nuclear_cutoff_bohr = 6.0

    d_default = s_fold_truncation_drift(basis, sysp, lat)
    d_gamma = s_fold_truncation_drift(
        basis, sysp, lat, k_points=[np.zeros(3)]
    )
    assert d_gamma == pytest.approx(d_default, abs=1e-15)

    mp = monkhorst_pack(sysp, [2, 1, 1])
    d_mesh = s_fold_truncation_drift(
        basis,
        sysp,
        lat,
        k_points=[np.asarray(k, dtype=float) for k in mp.kpoints],
    )
    assert d_mesh >= d_gamma - 1e-15


# ---------------------------------------------------------------------------
# Incremental / differential Fock (use_incremental_fock, 2026-06-14)
# ---------------------------------------------------------------------------
# F^2e_SR is linear in the density, so building J_SR/K_SR from the
# iter-to-iter ΔD and accumulating is energy-identical to the full build
# (the post-convergence rebuild is full in both; the per-iter screening
# drift is reset every N iters). The win is wall-time: the C++ builder's
# density-envelope screening cheapens build_jk(ΔD) as ΔD → 0.


def test_incremental_fock_matches_full_build_gamma():
    """use_incremental_fock=True is energy-identical to the full build (Γ)."""
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    e = {}
    for inc in (False, True):
        opts = _rhf_opts(4.0, max_iter=40)
        opts.conv_tol_energy = 1e-9
        opts.conv_tol_grad = 1e-7
        r = run_pbc_bipole_rhf(
            sysp,
            basis,
            monkhorst_pack(sysp, [1, 1, 1]),
            opts,
            use_exchange_ewald_split=True,
            use_incremental_fock=inc,
            ewald_precision=1e-6,
            progress=False,
        )
        assert r.converged
        e[inc] = r.energy
    assert e[True] == pytest.approx(e[False], abs=1e-9)


def test_incremental_fock_matches_full_build_multik():
    """Multi-k: incremental ≡ full through the BvK-torus fold + q-channels.
    Cutoff 7 (wide list 14) covers the [2,1,1] torus of the 12-bohr box."""
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    e = {}
    for inc in (False, True):
        opts = _rhf_opts(7.0, max_iter=40)
        opts.conv_tol_energy = 1e-9
        opts.conv_tol_grad = 1e-7
        r = run_pbc_bipole_rhf(
            sysp,
            basis,
            monkhorst_pack(sysp, [2, 1, 1]),
            opts,
            use_exchange_ewald_split=True,
            use_incremental_fock=inc,
            ewald_precision=1e-6,
            progress=False,
        )
        assert r.converged
        e[inc] = r.energy
    assert e[True] == pytest.approx(e[False], abs=1e-8)


def test_incremental_fock_reset_preserves_identity():
    """A short reset_every (forcing mid-SCF full rebuilds) still lands the
    same energy — the reset re-syncs the accumulator to the exact JK."""
    from vibeqc.bipole_fock_ewald import IncrementalJK

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _rhf_opts(4.0, max_iter=40)
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    r_full = run_pbc_bipole_rhf(
        sysp, basis, monkhorst_pack(sysp, [1, 1, 1]), opts,
        use_exchange_ewald_split=True, use_incremental_fock=False,
        ewald_precision=1e-6, progress=False,
    )
    # Monkeypatch the default reset_every to 1 (full rebuild every other
    # iter) via the dataclass default — exercises the reset path.
    orig = IncrementalJK.reset_every
    try:
        IncrementalJK.reset_every = 1
        r_inc = run_pbc_bipole_rhf(
            sysp, basis, monkhorst_pack(sysp, [1, 1, 1]), opts,
            use_exchange_ewald_split=True, use_incremental_fock=True,
            ewald_precision=1e-6, progress=False,
        )
    finally:
        IncrementalJK.reset_every = orig
    assert r_inc.converged
    assert r_inc.energy == pytest.approx(r_full.energy, abs=1e-9)


def test_incremental_fock_matches_full_build_rks():
    """RKS: pure-DFT (J-only ΔD) and hybrid (J+K ΔD) both ≡ full build."""
    from vibeqc._vibeqc_core import PeriodicKSOptions
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    for func in ("svwn", "pbe0"):
        e = {}
        for inc in (False, True):
            opts = PeriodicKSOptions()
            opts.functional = func
            opts.lattice_opts.cutoff_bohr = 4.0
            opts.lattice_opts.nuclear_cutoff_bohr = 4.0
            opts.max_iter = 50
            opts.use_diis = True
            opts.initial_guess = InitialGuess.SAD
            opts.conv_tol_energy = 1e-9
            opts.conv_tol_grad = 1e-7
            r = run_pbc_bipole_rks(
                sysp, basis, monkhorst_pack(sysp, [1, 1, 1]), opts,
                use_exchange_ewald_split=True, use_incremental_fock=inc,
                ewald_precision=1e-6, progress=False,
            )
            assert r.converged, func
            e[inc] = r.energy
        assert e[True] == pytest.approx(e[False], abs=1e-9), func


def test_incremental_fock_matches_full_build_uhf():
    """UHF: per-spin differential ΔD builds ≡ full build (triplet H2)."""
    from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

    sysp = _h2_triplet_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    e = {}
    for inc in (False, True):
        opts = _rhf_opts(4.0, max_iter=50)
        opts.conv_tol_energy = 1e-9
        opts.conv_tol_grad = 1e-7
        r = run_pbc_bipole_uhf(
            sysp, basis, monkhorst_pack(sysp, [1, 1, 1]), opts,
            use_exchange_ewald_split=True, use_incremental_fock=inc,
            ewald_precision=1e-6, progress=False,
        )
        assert r.converged
        e[inc] = r.energy
    assert e[True] == pytest.approx(e[False], abs=1e-9)


def test_incremental_fock_matches_full_build_uks():
    """UKS: pure-DFT (1 total accumulator) + hybrid (2 per-spin) ≡ full."""
    from vibeqc._vibeqc_core import PeriodicKSOptions
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    sysp = _h2_triplet_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    for func in ("svwn", "pbe0"):
        e = {}
        for inc in (False, True):
            opts = PeriodicKSOptions()
            opts.functional = func
            opts.lattice_opts.cutoff_bohr = 4.0
            opts.lattice_opts.nuclear_cutoff_bohr = 4.0
            opts.max_iter = 50
            opts.use_diis = True
            opts.initial_guess = InitialGuess.SAD
            opts.conv_tol_energy = 1e-9
            opts.conv_tol_grad = 1e-7
            r = run_pbc_bipole_uks(
                sysp, basis, monkhorst_pack(sysp, [1, 1, 1]), opts,
                use_exchange_ewald_split=True, use_incremental_fock=inc,
                ewald_precision=1e-6, progress=False,
            )
            assert r.converged, func
            e[inc] = r.energy
        assert e[True] == pytest.approx(e[False], abs=1e-9), func


# ---------------------------------------------------------------------------
# GitLab #116: the terminal check judges the operator the loop converged on.
# ---------------------------------------------------------------------------


def test_incremental_jk_seed_resyncs_the_chain_to_an_exact_build():
    """After ``seed`` the accumulator holds the exact build and the next
    ``build`` continues the ΔD chain from it (no full rebuild, ΔD = 0)."""
    from types import SimpleNamespace

    from vibeqc._vibeqc_core import make_lattice_matrix_set
    from vibeqc.bipole_fock_ewald import IncrementalJK

    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    nbf = int(basis.nbasis)
    from vibeqc._vibeqc_core import direct_lattice_cells

    cells = list(direct_lattice_cells(sysp, 4.0))
    rng = np.random.default_rng(116)

    def lattice(scale):
        blocks = []
        for _ in cells:
            b = rng.standard_normal((nbf, nbf))
            blocks.append(scale * (b + b.T))
        return make_lattice_matrix_set(nbf, cells, blocks)

    calls = []

    def build_jk(density):
        calls.append(float(max(np.max(np.abs(b)) for b in density.blocks)))
        return SimpleNamespace(
            J=make_lattice_matrix_set(nbf, cells, [2.0 * np.asarray(b) for b in density.blocks]),
            K=make_lattice_matrix_set(nbf, cells, [0.5 * np.asarray(b) for b in density.blocks]),
        )

    acc = IncrementalJK()
    d0 = lattice(1.0)
    acc.build(d0, build_jk)  # full build
    d1 = lattice(1.0)
    acc.build(d1, build_jk)  # incremental (ΔD = d1 - d0)
    assert len(calls) == 2
    # Poison the accumulated state the way screening error would.
    acc._J_acc[0][0, 0] += 1.0
    exact = build_jk(d1)
    acc.seed(d1, exact.J, exact.K)
    j2, k2 = acc.build(d1, build_jk)  # ΔD = 0 from the seeded exact state
    assert calls[-1] == pytest.approx(0.0)  # the increment build saw ΔD = 0
    for got, want in zip(j2.blocks, exact.J.blocks):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-14)
    for got, want in zip(k2.blocks, exact.K.blocks):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-14)
    # seed() copies: mutating the caller's lattice afterwards is harmless.
    exact.J.set_block(0, np.asarray(exact.J.blocks[0]) + 5.0)
    j3, _ = acc.build(d1, build_jk)
    np.testing.assert_allclose(np.asarray(j3.blocks[0]), np.asarray(j2.blocks[0]), atol=1e-14)


def _beh2_box_9():
    return vq.PeriodicSystem(
        3,
        9.0 * np.eye(3),
        [
            vq.Atom(4, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 2.45]),
            vq.Atom(1, [0.0, 0.0, -2.75]),
        ],
    )


def _run_beh2_with_incremental_drift(eps: float, log_path=None):
    """Corrected-gauge BeH2/STO-3G (2,1,1) at cutoff 13 with every incremental
    ΔD scaled by ``1 + eps`` inside :class:`IncrementalJK`, so the accumulated
    J_SR/K_SR drifts from the exact operator the way screening error does."""
    from vibeqc.bipole_fock_ewald import IncrementalJK
    from vibeqc.structured_log import structured_log

    orig_delta = IncrementalJK._delta

    def drifted_delta(self, density, make_lattice_matrix_set, _cell_key):
        d = orig_delta(self, density, make_lattice_matrix_set, _cell_key)
        if eps == 0.0:
            return d
        return make_lattice_matrix_set(
            int(d.nbf),
            list(d.cells),
            [(1.0 + eps) * np.asarray(b, dtype=float) for b in d.blocks],
        )

    sysp = _beh2_box_9()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [2, 1, 1], use_symmetry=False)
    opts = _rhf_opts(13.0, max_iter=40)
    opts.conv_tol_energy = 1e-8
    IncrementalJK._delta = drifted_delta
    try:
        if log_path is None:
            return run_pbc_bipole_rhf(
                sysp, basis, kmesh, opts,
                use_ewald_j_split=True, ewald_precision=1e-6, progress=False,
            )
        with structured_log(log_path, enabled=True):
            return run_pbc_bipole_rhf(
                sysp, basis, kmesh, opts,
                use_ewald_j_split=True, ewald_precision=1e-6, progress=False,
            )
    finally:
        IncrementalJK._delta = orig_delta


def test_terminal_check_converges_on_the_exact_operator_instead_of_revoking(tmp_path):
    """GitLab #116 on the laptop.

    Pre-fix, a 1e-6 relative drift per increment converged the loop at
    iteration 7 and the non-incremental terminal rebuild sat +2.6e-8 Ha
    above it (2.1e-8 in the field: validation-job, LiH (2,2,2)), so
    ``converged`` flipped to False after all the work was done. Now the
    in-loop confirmation judges the exact operator's energy, reseeds the
    incremental chain from that exact build and keeps iterating: the run
    converges, on the exact operator, at the undrifted energy.
    """
    import json

    clean = _run_beh2_with_incremental_drift(0.0)
    assert clean.converged
    log_path = tmp_path / "drift.jsonl"
    drifted = _run_beh2_with_incremental_drift(1e-6, log_path)
    assert drifted.converged, (
        "the terminal rebuild withdrew a converged SCF (#116): "
        f"final delta {drifted.scf_trace[-1].delta_e:+.3e} Ha"
    )
    assert drifted.n_iter > clean.n_iter  # the reseeded extra iteration(s)
    assert abs(drifted.scf_trace[-1].delta_e) < 1e-8
    assert drifted.scf_trace[-1].grad_norm < 1e-6
    assert drifted.energy == pytest.approx(clean.energy, abs=5e-8)
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    checks = [e for e in events if e.get("event") == "scf_terminal_check"]
    in_loop = [e for e in checks if e["phase"] == "in_loop"]
    post = [e for e in checks if e["phase"] == "post_loop"]
    # The drift was caught in the loop (one refused confirmation with the
    # measured delta), then confirmed, and the post-loop re-check agrees.
    assert any(not e["passed"] and abs(e["delta_e"]) > 1e-8 for e in in_loop)
    assert in_loop[-1]["passed"]
    assert len(post) == 1 and post[0]["passed"]
    assert post[0]["delta_e"] == pytest.approx(drifted.scf_trace[-1].delta_e, abs=1e-12)
    for e in checks:
        assert e["conv_tol_energy"] == pytest.approx(1e-8)
        assert "grad_norm_exact" in e and "grad_norm_loop" in e


def test_terminal_check_stays_silent_when_nothing_drifted(tmp_path):
    """An undrifted run confirms at the first attempt: one in-loop check,
    passed, and the post-loop check restates the same numbers."""
    import json

    log_path = tmp_path / "clean.jsonl"
    r = _run_beh2_with_incremental_drift(0.0, log_path)
    assert r.converged
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    checks = [e for e in events if e.get("event") == "scf_terminal_check"]
    assert [e["passed"] for e in checks] == [True, True]
    assert [e["phase"] for e in checks] == ["in_loop", "post_loop"]
    assert checks[0]["energy_exact"] == pytest.approx(checks[1]["energy_exact"], abs=1e-12)
